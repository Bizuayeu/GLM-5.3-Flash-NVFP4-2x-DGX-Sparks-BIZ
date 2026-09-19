import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup import server
from glm53_setup import server_config as config
from glm53_setup.runtime.memory_probe import (
    MemoryProbeWorker,
    summarize,
    summarize_host,
)

ROOT = Path(__file__).resolve().parents[1]


class MemoryProbeTests(unittest.TestCase):
    def test_summary_keeps_the_allocator_counters_that_explain_growth(self):
        stats = {
            "reserved_bytes.all.current": 10 * 2**30,
            "reserved_bytes.all.peak": 11 * 2**30,
            "allocated_bytes.all.current": 8 * 2**30,
            "active_bytes.all.current": 8 * 2**30,
            "inactive_split_bytes.all.current": 2**30,
            "segment.all.current": 40,
            "num_alloc_retries": 3,
            "num_ooms": 0,
            "num_device_alloc": 120,
            "num_device_free": 80,
        }
        row = summarize(stats, (5 * 2**30, 128 * 2**30))
        self.assertEqual(row["reserved_gib"], 10.0)
        self.assertEqual(row["allocated_gib"], 8.0)
        self.assertEqual(row["inactive_split_gib"], 1.0)
        self.assertEqual(row["segments"], 40)
        self.assertEqual(row["alloc_retries"], 3)
        self.assertEqual(row["device_allocs"], 120)
        self.assertEqual(row["device_frees"], 80)
        self.assertEqual(row["device_free_gib"], 5.0)
        # Absent counters read as zero rather than failing the sample.
        self.assertEqual(summarize({}, (0, 0))["reserved_gib"], 0.0)

    def test_worker_method_reads_torch_and_labels_the_rank(self):
        worker = MemoryProbeWorker()
        worker.rank = 1
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                memory_stats=lambda: {"reserved_bytes.all.current": 2**30},
                mem_get_info=lambda: (2**30, 4 * 2**30),
            )
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            row = worker.allocator_stats()
        self.assertEqual(row["rank"], 1)
        self.assertEqual(row["reserved_gib"], 1.0)
        self.assertEqual(row["device_total_gib"], 4.0)

    def test_host_summary_separates_used_retained_and_mapped_heap(self):
        heap = {"arena": 3 * 2**20, "hblkhd": 2 * 2**20, "uordblks": 2**20}
        heap["fordblks"] = 2 * 2**20
        status = "\n".join(
            ["Name: x", "RssAnon:   2048 kB", "RssFile:  1024 kB", "RssShmem: 0 kB"]
        )
        pinned = {
            "allocated_bytes.current": 4 * 2**20,
            "reserved_bytes.current": 8 * 2**20,
        }
        row = summarize_host(heap, status, pinned)
        self.assertEqual(row["heap_in_use_mib"], 1.0)
        self.assertEqual(row["heap_free_retained_mib"], 2.0)
        self.assertEqual(row["heap_arena_mib"], 3.0)
        self.assertEqual(row["heap_mmapped_mib"], 2.0)
        self.assertEqual(row["rss_anon_mib"], 2.0)
        self.assertEqual(row["rss_file_mib"], 1.0)
        self.assertEqual(row["pinned_reserved_mib"], 8.0)
        self.assertEqual(row["pinned_allocated_mib"], 4.0)
        # A torch without host allocator statistics still yields the heap row.
        self.assertNotIn("pinned_reserved_mib", summarize_host(heap, status, None))

    def test_host_stats_can_trim_the_heap_and_report_both_sides(self):
        worker = MemoryProbeWorker()
        worker.rank = 0
        calls = []
        with (
            patch(
                "glm53_setup.runtime.memory_probe.read_host",
                side_effect=lambda: (
                    calls.append("read") or {"rss_anon_mib": len(calls)}
                ),
            ),
            patch(
                "glm53_setup.runtime.memory_probe.trim_heap",
                side_effect=lambda: calls.append("trim") or 1,
            ),
        ):
            plain = worker.host_stats()
            trimmed = worker.host_stats(trim=True)
        self.assertEqual(plain, {"rank": 0, "rss_anon_mib": 1})
        self.assertEqual(calls, ["read", "read", "trim", "read"])
        self.assertEqual(trimmed["before_trim"], {"rss_anon_mib": 2})
        self.assertEqual(trimmed["trim_released"], 1)
        self.assertEqual(trimmed["rss_anon_mib"], 4)

    def test_fa2_stage_off_serves_the_reference_path_and_full_restores(self):
        from glm53_setup.runtime import fa2_attention, memory_probe

        original = (fa2_attention.sparse_nope_fa2, fa2_attention.use_fa2)
        seen = []

        def reference(query, cache, indices, scale):
            # The reference module asks use_fa2 again; inside the partial stage
            # the answer has to be no, or the call would recurse.
            seen.append(fa2_attention.use_fa2(2048))
            return "reference result"

        fake = {
            "torch": SimpleNamespace(),
            "glm53_reference": SimpleNamespace(sparse_nope_reference=reference),
        }
        try:
            with (
                patch.dict(sys.modules, fake),
                patch.dict("os.environ", {"GLM53_FA2_ATTENTION": "1"}),
            ):
                self.assertEqual(memory_probe.fa2_stage("off"), "off")
                self.assertTrue(fa2_attention.use_fa2(2048))
                result = fa2_attention.sparse_nope_fa2("q", "cache", "idx", 1.0)
                self.assertEqual(result, "reference result")
                self.assertEqual(seen, [False])
                with self.assertRaises(ValueError):
                    memory_probe.fa2_stage("half")
                self.assertEqual(memory_probe.fa2_stage("full"), "full")
            self.assertEqual(
                (fa2_attention.sparse_nope_fa2, fa2_attention.use_fa2), original
            )
        finally:
            fa2_attention.sparse_nope_fa2, fa2_attention.use_fa2 = original

    def test_census_counts_cpu_tensor_storage_once(self):
        from glm53_setup.runtime import memory_probe

        class Storage:
            def __init__(self, ptr, size):
                self.ptr, self.size = ptr, size

            def data_ptr(self):
                return self.ptr

            def nbytes(self):
                return self.size

        class Tensor:
            def __init__(self, kind, storage):
                self.device = SimpleNamespace(type=kind)
                self.storage = storage

            def untyped_storage(self):
                return self.storage

        shared = Storage(1, 2 * 2**20)
        live = [Tensor("cpu", shared), Tensor("cpu", shared), Tensor("cuda", shared)]
        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(Tensor=Tensor)}),
            patch("gc.get_objects", return_value=live + ["text"]),
        ):
            row = memory_probe.census()
        self.assertEqual(row["cpu_tensors"], 2)
        self.assertEqual(row["cpu_tensor_mib"], 2.0)
        self.assertEqual(row["top_types"]["Tensor"], 3)
        self.assertEqual(row["objects"], 4)

    def test_trace_names_the_first_differing_call_in_execution_order(self):
        from glm53_setup.runtime.memory_probe import trace_differences

        reference = [["embed_tokens", 5, 1], ["layers.0.mlp", 5, 7], ["lm_head", 5, 9]]
        same = trace_differences(reference, [list(row) for row in reference])
        self.assertIsNone(same["first"])
        self.assertEqual(same["modules"], {})
        moved = [["embed_tokens", 5, 1], ["layers.0.mlp", 5, 8], ["lm_head", 5, 10]]
        result = trace_differences(reference, moved)
        self.assertEqual(result["first"]["module"], "layers.0.mlp")
        self.assertEqual(result["first"]["order"], 1)
        self.assertEqual(result["modules"], {"layers.0.mlp": 1, "lm_head": 1})
        self.assertEqual(result["calls"], [3, 3])

    def test_trace_covers_layers_and_two_levels_below_them(self):
        import re

        from glm53_setup.runtime.memory_probe import TRACED

        pattern = re.compile(TRACED)
        for name in (
            "model.embed_tokens",
            "model.layers.3",
            "model.layers.3.self_attn",
            "model.layers.3.mlp.experts",
            "lm_head",
            "model.norm",
        ):
            self.assertTrue(pattern.search(name), name)
        for name in ("model", "model.layers", "model.layers.3.mlp.experts.w13"):
            self.assertFalse(pattern.search(name), name)

    def test_profile_key_mounts_the_probe_and_stays_exclusive(self):
        profile = config.load(ROOT / "examples/server.example.toml")
        self.assertNotIn(
            "--worker-extension-cls", config.serve_args(profile, 0, "/hf/model")
        )
        profile["validation"]["memory_probe"] = True
        config.validate(profile)
        args = config.serve_args(profile, 0, "/hf/model")
        self.assertEqual(
            args[args.index("--worker-extension-cls") + 1],
            "glm53_setup.runtime.memory_probe.MemoryProbeWorker",
        )
        self.assertIn("--enable-prefix-caching", args)  # nothing else changes
        command = server.command(
            profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
        )
        self.assertTrue(
            any(
                v.endswith(":/opt/glm53/glm53_setup/runtime/memory_probe.py:ro")
                for v in command
            ),
            command,
        )
        for section, key in (
            ("lpa", "enabled"),
            ("validation", "component_worker"),
            ("validation", "expert_worker"),
        ):
            p = copy.deepcopy(profile)
            p[section][key] = True
            # Their own scope guards may fire first on the distributed template.
            with self.assertRaises(ValueError):
                config.validate(p)
        profile["validation"]["memory_probe"] = "yes"
        with self.assertRaisesRegex(ValueError, "memory_probe"):
            config.validate(profile)


if __name__ == "__main__":
    unittest.main()
