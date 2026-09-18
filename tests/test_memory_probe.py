import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup import server_config as config
from glm53_setup.runtime.memory_probe import MemoryProbeWorker, summarize

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
