import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup.runtime import memory_probe
from glm53_setup.runtime.memory_probe import (
    MemoryProbeWorker,
    kernel_hash_differences,
)

HASHES = {
    "head_gate_fp32_rows4": "aa",
    "paged_mqa_logits": "bb",
    "paged_mqa_topk_set": "cc",
}


class FakeTensor:
    def __init__(self, shape, tag, strided=False, parent_width=None):
        self.shape, self.tag = tuple(shape), tag
        self.strided, self.parent_width = strided, parent_width

    def to(self, *args, **kwargs):
        return self

    def float(self):
        return self

    def type_as(self, other):
        return self

    def __getitem__(self, index):
        cols = index[1]
        width = len(range(*cols.indices(self.shape[1])))
        return FakeTensor(
            (self.shape[0], width), self.tag + "[view]", True, self.shape[1]
        )


def fake_layer(rope_on="owner"):
    """A layer-19 indexer whose projection is 160 wide (128 key + 32 head gates)."""

    def randn(*shape, generator):
        return FakeTensor(shape, f"randn{generator.seed}")

    class Generator:
        def manual_seed(self, seed):
            self.seed = seed
            return self

    fake_torch = SimpleNamespace(
        Generator=Generator,
        randn=randn,
        bfloat16="bf16",
        arange=lambda n, device=None: FakeTensor((n,), "arange"),
        cuda=SimpleNamespace(synchronize=lambda: None),
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                layer_norm=lambda x, *a: FakeTensor(x.shape, "eager")
            )
        ),
    )

    def project(h):
        return FakeTensor((h.shape[0], 160), "kw"), None

    indexer = SimpleNamespace(
        head_dim=128,
        n_head=32,
        rope_dim=64,
        k_norm=SimpleNamespace(
            weight=SimpleNamespace(device="cuda"), bias="b", eps=1e-6
        ),
        wk_weights_proj=type(
            "Projection", (), {"input_size": 2048, "__call__": lambda s, h: project(h)}
        )(),
    )

    def rope(positions, q, k):
        return FakeTensor(q.shape, "rope_q"), FakeTensor(k.shape, "rope_k")

    wrapper = SimpleNamespace(
        indexer=indexer,
        indexer_rotary_emb=rope if rope_on == "wrapper" else None,
    )
    owner = SimpleNamespace(
        indexer=indexer,
        indexer_rope_emb=rope if rope_on == "owner" else None,
        mla_attn=wrapper,
    )
    name = "language_model.model.layers.19.self_attn"
    model = SimpleNamespace(
        named_modules=lambda: [
            (name, owner),
            (f"{name}.indexer", indexer),
            (f"{name}.mla_attn", wrapper),
        ]
    )
    return fake_torch, indexer, owner, model


class InductorStateTests(unittest.TestCase):
    def test_reads_the_settings_codegen_uses_and_the_environment(self):
        inductor = SimpleNamespace(
            deterministic=False,
            batch_invariant=False,
            compile_threads=20,
            worker_start_method="subprocess",
            autotune_local_cache=True,
            force_disable_caches=False,
        )
        environ = {
            "TORCHINDUCTOR_DETERMINISTIC": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor-deterministic",
            "PATH": "/usr/bin",
        }
        state = memory_probe.inductor_state(inductor, environ)
        self.assertEqual(
            state["config"],
            {
                "deterministic": False,
                "batch_invariant": False,
                "compile_threads": 20,
                "worker_start_method": "subprocess",
                "autotune_local_cache": True,
                "force_disable_caches": False,
            },
        )
        self.assertEqual(
            state["environ"],
            {
                "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor-deterministic",
                "TORCHINDUCTOR_DETERMINISTIC": "1",
            },
        )
        # The variable says 1 and the config says False: named, not inferred.
        self.assertTrue(state["environ_disagrees"])
        inductor.deterministic = True
        self.assertFalse(
            memory_probe.inductor_state(inductor, environ)["environ_disagrees"]
        )
        json.dumps(state)


class ConfigWatchTests(unittest.TestCase):
    def test_writes_to_a_watched_entry_are_recorded_with_their_caller(self):
        import contextvars

        sentinel = object()
        var = contextvars.ContextVar("deterministic", default=sentinel)
        entry = SimpleNamespace(default=True, user_override=var)
        config = SimpleNamespace(_config={"deterministic": entry})
        writes = []
        memory_probe.watch_config_entry(config, "deterministic", writes, limit=2)
        # The entry still behaves as the ContextVar it wraps.
        self.assertIs(entry.user_override.get(), sentinel)
        token = entry.user_override.set(False)
        self.assertIs(entry.user_override.get(), False)
        entry.user_override.reset(token)
        self.assertIs(entry.user_override.get(), sentinel)
        entry.user_override.set(True)
        self.assertEqual(
            [(w["name"], w["op"], w["value"]) for w in writes],
            [("deterministic", "set", "False"), ("deterministic", "reset", None)],
        )  # the limit keeps the third write out
        self.assertTrue(
            any("test_kernel_hashes" in line for line in writes[0]["stack"])
        )
        self.assertIn("thread", writes[0])
        json.dumps(writes)
        # Watching twice does not wrap twice.
        memory_probe.watch_config_entry(config, "deterministic", writes)
        self.assertIs(entry.user_override.inner, var)

    def test_state_reports_the_entry_default_and_override(self):
        import contextvars

        sentinel = object()
        var = contextvars.ContextVar("deterministic", default=sentinel)
        entry = SimpleNamespace(
            default=True, user_override=var, env_value_force=sentinel
        )
        inductor = SimpleNamespace(
            deterministic=False,
            _config={"deterministic": entry},
            codegen_config=lambda: "deterministic = False",
        )
        var.set(False)
        state = memory_probe.inductor_state(inductor, {})
        self.assertEqual(
            state["entry"],
            {"default": True, "user_override": "False", "unset": False, "forced": None},
        )
        entry.env_value_force = True  # the pin of glm53-inductor-pin.pth
        self.assertEqual(
            memory_probe.inductor_state(inductor, {})["entry"]["forced"], "True"
        )
        self.assertEqual(state["codegen_config"], "deterministic = False")


class AutotunerTests(unittest.TestCase):
    def test_rows_name_each_kernels_file_configs_and_served_launchers(self):
        class Autotuner:
            def __init__(self, name, filename, configs, launchers):
                self.inductor_meta = {"kernel_name": name}
                self.filename = filename
                self.size_hints = {"x": 512, "r0_": 128}
                self.configs = configs
                self.launchers = [SimpleNamespace(config=c) for c in launchers]

        def config(xblock, warps):
            return SimpleNamespace(
                kwargs={"XBLOCK": xblock}, num_warps=warps, num_stages=1
            )

        norm = Autotuner(
            "triton_per_fused__to_copy_native_layer_norm_0",
            "/root/.cache/torchinductor/pf/cpfowybf.py",
            [config(1, 2), config(8, 2), config(32, 2)],
            [config(8, 2)],
        )
        norm.autotune_cache_info = {"autotune_cache_state": "hit", "best_config": ()}
        norm.inductor_meta["deterministic"] = False
        other = Autotuner("triton_poi_fused_add_0", "/c/zz/czz.py", [], [])
        other.configs = None  # torch 2.13 drops the candidates after precompile

        class DeadProxy:  # gc.get_objects() holds weak proxies whose referent died
            @property
            def __class__(self):
                raise ReferenceError("weakly-referenced object no longer exists")

        rows = memory_probe.autotuner_rows(
            [object(), DeadProxy(), other, norm], Autotuner, "layer_norm"
        )
        self.assertEqual(
            rows,
            [
                {
                    "kernel": "triton_per_fused__to_copy_native_layer_norm_0",
                    "file": "pf/cpfowybf.py",
                    "size_hints": {"x": 512, "r0_": 128},
                    "configs": [
                        {"XBLOCK": 1, "num_warps": 2, "num_stages": 1},
                        {"XBLOCK": 8, "num_warps": 2, "num_stages": 1},
                        {"XBLOCK": 32, "num_warps": 2, "num_stages": 1},
                    ],
                    "launchers": [{"XBLOCK": 8, "num_warps": 2, "num_stages": 1}],
                    "cache": "hit",
                    "deterministic": False,
                }
            ],
        )
        self.assertEqual(len(memory_probe.autotuner_rows([other, norm], Autotuner)), 2)
        json.dumps(rows)

    def test_differences_name_the_files_whose_launchers_differ_between_ranks(self):
        def row(xblock):
            return {"file": "pf/c.py", "launchers": [{"XBLOCK": xblock}]}

        same = {"file": "7i/c.py", "launchers": [{"XBLOCK": 1}]}
        ranks = [
            {"rank": 0, "autotuners": [row(8), same]},
            {"rank": 1, "autotuners": [row(1), same]},
        ]
        self.assertEqual(memory_probe.autotuner_differences(ranks), ["pf/c.py"])
        ranks[1]["autotuners"] = [row(8), same]
        self.assertEqual(memory_probe.autotuner_differences(ranks), [])
        ranks[1]["autotuners"] = [row(8)]  # a kernel one rank never loaded
        self.assertEqual(memory_probe.autotuner_differences(ranks), ["7i/c.py"])


class DifferenceTests(unittest.TestCase):
    def test_ranks_that_agree_and_the_keys_on_which_they_differ(self):
        same = [{"rank": 0, "hashes": HASHES}, {"rank": 1, "hashes": dict(HASHES)}]
        self.assertEqual(
            kernel_hash_differences(same),
            {"ranks": [0, 1], "agree": True, "differing": []},
        )
        moved = [
            {"rank": 0, "hashes": HASHES},
            {"rank": 1, "hashes": {**HASHES, "paged_mqa_logits": "xx"}},
        ]
        self.assertEqual(
            kernel_hash_differences(moved)["differing"], ["paged_mqa_logits"]
        )
        self.assertFalse(kernel_hash_differences(moved)["agree"])
        # A key one rank lacks counts as a difference.
        short = [
            {"rank": 0, "hashes": HASHES},
            {"rank": 1, "hashes": {"head_gate_fp32_rows4": "aa"}},
        ]
        self.assertEqual(
            kernel_hash_differences(short)["differing"],
            ["paged_mqa_logits", "paged_mqa_topk_set"],
        )

    def test_worker_method_labels_the_rank_and_hands_the_kernels_their_modules(self):
        worker = MemoryProbeWorker()
        worker.rank = 1
        seen = {}

        def fake_hashes(torch, kpool_ops, deep_gemm, sms, seed):
            seen.update(sms=sms, seed=seed, kpool=kpool_ops, dg=deep_gemm)
            return dict(HASHES)

        modes = []
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda i: SimpleNamespace(
                    multi_processor_count=48
                )
            ),
            inference_mode=lambda: modes.append("inference") or nullcontext(),
        )
        kpool = SimpleNamespace()
        dg = SimpleNamespace()
        modules = {
            "torch": fake_torch,
            "vllm": SimpleNamespace(),
            "vllm.models": SimpleNamespace(),
            "vllm.models.glm5next": SimpleNamespace(),
            "vllm.models.glm5next.nvidia": SimpleNamespace(),
            "vllm.models.glm5next.nvidia.ops": SimpleNamespace(kpool_compress=kpool),
            "vllm.models.glm5next.nvidia.ops.kpool_compress": kpool,
            "vllm.utils": SimpleNamespace(deep_gemm=dg),
            "vllm.utils.deep_gemm": dg,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(memory_probe, "indexer_kernel_hashes", fake_hashes),
        ):
            row = worker.kernel_hashes(seed=7)
        self.assertEqual(row["rank"], 1)
        self.assertEqual(row["seed"], 7)
        self.assertEqual(row["sms"], 48)
        kernels = {k: v for k, v in row["hashes"].items() if not k.startswith("layer")}
        self.assertEqual(kernels, HASHES)
        # Without the attention module the layer stages name their failure.
        self.assertIn("ImportError", row["hashes"]["layer19_error"])
        self.assertEqual(seen, {"sms": 48, "seed": 7, "kpool": kpool, "dg": dg})
        # As served: grad mode is guarded, so outside it Dynamo compiles anew.
        self.assertEqual(modes, ["inference"])
        json.dumps(row)

    def test_the_layer_stages_join_the_hashes_and_a_failure_names_itself(self):
        worker = MemoryProbeWorker()
        worker.rank = 0
        worker.get_model = lambda: "model"
        modes = []
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda i: SimpleNamespace(
                    multi_processor_count=48
                )
            ),
            inference_mode=lambda: modes.append("inference") or nullcontext(),
        )
        attention = SimpleNamespace()
        modules = {
            "torch": fake_torch,
            "vllm": SimpleNamespace(),
            "vllm.models": SimpleNamespace(),
            "vllm.models.glm5next": SimpleNamespace(),
            "vllm.models.glm5next.nvidia": SimpleNamespace(attention=attention),
            "vllm.models.glm5next.nvidia.attention": attention,
            "vllm.models.glm5next.nvidia.ops": SimpleNamespace(
                kpool_compress=SimpleNamespace()
            ),
            "vllm.models.glm5next.nvidia.ops.kpool_compress": SimpleNamespace(),
            "vllm.utils": SimpleNamespace(deep_gemm=SimpleNamespace()),
            "vllm.utils.deep_gemm": SimpleNamespace(),
        }
        seen = {}

        def stages(torch, glm_attention, model, layer, seed):
            seen.update(model=model, layer=layer, seed=seed, att=glm_attention)
            return {"k_norm_compiled": "kn", "rope_k": "rk"}

        with (
            patch.dict(sys.modules, modules),
            patch.object(
                memory_probe, "indexer_kernel_hashes", lambda *a: dict(HASHES)
            ),
            patch.object(memory_probe, "indexer_stage_hashes", stages),
        ):
            row = worker.kernel_hashes(layer=7)
        self.assertEqual(
            seen, {"model": "model", "layer": 7, "seed": 0, "att": attention}
        )
        self.assertEqual(row["hashes"]["layer7_k_norm_compiled"], "kn")
        self.assertEqual(row["hashes"]["layer7_rope_k"], "rk")
        self.assertEqual(row["hashes"]["paged_mqa_logits"], "bb")

        def failing(*args):
            raise KeyError("no module named *.layers.7.self_attn.indexer")

        with (
            patch.dict(sys.modules, modules),
            patch.object(
                memory_probe, "indexer_kernel_hashes", lambda *a: dict(HASHES)
            ),
            patch.object(memory_probe, "indexer_stage_hashes", failing),
        ):
            row = worker.kernel_hashes(layer=7)
        self.assertIn("no module named", row["hashes"]["layer7_error"])
        self.assertEqual(row["hashes"]["paged_mqa_logits"], "bb")

    def test_indexer_layer_finds_the_module_and_its_owner_by_suffix(self):
        indexer = SimpleNamespace()
        owner = SimpleNamespace(indexer=indexer)
        model = SimpleNamespace(
            named_modules=lambda: [
                ("language_model.model.layers.19.self_attn", owner),
                ("language_model.model.layers.19.self_attn.indexer", indexer),
                ("language_model.model.layers.23.self_attn.indexer", SimpleNamespace()),
            ]
        )
        self.assertEqual(memory_probe.indexer_layer(model, 19), (indexer, owner))
        with self.assertRaises(KeyError):
            memory_probe.indexer_layer(model, 5)

    def test_the_served_key_norm_reads_a_strided_view_of_the_projection(self):
        # 2026-09-24: the served call is _fused_indexer_k_norm(kw[:, :head_dim]),
        # a strided view of the 160-wide projection output, and Inductor gives
        # it its own kernel; a contiguous input runs a different one.
        fake_torch, indexer, owner, model = fake_layer()
        seen = []

        def k_norm(x, weight, bias, dim, eps):
            seen.append(x)
            return FakeTensor(x.shape, "normed")

        attention = SimpleNamespace(_fused_indexer_k_norm=k_norm)
        with patch.object(memory_probe, "bytes_digest", lambda t, torch: t.tag):
            hashes = memory_probe.indexer_stage_hashes(
                fake_torch, attention, model, 19, seed=0, rows=2048
            )
        served = [x for x in seen if x.strided]
        self.assertEqual([x.shape for x in served], [(2048, 128), (8, 128), (4, 128)])
        self.assertTrue(all(x.parent_width == 160 for x in served))
        self.assertEqual(
            [x for x in seen if not x.strided][0].shape, (2048, 128)
        )  # the 1.11.1 contiguous hash stays, for comparison with its records
        for rows in (2048, 8, 4):
            self.assertIn(f"k_norm_served_rows{rows}", hashes)
        self.assertIn("k_norm_served_eager_fp32", hashes)
        self.assertEqual(hashes["rope_source"], "indexer_rope_emb")
        self.assertIn("rope_k", hashes)

    def test_the_rope_is_found_on_the_owner_or_its_wrapper_and_its_absence_named(self):
        fake_torch, indexer, owner, model = fake_layer(rope_on="wrapper")
        attention = SimpleNamespace(
            _fused_indexer_k_norm=lambda x, *a: FakeTensor(x.shape, "n")
        )
        with patch.object(memory_probe, "bytes_digest", lambda t, torch: t.tag):
            hashes = memory_probe.indexer_stage_hashes(fake_torch, attention, model, 19)
        self.assertEqual(hashes["rope_source"], "mla_attn.indexer_rotary_emb")
        self.assertIn("rope_q", hashes)
        fake_torch, indexer, owner, model = fake_layer(rope_on=None)
        with patch.object(memory_probe, "bytes_digest", lambda t, torch: t.tag):
            hashes = memory_probe.indexer_stage_hashes(fake_torch, attention, model, 19)
        self.assertEqual(hashes["rope_source"], "none")
        self.assertNotIn("rope_q", hashes)

    def test_the_kernel_shapes_are_the_served_indexers(self):
        # index_n_heads 32 x 128, kpool 4, 512 of 540 pools, a 64-pool block
        # (config.json of the served checkpoint; docs/validation.md).
        self.assertEqual(
            memory_probe.INDEXER_SHAPES,
            {
                "heads": 32,
                "dim": 128,
                "kpool": 4,
                "select": 512,
                "pools": 540,
                "block": 64,
            },
        )


class ToolTests(unittest.TestCase):
    def run_tool(self, ranks, reference=None, autotuners=None):
        autotuners = autotuners or [
            {"rank": 0, "autotuners": []},
            {"rank": 1, "autotuners": []},
        ]
        from tools import kernel_hashes

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kernels.json"
            args = ["--output", str(output)]
            if reference is not None:
                path = Path(directory) / "reference.json"
                path.write_text(json.dumps(reference), encoding="utf-8")
                args += ["--reference", str(path)]
            with (
                patch.object(
                    kernel_hashes.server,
                    "running_head",
                    return_value=({"name": "c"}, {"Image": "sha256:img"}),
                ),
                patch.object(
                    kernel_hashes.server,
                    "collective_rpc",
                    side_effect=lambda profile, method, **kw: {
                        "autotuners": autotuners,
                        "inductor_state": [
                            {"rank": 1, "environ_disagrees": True},
                            {"rank": 0, "environ_disagrees": False},
                        ],
                    }.get(method, ranks),
                ) as rpc,
                patch.object(
                    kernel_hashes.server_config,
                    "load",
                    return_value={"runtime": {}, "api": {}},
                ),
                patch.object(
                    kernel_hashes.server_config, "fingerprint", return_value="fp"
                ),
            ):
                code = kernel_hashes.main(args)
            # The served launchers are read before kernel_hashes runs anything.
            self.assertEqual(
                [c.args[1] for c in rpc.call_args_list],
                ["autotuners", "inductor_state", "kernel_hashes", "autotuners"],
            )
            self.assertEqual(rpc.call_args_list[2].kwargs["seed"], 0)
            return code, json.loads(output.read_text(encoding="utf-8"))

    def test_records_both_ranks_and_compares_them_and_a_reference(self):
        ranks = [
            {"rank": 1, "seed": 0, "sms": 48, "hashes": dict(HASHES)},
            {"rank": 0, "seed": 0, "sms": 48, "hashes": dict(HASHES)},
        ]
        code, record = self.run_tool(ranks)
        self.assertEqual(code, 0)
        self.assertEqual([r["rank"] for r in record["ranks"]], [0, 1])
        self.assertTrue(record["across_ranks"]["agree"])
        # The container rank 0's state names, not the state itself.
        self.assertEqual(record["container"], "c")
        same, compared = self.run_tool(ranks, reference=record)
        self.assertEqual(same, 0)
        self.assertEqual(compared["against_reference"], {"0": [], "1": []})
        moved = json.loads(json.dumps(ranks))
        moved[0]["hashes"]["paged_mqa_logits"] = "xx"  # rank 1
        code, compared = self.run_tool(moved, reference=record)
        self.assertEqual(code, 1)
        self.assertEqual(compared["across_ranks"]["differing"], ["paged_mqa_logits"])
        self.assertEqual(
            compared["against_reference"], {"0": [], "1": ["paged_mqa_logits"]}
        )

    def test_records_the_served_launchers_and_names_a_rank_difference(self):
        ranks = [
            {"rank": 0, "seed": 0, "sms": 48, "hashes": dict(HASHES)},
            {"rank": 1, "seed": 0, "sms": 48, "hashes": dict(HASHES)},
        ]
        tuned = [
            {
                "rank": 1,
                "autotuners": [{"file": "pf/c.py", "launchers": [{"XBLOCK": 1}]}],
            },
            {
                "rank": 0,
                "autotuners": [{"file": "pf/c.py", "launchers": [{"XBLOCK": 8}]}],
            },
        ]
        code, record = self.run_tool(ranks, autotuners=tuned)
        self.assertEqual([r["rank"] for r in record["autotuners"]], [0, 1])
        self.assertEqual(record["autotuners_differing"], ["pf/c.py"])
        self.assertEqual([r["rank"] for r in record["inductor_state"]], [0, 1])
        # Nothing new compiled by the hashes (the same answers before and after).
        self.assertEqual(record["autotuners_new_after_hashes"], {"0": [], "1": []})
        self.assertEqual(code, 0)  # informative: several kernels differ by design


if __name__ == "__main__":
    unittest.main()
