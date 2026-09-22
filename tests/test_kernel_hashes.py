import json
import sys
import tempfile
import unittest
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

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda i: SimpleNamespace(
                    multi_processor_count=48
                )
            )
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
        json.dumps(row)

    def test_the_layer_stages_join_the_hashes_and_a_failure_names_itself(self):
        worker = MemoryProbeWorker()
        worker.rank = 0
        worker.get_model = lambda: "model"
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda i: SimpleNamespace(
                    multi_processor_count=48
                )
            )
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
    def run_tool(self, ranks, reference=None):
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
                    return_value=("c", {"Image": "sha256:img"}),
                ),
                patch.object(
                    kernel_hashes.server, "collective_rpc", return_value=ranks
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
            self.assertEqual(rpc.call_args.args[1], "kernel_hashes")
            self.assertEqual(rpc.call_args.kwargs["seed"], 0)
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


if __name__ == "__main__":
    unittest.main()
