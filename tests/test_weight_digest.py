import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup.runtime.memory_probe import (
    MemoryProbeWorker,
    digest_differences,
    digest_summary,
    layer_of,
)

ROWS = [
    ["model.layers.0.mlp.experts.w13", 8, 123],
    ["model.layers.0.input_layernorm.weight", 4, 5],
    ["model.layers.1.self_attn.q_proj.weight", 2, 9],
    ["lm_head.weight", 3, 7],
]


class Module:
    """The two iterators a digest walks, on a class the fake torch calls nn.Module."""

    def __init__(self, parameters, buffers=()):
        self._parameters = dict(parameters)
        self._buffers = dict(buffers)

    def named_parameters(self):
        return self._parameters.items()

    def named_buffers(self):
        return self._buffers.items()


class DigestTests(unittest.TestCase):
    def test_layer_of_groups_a_layer_and_names_what_is_outside_them(self):
        self.assertEqual(layer_of("model.layers.12.mlp.experts.w13"), "model.layers.12")
        self.assertEqual(layer_of("drafter:model.layers.0.x"), "drafter:model.layers.0")
        self.assertEqual(layer_of("lm_head.weight"), "lm_head")
        self.assertEqual(layer_of("model.embed_tokens.weight"), "model.embed_tokens")
        self.assertEqual(layer_of("model.norm.weight"), "model.norm")

    def test_summary_is_order_independent_and_moves_with_one_tensor(self):
        summary = digest_summary(ROWS)
        self.assertEqual(summary["tensors"], 4)
        self.assertEqual(summary["elements"], 17)
        self.assertEqual(
            set(summary["layers"]), {"model.layers.0", "model.layers.1", "lm_head"}
        )
        self.assertEqual(
            digest_summary(list(reversed(ROWS)))["overall"], summary["overall"]
        )
        moved = [list(row) for row in ROWS]
        moved[0][2] = 124
        other = digest_summary(moved)
        self.assertNotEqual(other["overall"], summary["overall"])
        self.assertNotEqual(
            other["layers"]["model.layers.0"], summary["layers"]["model.layers.0"]
        )
        self.assertEqual(
            other["layers"]["model.layers.1"], summary["layers"]["model.layers.1"]
        )

    def test_differences_name_moved_missing_and_added_tensors(self):
        rows = [list(row) for row in ROWS[:3]]
        rows[0][2] = 124
        rows.append(["model.layers.1.extra", 1, 1])
        result = digest_differences(ROWS, rows)
        self.assertEqual(result["differing"], ["model.layers.0.mlp.experts.w13"])
        self.assertEqual(result["missing"], ["lm_head.weight"])
        self.assertEqual(result["added"], ["model.layers.1.extra"])
        self.assertEqual(result["layers"], {"model.layers.0": 1})
        self.assertEqual(result["same"], 2)
        self.assertEqual(digest_differences(ROWS, ROWS)["differing"], [])

    def test_worker_digest_walks_every_model_and_moves_the_prints_off_the_device(self):
        def tensor(numel, value, contiguous=True, offset=0):
            return SimpleNamespace(
                numel=lambda: numel,
                value=value,
                is_contiguous=lambda: contiguous,
                storage_offset=lambda: offset,
            )

        main = Module(
            {"model.layers.0.a": tensor(8, 11)},
            {"model.layers.0.buf": tensor(2, 3, contiguous=False)},
        )
        draft = Module({"model.layers.0.a": tensor(8, 12, offset=4)})
        worker = MemoryProbeWorker()
        worker.rank = 1
        worker.get_model = lambda: main
        worker.model_runner = SimpleNamespace(
            drafter=SimpleNamespace(model=draft), steps=3, main=main
        )

        class Print:  # what tensor_fingerprint returns: a device scalar, not an int
            def __init__(self, value):
                self.value = value

        class Stacked:
            def __init__(self, prints):
                self.prints = prints

            def cpu(self):
                return self

            def tolist(self):
                return [p.value for p in self.prints]

        fake_torch = SimpleNamespace(
            nn=SimpleNamespace(Module=Module), stack=lambda prints: Stacked(prints)
        )
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch(
                "glm53_setup.runtime.memory_probe.tensor_fingerprint",
                lambda tensor: Print(tensor.value),
            ),
        ):
            full = worker.weight_digest(tensors=True)
            short = worker.weight_digest()
        self.assertEqual(full["rank"], 1)
        self.assertEqual(
            full["rows"],
            [
                ["model.layers.0.a", 8, 11],
                ["model.layers.0.buf", 2, 3],
                ["drafter:model.layers.0.a", 8, 12],
            ],
        )
        self.assertEqual(full["summary"]["tensors"], 3)
        self.assertEqual(full["models"], ["", "drafter:"])
        self.assertEqual(
            full["copied"], ["model.layers.0.buf", "drafter:model.layers.0.a"]
        )
        self.assertNotIn("rows", short)
        self.assertEqual(short["summary"], full["summary"])
        json.dumps(full)  # the RPC answer must serialise

    def test_flat_bytes_flattens_before_viewing_so_scalars_work(self):
        # A 0-dim parameter (a scale) cannot be viewed as bytes; reshape(-1) first.
        # Measured on the reference pair, 2026-09-23: the first weight_digest raised
        # "self.dim() cannot be 0 to view Float as Byte".
        from unittest.mock import MagicMock, call

        from glm53_setup.runtime.memory_probe import flat_bytes

        tensor = MagicMock()
        fake_torch = SimpleNamespace(uint8="uint8")
        flat_bytes(tensor, fake_torch)
        chain = tensor.detach.return_value.contiguous.return_value
        chain.reshape.assert_called_once_with(-1)
        self.assertEqual(chain.reshape.return_value.view.call_args, call("uint8"))

    def test_trace_end_can_export_its_rows_for_a_later_launch(self):
        worker = MemoryProbeWorker()
        worker.rank = 0
        worker.probe_trace = {
            "handles": [],
            "names": [],
            "rows": [],
            "prints": [],
            "patched": [],
        }
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            kept = worker.trace_end(keep=True, export=True)
        self.assertEqual(kept["rows"], [])
        self.assertEqual(kept["kept"], 0)


class ToolTests(unittest.TestCase):
    def run_tool(self, argv, ranks, reference=None):
        from tools import weight_digest

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "digest.json"
            args = ["--output", str(output), *argv]
            if reference is not None:
                path = Path(directory) / "reference.json"
                path.write_text(json.dumps(reference), encoding="utf-8")
                args += ["--reference", str(path)]
            with (
                patch.object(
                    weight_digest.server,
                    "running_head",
                    return_value=("c", {"Image": "sha256:img"}),
                ),
                patch.object(
                    weight_digest.server, "collective_rpc", return_value=ranks
                ) as rpc,
                patch.object(
                    weight_digest.server_config,
                    "load",
                    return_value={"runtime": {}, "api": {}},
                ),
                patch.object(
                    weight_digest.server_config, "fingerprint", return_value="fp"
                ),
            ):
                code = weight_digest.main(args)
            rpc.assert_called_once()
            self.assertEqual(rpc.call_args.args[1], "weight_digest")
            self.assertTrue(rpc.call_args.kwargs["tensors"])
            return code, json.loads(output.read_text(encoding="utf-8"))

    def test_records_every_rank_and_compares_against_a_reference(self):
        ranks = [
            {"rank": 0, "summary": digest_summary(ROWS), "rows": ROWS},
            {"rank": 1, "summary": digest_summary(ROWS), "rows": ROWS},
        ]
        code, record = self.run_tool([], ranks)
        self.assertEqual(code, 0)
        self.assertEqual(record["fingerprint"], "fp")
        self.assertEqual(record["image"], "sha256:img")
        self.assertEqual([r["rank"] for r in record["ranks"]], [0, 1])
        same, _ = self.run_tool([], ranks, reference=record)
        self.assertEqual(same, 0)
        moved = json.loads(json.dumps(ranks))
        moved[1]["rows"][0][2] = 999
        code, compared = self.run_tool([], moved, reference=record)
        self.assertEqual(code, 1)
        self.assertEqual(
            compared["comparison"]["1"]["differing"], ["model.layers.0.mlp.experts.w13"]
        )
        self.assertEqual(compared["comparison"]["0"]["differing"], [])


if __name__ == "__main__":
    unittest.main()
