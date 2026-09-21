import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from glm53_setup.runtime import draft_observe

TORCH = importlib.util.find_spec("torch") is not None


class RowTests(unittest.TestCase):
    def test_one_row_per_request_that_verified_drafts(self):
        rows = draft_observe.rows_of(
            req_ids=["prefill", "decode"],
            cu_num_logits=[0, 1, 4],
            prompt_lens=[50, 10],
            num_sampled=[1, 2],
            ids=[[7, 3, 21, 22], [9, 21, 40, 41], [49, 14, 15, 16]],
            floats=[[-9.0, -0.1, -4.0, -7.0], [2.0, 3.0, 0.0, 1.5]],
            confidence=[[[0.0, 0.0]] * 2, [[0.91, 4.5], [0.4, 0.0625]]],
            top=[[[0] * 5] * 2, [[21, 1, 2, 3, 4], [22, 40, 6, 7, 8]]],
        )
        self.assertEqual(
            rows,
            [
                {
                    "r": "decode",
                    "p": 5,
                    "a": 1,
                    "d": [21, 22],
                    "dp": [0.91, 0.4],
                    "dm": [4.5, 0.0625],
                    "d5": [[21, 1, 2, 3, 4], [22, 40, 6, 7, 8]],
                    "t": [21, 40, 41],
                    "tl": [-0.1, -4.0],
                    "tm": [3.0, 0.0, 1.5],
                }
            ],
        )

    def test_lines_are_appended_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observe.jsonl"
            with (
                mock.patch.dict(os.environ, {"GLM53_DRAFT_OBSERVE": str(path)}),
                mock.patch.object(draft_observe, "_lines", None),
            ):
                draft_observe._append([{"r": "a", "p": 1}, {"r": "b", "p": 2}])
                draft_observe._lines["file"].close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows, [{"r": "a", "p": 1}, {"r": "b", "p": 2}])

    def test_the_switch_is_read_once(self):
        with mock.patch.object(draft_observe, "_active", None):
            with mock.patch.dict(os.environ, clear=True):
                self.assertFalse(draft_observe.active())
            with mock.patch.dict(os.environ, {"GLM53_DRAFT_OBSERVE": "/x"}):
                self.assertFalse(draft_observe.active())


@unittest.skipUnless(TORCH, "torch required")
class TensorTests(unittest.TestCase):
    def speculator(self, logits, local=False):
        return SimpleNamespace(
            use_local_argmax_reduction=local,
            model=SimpleNamespace(compute_logits=lambda hidden: logits[hidden]),
            _greedy_sample_draft=lambda hidden: "local",
            max_num_reqs=4,
            num_speculative_steps=3,
        )

    def test_the_draft_token_is_the_stock_argmax_and_its_confidence_is_kept(self):
        import torch

        logits = torch.tensor(
            [[0.0, 2.0, 1.0, -1.0, 0.5, 0.25], [3.0, 3.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.bfloat16,
        )
        speculator = self.speculator(logits)
        hidden, slots = torch.tensor([0, 1]), torch.tensor([2, 0], dtype=torch.int32)
        tokens = draft_observe.draft(speculator, hidden, slots, torch.tensor(1))
        self.assertEqual(tokens.tolist(), logits.argmax(dim=-1).tolist())
        stats, top = speculator.glm53_draft_observe
        self.assertEqual(top[2, 1].tolist()[:3], [1, 2, 4])
        self.assertAlmostEqual(
            stats[2, 1, 0].item(), logits[0].float().softmax(-1)[1].item(), places=5
        )
        self.assertEqual(stats[2, 1, 1].item(), 1.0)
        # An exact tie in the raw dtype shows as margin 0.
        self.assertEqual(stats[0, 1, 1].item(), 0.0)
        self.assertEqual(stats[:, 0].abs().sum().item(), 0.0)

    def test_the_local_reduction_is_left_alone(self):
        import torch

        speculator = self.speculator(torch.zeros(1, 6), local=True)
        result = draft_observe.draft(speculator, torch.tensor([0]), None, None)
        self.assertEqual(result, "local")
        self.assertFalse(hasattr(speculator, "glm53_draft_observe"))

    def test_the_target_row_scores_the_input_of_the_next_row(self):
        import torch

        logits = torch.tensor(
            [[0.0, 5.0, 1.0, 0.0], [0.0, 0.0, 4.0, 3.5], [9.0, 0.0, 0.0, 0.0]]
        )
        batch = SimpleNamespace(
            input_ids=torch.tensor([8, 8, 3, 1, 3], dtype=torch.int32),
            logits_indices=torch.tensor([2, 3, 4]),
            positions=torch.tensor([0, 0, 40, 41, 42]),
        )
        pending = draft_observe.target(logits, batch)
        inputs, argmax, positions = pending["ids"].tolist()
        self.assertEqual(inputs, [3, 1, 3])
        self.assertEqual(argmax, [1, 2, 0])
        self.assertEqual(positions, [40, 41, 42])
        logprob, margin = pending["floats"].tolist()
        # Row 0 verifies draft 1 (token 1, the argmax), row 1 verifies draft 2 (token 3).
        self.assertAlmostEqual(
            logprob[0], logits[0].log_softmax(-1)[1].item(), places=5
        )
        self.assertAlmostEqual(
            logprob[1], logits[1].log_softmax(-1)[3].item(), places=5
        )
        self.assertEqual(margin, [4.0, 0.5, 9.0])

    def test_write_joins_both_sides_by_request_slot(self):
        import torch

        speculator = SimpleNamespace(
            glm53_draft_observe=(
                torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2),
                torch.arange(4 * 3 * 5).reshape(4, 3, 5),
            )
        )
        batch = SimpleNamespace(
            idx_mapping=torch.tensor([3], dtype=torch.int32),
            req_ids=["r"],
            cu_num_logits_np=SimpleNamespace(tolist=lambda: [0, 3]),
            prefill_len_np=SimpleNamespace(tolist=lambda: [30]),
        )
        pending = {
            "ids": torch.tensor([[5, 6, 7], [6, 9, 9], [40, 41, 42]]),
            "floats": torch.tensor([[-0.5, -3.0, 0.0], [1.0, 2.0, 3.0]]),
        }
        with mock.patch.object(draft_observe, "_append") as append:
            draft_observe.write(speculator, batch, pending, torch.tensor([2]))
        (row,) = append.call_args.args[0]
        self.assertEqual((row["r"], row["p"], row["a"]), ("r", 11, 1))
        self.assertEqual(row["d"], [6, 7])
        self.assertEqual(row["dp"], [18.0, 20.0])
        self.assertEqual(row["d5"][0], [45, 46, 47, 48, 49])


if __name__ == "__main__":
    unittest.main()
