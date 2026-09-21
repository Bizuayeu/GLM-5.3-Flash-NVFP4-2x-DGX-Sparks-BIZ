import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from glm53_setup.runtime import draft_gate

TORCH = importlib.util.find_spec("torch") is not None


class SettingTests(unittest.TestCase):
    def read(self, value):
        env = {} if value is None else {"GLM53_DRAFT_GATE": value}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(draft_gate, "_tau", False),
        ):
            return draft_gate.tau(), draft_gate.active()

    def test_off_unless_a_probability_is_given(self):
        self.assertEqual(self.read(None), (None, False))
        self.assertEqual(self.read(""), (None, False))
        self.assertEqual(self.read("0.85"), (0.85, True))
        # 0 never stops and 1 stops after the first depth: the fixture's two ends.
        self.assertEqual(self.read("0"), (0.0, True))
        self.assertEqual(self.read("1"), (1.0, True))
        for bad in ("1.5", "-0.1", "high"):
            with self.assertRaises(ValueError):
                self.read(bad)

    def test_off_means_no_question_is_asked_of_the_speculator(self):
        with mock.patch.object(draft_gate, "_tau", None):
            self.assertFalse(draft_gate.stop(object(), 1, 1))

    def test_the_runner_hands_over_the_columns_that_were_drafted(self):
        self.assertEqual(draft_gate.drafted(SimpleNamespace(), 5), 5)
        self.assertEqual(draft_gate.drafted(SimpleNamespace(glm53_drafted=None), 5), 5)
        self.assertEqual(draft_gate.drafted(SimpleNamespace(glm53_drafted=2), 5), 2)
        self.assertEqual(draft_gate.drafted(SimpleNamespace(glm53_drafted=4), 3), 3)


@unittest.skipUnless(TORCH, "torch required")
class StopTests(unittest.TestCase):
    def speculator(self, probabilities, asynchronous=False):
        import torch

        stats = torch.zeros(4, 5, 2)
        for slot, row in probabilities.items():
            stats[slot, : len(row), 0] = torch.tensor(row)
        scheduler = SimpleNamespace(async_scheduling=asynchronous)
        return SimpleNamespace(
            vllm_config=SimpleNamespace(scheduler_config=scheduler),
            idx_mapping=torch.tensor([2, 0, -1, -1], dtype=torch.int32),
            glm53_draft_observe=(stats, None),
            glm53_drafted=4,
        )

    def run_gate(self, speculator, num_reqs, threshold=0.8):
        with (
            mock.patch.object(draft_gate, "_tau", threshold),
            mock.patch.object(draft_gate, "_broadcast", side_effect=lambda flag: flag),
        ):
            for drafted in range(1, 5):
                if draft_gate.stop(speculator, num_reqs, drafted):
                    break
        return draft_gate.drafted(speculator, 5)

    def test_it_stops_after_the_first_unsure_depth_and_keeps_that_draft(self):
        self.assertEqual(
            self.run_gate(self.speculator({2: [0.95, 0.9, 0.4, 0.99]}), 1), 3
        )
        self.assertEqual(self.run_gate(self.speculator({2: [0.3, 0.99]}), 1), 1)
        self.assertEqual(
            self.run_gate(self.speculator({2: [0.9, 0.9, 0.9, 0.9]}), 1), 5
        )

    def test_a_batch_drafts_as_deep_as_its_most_confident_request(self):
        speculator = self.speculator(
            {2: [0.3, 0.2, 0.1, 0.1], 0: [0.9, 0.85, 0.5, 0.9]}
        )
        self.assertEqual(self.run_gate(speculator, 2), 3)

    def test_the_other_ranks_follow_rank_zero(self):
        import torch

        speculator = self.speculator({2: [0.99, 0.99]})
        with (
            mock.patch.object(draft_gate, "_tau", 0.8),
            mock.patch.object(
                draft_gate, "_broadcast", side_effect=lambda flag: torch.ones_like(flag)
            ),
        ):
            self.assertTrue(draft_gate.stop(speculator, 1, 1))
        self.assertEqual(draft_gate.drafted(speculator, 5), 1)

    def test_without_the_draft_side_numbers_it_never_stops(self):
        speculator = self.speculator({})
        del speculator.glm53_draft_observe
        with mock.patch.object(draft_gate, "_tau", 0.8):
            self.assertFalse(draft_gate.stop(speculator, 1, 1))
        self.assertEqual(draft_gate.drafted(speculator, 5), 5)

    def test_the_asynchronous_scheduler_is_refused(self):
        with mock.patch.object(draft_gate, "_tau", 0.8):
            with self.assertRaises(ValueError):
                draft_gate.stop(self.speculator({}, asynchronous=True), 1, 1)


if __name__ == "__main__":
    unittest.main()
