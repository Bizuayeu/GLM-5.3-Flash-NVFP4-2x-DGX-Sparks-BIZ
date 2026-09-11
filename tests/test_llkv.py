"""Contracts that keep experimental prefill changes out of decode/cache reuse."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup.runtime.llkv import AttentionInputExperiment, ExperimentSpec


class ExperimentSpecTests(unittest.TestCase):
    def test_fully_protected_prompt_bypasses_loading_and_prediction(self):
        experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
        experiment.layers = [None, None]
        experiment.predictor_identity = None
        experiment.reference_mask = None
        result = experiment.configure(
            "predict", 0, 20, tail=20, predictor_path="not-present-projector.pt"
        )
        self.assertEqual(result["mode"], "off")
        self.assertEqual(result["requested_mode"], "predict")
        self.assertEqual(experiment.spec.mode, "off")

    def test_new_prompt_reuses_loaded_artifact_but_file_change_reloads_it(self):
        experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
        experiment.layers = [None, None]
        experiment.predictor_identity = None
        experiment.reference_mask = None
        stat = SimpleNamespace(st_mtime_ns=1, st_size=100)
        with (
            patch("pathlib.Path.stat", return_value=stat),
            patch.object(
                experiment, "_load_predictor", return_value={1: "cached"}
            ) as load,
        ):
            experiment.configure("predict", 0, 10, predictor_path="projector.pt")
            experiment.configure("predict", 0, 20, predictor_path="projector.pt")
            self.assertEqual(load.call_count, 1)
            stat.st_mtime_ns = 2
            experiment.configure("predict", 0, 30, predictor_path="projector.pt")
            self.assertEqual(load.call_count, 2)

    def test_last_prompt_token_and_decode_are_protected(self):
        spec = ExperimentSpec("oracle", 2, 9, tail=1)
        self.assertEqual(spec.approximate_count([0, 1, 2, 3]), 4)
        self.assertEqual(spec.approximate_count([7, 8]), 1)
        self.assertEqual(spec.approximate_count([9]), 0)

    def test_single_token_prompt_never_approximates(self):
        self.assertEqual(ExperimentSpec("oracle", 0, 1).approximate_count([0]), 0)

    def test_reordered_or_batched_positions_are_rejected(self):
        spec = ExperimentSpec("oracle", 2, 9)
        for positions in ([0, 1, 0], [2, 4], [1, 1], [-1]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                spec.approximate_count(positions)

    def test_invalid_spec_cannot_silently_enable_an_experiment(self):
        for kwargs in (
            {"mode": "unknown"},
            {"cut": -1},
            {"prompt_length": 0},
            {"tail": 0},
            {"tail": 10},
            {"cut": 2.5},
            {"prompt_length": True},
            {"mode": []},
        ):
            values = dict(mode="oracle", cut=2, prompt_length=9)
            values.update(kwargs)
            with self.assertRaises(ValueError):
                ExperimentSpec(**values)


if __name__ == "__main__":
    unittest.main()
