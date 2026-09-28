"""Contracts that keep experimental prefill changes out of decode/cache reuse."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup.runtime.lpa import (
    AttentionInputExperiment,
    ExperimentSpec,
    LPAWorkerExtension,
)


class ExperimentSpecTests(unittest.TestCase):
    def test_speculative_cursor_correction_cannot_cross_into_the_prompt(self):
        spec = ExperimentSpec("predict", 32, 4012, tail=512)
        positions = [4013, 4014, 4015, 4016]
        self.assertTrue(spec.validate_scheduled_start(positions, 4016, True))
        self.assertEqual(spec.approximate_count(positions), 0)
        self.assertFalse(spec.validate_scheduled_start([0, 1], 0, False))
        for actual, expected, speculative in (
            (4011, 4016, True),
            (4017, 4016, True),
            (4013, 4016, False),
            (16, 20, True),
        ):
            with (
                self.subTest(actual=actual, expected=expected),
                self.assertRaises(ValueError),
            ):
                spec.validate_scheduled_start([actual], expected, speculative)

    def test_resumed_exact_prefix_and_tail_surround_the_approximate_span(self):
        spec = ExperimentSpec("predict", 0, 12, tail=2, approximate_start=4)
        self.assertEqual(spec.approximate_span([0, 1, 2, 3]), (0, 0))
        self.assertEqual(spec.approximate_span([2, 3, 4, 5]), (2, 4))
        self.assertEqual(spec.approximate_span([8, 9, 10, 11]), (0, 2))
        self.assertEqual(spec.approximate_count([12, 13]), 0)

    def test_ordinary_path_does_not_copy_gpu_positions_without_diagnostics(self):
        experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
        experiment.spec = ExperimentSpec("off", 0, 12)
        experiment.verify_state = False
        self.assertIsNone(
            experiment._attention_hook(0)(
                None,
                (),
                {
                    "hidden_states": object(),
                    "positions": object(),
                },
            )
        )

    def test_mlp_preserves_both_exact_sides_after_prefix_eviction(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch environment required")
        experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
        experiment.torch = torch
        experiment.spec = ExperimentSpec("predict", 0, 12, tail=2, approximate_start=4)
        experiment.skip_mlp = True
        experiment.counts = {"mlp_skipped_tokens": {}}
        experiment.current_positions = list(range(2, 12))
        x = torch.arange(10).float().reshape(10, 1)
        result = experiment._mlp_forward(0, lambda v: v * 2)(x)
        expected = x * 2
        expected[2:8] = 0
        self.assertTrue(torch.equal(result, expected))

    def test_prefill_state_diagnostic_ignores_unused_mtp_conv_slots(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch environment required")
        for dim_first in (False, True):
            experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
            experiment.torch = torch
            experiment.spec = ExperimentSpec("capture", 0, 5)
            experiment.verify_state = True
            experiment.current_positions = [0, 1, 2, 3]
            experiment.state_reference = {}
            experiment.state_errors = []
            conv = torch.zeros((1, 8, 6) if dim_first else (1, 6, 8))
            module = SimpleNamespace(
                prefix="layer",
                kv_cache=(conv, torch.zeros(1, 2, 2)),
                _conv_state_dim_first=dim_first,
                conv_size=4,
            )
            metadata = SimpleNamespace(non_spec_state_indices_tensor=torch.tensor([0]))
            context = SimpleNamespace(attn_metadata={"layer": metadata})
            fake = SimpleNamespace(get_forward_context=lambda: context)
            with patch.dict(sys.modules, {"vllm.forward_context": fake}):
                hook = experiment._state_hook(0)
                hook(module, (), None)
                conv.narrow(2 if dim_first else 1, 3, 3).fill_(123)
                experiment.spec = ExperimentSpec("oracle", 0, 5)
                hook(module, (), None)
                self.assertTrue(
                    all(row["max_abs"] == 0 for row in experiment.state_errors)
                )
                conv.narrow(2 if dim_first else 1, 0, 3).fill_(1)
                hook(module, (), None)
                self.assertEqual(experiment.state_errors[-2]["max_abs"], 1)

    def test_mtp_verification_and_rollback_positions_are_never_approximated(self):
        spec = ExperimentSpec("predict", 32, 1024, tail=512)
        for positions in ([1023], [1024, 1025, 1026, 1027], [1025, 1026, 1027, 1028]):
            self.assertEqual(spec.approximate_count(positions), 0)

    def test_mtp_requires_explicit_opt_in_and_supported_method(self):
        worker = LPAWorkerExtension()
        worker.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(enforce_eager=True),
            speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=3),
        )
        worker.get_model = lambda: "target-only"
        with self.assertRaises(ValueError):
            worker.lpa_configure()
        with patch("glm53_setup.runtime.lpa.AttentionInputExperiment") as experiment:
            worker.lpa_configure(allow_mtp=True, mode="off")
            experiment.assert_called_once_with("target-only")
            experiment.return_value.configure.assert_called_once_with(mode="off")
        worker.vllm_config.speculative_config.method = "eagle"
        with self.assertRaises(ValueError):
            worker.lpa_configure(allow_mtp=True)

    def test_mtp_depths_one_to_three_are_accepted(self):
        for depth in (1, 2, 3, 4, 5):
            with self.subTest(depth=depth):
                worker = LPAWorkerExtension()
                worker.vllm_config = SimpleNamespace(
                    scheduler_config=SimpleNamespace(max_num_seqs=1),
                    cache_config=SimpleNamespace(enable_prefix_caching=False),
                    model_config=SimpleNamespace(enforce_eager=True),
                    speculative_config=SimpleNamespace(
                        method="mtp", num_speculative_tokens=depth
                    ),
                )
                worker.get_model = lambda: "target-only"
                with patch("glm53_setup.runtime.lpa.AttentionInputExperiment"):
                    if depth <= 3:
                        worker.lpa_configure(allow_mtp=True, mode="off")
                        continue
                    with self.assertRaises(ValueError):
                        worker.lpa_configure(allow_mtp=True, mode="off")

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
