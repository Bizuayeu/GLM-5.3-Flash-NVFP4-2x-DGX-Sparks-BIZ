import copy
import importlib.util
import unittest
from types import SimpleNamespace as NS

from glm53_setup.runtime.pipeline_state import validate_pipeline


class PipelineScopeTests(unittest.TestCase):
    def test_stage_layout_intersection_preserves_support_and_preference(self):
        from glm53_setup.runtime.pipeline_state import common_layout_names

        self.assertEqual(
            common_layout_names([["LBNHC", "LBHNC"], ["LBHNC"]]), ["LBHNC"]
        )
        self.assertEqual(
            common_layout_names([["LBHNC", "LBNHC"], ["LBNHC", "LBHNC"]]),
            ["LBHNC", "LBNHC"],
        )
        for layouts in ([], [[]], [["LBHNC"], []], [["LBHNC"], ["LBNHC"]]):
            with self.assertRaises(ValueError):
                common_layout_names(layouts)

    def test_independent_scope_rejects_untested_combinations(self):
        config = NS(
            parallel_config=NS(
                pipeline_parallel_size=2,
                tensor_parallel_size=1,
                use_sequence_parallel_moe=False,
            ),
            speculative_config=None,
            model_config=NS(enforce_eager=True, hf_text_config=NS(mhc=True)),
            cache_config=NS(enable_prefix_caching=False),
        )
        validate_pipeline(config)
        for section, key, value in (
            ("parallel_config", "pipeline_parallel_size", 3),
            ("parallel_config", "tensor_parallel_size", 2),
            ("parallel_config", "use_sequence_parallel_moe", True),
            ("model_config", "enforce_eager", False),
            ("cache_config", "enable_prefix_caching", True),
        ):
            trial = copy.deepcopy(config)
            setattr(getattr(trial, section), key, value)
            with self.assertRaises(ValueError):
                validate_pipeline(trial)
        config.speculative_config = object()
        with self.assertRaises(ValueError):
            validate_pipeline(config)
        config.parallel_config.pipeline_parallel_size = 1
        validate_pipeline(config)  # Existing TP control is unchanged.


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required")
class PipelineBufferTests(unittest.TestCase):
    def test_layout_preserves_fp32_mixes_separately_from_bf16_hidden(self):
        import torch

        from glm53_setup.runtime.pipeline_state import allocate_intermediate

        config = NS(mhc=True, mhc_num_residual_streams=4, hidden_size=4096)
        for tokens in (0, 1, 65):
            buffers = allocate_intermediate(config, tokens, torch.bfloat16, "cpu")
            self.assertEqual(
                {k: tuple(v.shape) for k, v in buffers.items()},
                {
                    "hidden_states": (tokens, 4096),
                    "residual": (tokens, 4, 4096),
                    "post": (tokens, 4, 1),
                    "comb": (tokens, 4, 4),
                },
            )
            self.assertEqual(buffers["post"].dtype, torch.float32)
            self.assertEqual(buffers["comb"].dtype, torch.float32)
            self.assertEqual(buffers["residual"].dtype, torch.bfloat16)
            self.assertTrue(
                all(torch.count_nonzero(v).item() == 0 for v in buffers.values())
            )
        with self.assertRaises(ValueError):
            allocate_intermediate(config, 1, torch.float16, "cpu")
