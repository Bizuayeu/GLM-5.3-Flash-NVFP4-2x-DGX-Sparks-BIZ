import copy
import importlib.util
import unittest
from types import SimpleNamespace as NS

from glm53_setup import server_config
from glm53_setup.config import ROOT
from glm53_setup.runtime.pipeline_state import validate_pipeline


class PipelineStartupTests(unittest.TestCase):
    def profile(self):
        profile = server_config.load(ROOT / "examples/server.example.toml")
        profile["mtp"]["enabled"] = False
        profile["lpa"]["enabled"] = False
        profile["cache"]["prefix_caching"] = False
        profile["cache"]["fused_unpack"] = False
        return profile

    def test_two_nodes_switch_from_tp_to_pp_with_explicit_stage_partition(self):
        profile = self.profile()
        before = server_config.fingerprint(profile)
        profile["runtime"]["pipeline_parallel_size"] = 2
        profile["runtime"]["pipeline_split_layer"] = 24
        server_config.validate(profile)
        self.assertNotEqual(before, server_config.fingerprint(profile))
        for rank in (0, 1):
            args = server_config.serve_args(profile, rank, "/hf/model")
            self.assertEqual(args[args.index("--nnodes") + 1], "2")
            self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "1")
            self.assertEqual(args[args.index("--pipeline-parallel-size") + 1], "2")
            self.assertNotIn("--enable-expert-parallel", args)
            self.assertEqual(
                server_config.environment(profile, rank)["VLLM_PP_LAYER_PARTITION"],
                "24,21",
            )

    def test_pp_rejects_missing_mla_stage_and_untested_combinations(self):
        profile = self.profile()
        profile["runtime"]["pipeline_parallel_size"] = 2
        for section, key, value in (
            ("runtime", "pipeline_parallel_size", 3),
            ("runtime", "pipeline_split_layer", 3),
            ("runtime", "pipeline_split_layer", 44),
            ("runtime", "expert_parallel", True),
            ("runtime", "decode_graphs", True),
            ("mtp", "enabled", True),
            ("lpa", "enabled", True),
            ("cache", "prefix_caching", True),
            ("cache", "fused_unpack", True),
            ("context", "max_num_seqs", 2),
        ):
            trial = copy.deepcopy(profile)
            trial[section][key] = value
            with self.assertRaises(ValueError):
                server_config.validate(trial)


class PipelineScopeTests(unittest.TestCase):
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
