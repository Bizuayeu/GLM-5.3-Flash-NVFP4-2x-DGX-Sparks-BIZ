import importlib.util
import unittest
from pathlib import Path

from glm53_setup import startup_config


class RetentionConfigTests(unittest.TestCase):
    def test_native_checkpoint_interval_is_optional_and_explicit(self):
        profile = startup_config.load(
            Path(__file__).resolve().parents[1] / "examples/startup.example.toml"
        )
        self.assertNotIn(
            "--prefix-cache-retention-interval",
            startup_config.serve_args(profile, 0, "/hf/model"),
        )
        for interval in (0, 4352, 4608):
            profile["cache"]["prefix_cache_retention_interval"] = interval
            startup_config.validate(profile)
            for rank in (0, 1):
                args = startup_config.serve_args(profile, rank, "/hf/model")
                self.assertEqual(
                    args[args.index("--prefix-cache-retention-interval") + 1],
                    str(interval),
                )
        for invalid in (-1, True, None, "dense"):
            profile["cache"]["prefix_cache_retention_interval"] = invalid
            with self.assertRaises(ValueError):
                startup_config.validate(profile)


@unittest.skipUnless(importlib.util.find_spec("vllm"), "Pinned vLLM image required")
class NativeRetentionTests(unittest.TestCase):
    def test_checkpoint_interval_restores_mamba_boundaries_without_thinning_full_attention(
        self,
    ):
        from vllm.v1.core.single_type_kv_cache_manager import (
            FullAttentionManager,
            MambaManager,
        )
        from vllm.v1.kv_cache_interface import MambaSpec

        # This mask contract only consumes the block size; tensor/state layout
        # remains the responsibility of the separate real GPU fixture.
        spec = object.__new__(MambaSpec)
        object.__setattr__(spec, "block_size", 4)
        values = dict(
            start_block=0,
            end_block=4,
            alignment_tokens=4,
            kv_cache_spec=spec,
            use_eagle=False,
            reachable_boundaries=[16],
        )
        self.assertEqual(
            MambaManager.reachable_block_mask(**values, retention_interval=0),
            [False, False, False, True],
        )
        self.assertIsNone(
            MambaManager.reachable_block_mask(**values, retention_interval=4)
        )
        self.assertIsNone(
            FullAttentionManager.reachable_block_mask(**values, retention_interval=0)
        )
        self.assertIsNone(
            FullAttentionManager.reachable_block_mask(**values, retention_interval=4)
        )
