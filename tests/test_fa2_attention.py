import copy
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import server
from glm53_setup import server_config as config
from glm53_setup.runtime import fa2_attention

ROOT = Path(__file__).resolve().parents[1]


class Fa2SwitchTests(unittest.TestCase):
    def test_only_prefill_sized_calls_take_the_kernel_and_only_when_switched_on(self):
        with patch.dict(os.environ, {"GLM53_FA2_ATTENTION": "1"}):
            self.assertFalse(fa2_attention.use_fa2(1))
            self.assertFalse(fa2_attention.use_fa2(6))  # k = 5 decode step
            self.assertTrue(fa2_attention.use_fa2(7))
            self.assertTrue(fa2_attention.use_fa2(2048))
        with patch.dict(os.environ, {"GLM53_FA2_ATTENTION": "0"}):
            self.assertFalse(fa2_attention.use_fa2(2048))
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLM53_FA2_ATTENTION", None)
            self.assertFalse(fa2_attention.use_fa2(2048))
        with patch.dict(os.environ, {"GLM53_FA2_ATTENTION": "yes"}):
            with self.assertRaises(ValueError):
                fa2_attention.use_fa2(2048)

    def test_profile_key_sets_the_switch_and_asks_the_image_for_the_path(self):
        profile = config.load(ROOT / "examples/server.example.toml")
        self.assertNotIn("GLM53_FA2_ATTENTION", config.environment(profile, 0))
        bare = {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
        self.assertNotIn("fa2_support", server.image_capability_checks(profile, bare))
        profile["runtime"]["fa2_attention"] = True
        config.validate(profile)
        for rank in (0, 1):
            self.assertEqual(
                config.environment(profile, rank)["GLM53_FA2_ATTENTION"], "1"
            )
        # The path, its dispatch and the fused unpack that takes its element count
        # at run time ship in the image; an image without them is refused, not
        # patched over with files from the checkout.
        self.assertIs(
            server.image_capability_checks(profile, bare)["fa2_support"], False
        )
        bare["Config"]["Env"].append("GLM53_FA2_ATTENTION_API=1")
        self.assertIs(
            server.image_capability_checks(profile, bare)["fa2_support"], True
        )
        command = server.command(
            profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
        )
        for name in ("fa2_attention.py", "fused_unpack.py", "glm53_reference.py"):
            self.assertFalse(any(v.endswith(f"{name}:ro") for v in command), name)
        profile["runtime"]["fa2_attention"] = False
        self.assertEqual(config.environment(profile, 0)["GLM53_FA2_ATTENTION"], "0")

    def test_the_switch_is_boolean_and_excludes_lpa(self):
        profile = config.load(ROOT / "examples/server.example.toml")
        for bad in (1, "true", None):
            p = copy.deepcopy(profile)
            p["runtime"]["fa2_attention"] = bad
            with self.assertRaisesRegex(ValueError, "fa2_attention"):
                config.validate(p)
        profile["runtime"]["fa2_attention"] = True
        profile["lpa"]["enabled"] = True
        # LPA hooks the reference computation (skip_mla_queries).
        with self.assertRaises(ValueError):
            config.validate(profile)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
class CompactCandidatesTests(unittest.TestCase):
    def test_rows_become_page_ranges_over_the_distinct_cache_rows(self):
        import torch

        indices = torch.tensor(
            [[-1, -1, 40, 7], [-1, 7, 9, 40], [-1, -1, -1, -1], [3, 7, 9, 40]]
        )
        rows, kv_indices, lengths = fa2_attention.compact_candidates(indices)
        self.assertEqual(lengths.tolist(), [2, 3, 0, 4])
        self.assertEqual(fa2_attention.indptr(lengths).tolist(), [0, 2, 5, 5, 9])
        # Every position points back at the physical row it came from, in row order.
        self.assertEqual(
            rows[kv_indices.long()].tolist(), [40, 7, 7, 9, 40, 3, 7, 9, 40]
        )
        self.assertEqual(kv_indices.dtype, torch.int32)
        self.assertEqual(sorted(set(rows.tolist()) - {0}), [3, 7, 9, 40])


if __name__ == "__main__":
    unittest.main()
