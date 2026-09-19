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

    def test_profile_key_sets_the_switch_and_mounts_the_newer_modules(self):
        profile = config.load(ROOT / "examples/server.example.toml")
        # On in the template from 1.6.0. A profile written before the key has none,
        # keeps the reference path and its fingerprint.
        self.assertIs(profile["runtime"]["fa2_attention"], True)
        earlier = copy.deepcopy(profile)
        earlier["runtime"].pop("fa2_attention")
        config.validate(earlier)
        self.assertNotIn("GLM53_FA2_ATTENTION", config.environment(earlier, 0))
        config.validate(profile)
        for rank in (0, 1):
            self.assertEqual(
                config.environment(profile, rank)["GLM53_FA2_ATTENTION"], "1"
            )
        command = server.command(
            profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
        )
        for target in (
            ":/opt/glm53/glm53_setup/runtime/fa2_attention.py:ro",
            ":/opt/glm53/glm53_setup/runtime/reference_attention.py:ro",
            # The image's copy compiles one kernel per size; FA2 needs the fixed one.
            ":/opt/glm53/glm53_setup/runtime/fused_unpack.py:ro",
            ":/usr/local/lib/python3.12/dist-packages/glm53_reference.py:ro",
        ):
            self.assertTrue(any(v.endswith(target) for v in command), target)
        # An image built before the path shipped in it stays a valid recovery
        # target: the mounts serve it, and the marker it lacks is not required.
        bare = {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
        self.assertNotIn("fa2_support", server.image_capability_checks(profile, bare))
        profile["runtime"]["fa2_attention"] = False
        self.assertEqual(config.environment(profile, 0)["GLM53_FA2_ATTENTION"], "0")
        off = server.command(
            profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
        )
        self.assertFalse(any("fa2_attention.py" in v for v in off))

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
