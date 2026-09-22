import copy
import unittest
from pathlib import Path

from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ROOT / "examples/server.example.toml"
AXL = ROOT / "examples/server.axl.example.toml"
SIX_GIB = 6 * 2**30


class AxlExampleTests(unittest.TestCase):
    def test_the_axl_example_serves_two_sequences_from_twice_the_kv(self):
        profile = config.load(AXL)
        config.validate(profile)
        derived = config.derived_checkpoint(profile)
        self.assertEqual(derived["requant_target"], "l")
        self.assertEqual(
            [o["target"] for o in derived["overlays"]], ["kda.py", "model.py"]
        )
        self.assertTrue(profile["runtime"]["prefix_page_dedup"])
        self.assertEqual(profile["context"]["max_num_seqs"], 2)
        self.assertEqual(profile["cache"]["kv_cache_memory_bytes"], SIX_GIB)
        for rank in (0, 1):
            args = config.serve_args(profile, rank, "/hf/mtp-view")
            self.assertEqual(args[args.index("--max-num-seqs") + 1], "2")
            self.assertEqual(
                args[args.index("--kv-cache-memory-bytes") + 1], str(SIX_GIB)
            )
            self.assertEqual(
                config.environment(profile, rank)["GLM53_PREFIX_PAGE_DEDUP"], "1"
            )
        # The two examples differ only where the option differs.
        defaults = config.load(DEFAULTS)
        self.assertEqual(profile["api"], defaults["api"])
        self.assertEqual(profile["nodes"], defaults["nodes"])
        self.assertEqual(profile["mtp"], defaults["mtp"])

    def test_a_kv_budget_above_three_gib_needs_the_derived_checkpoint(self):
        # Measured on the reference pair (2026-09-22): the pinned weights load 95.76 GiB
        # per rank and leave the head 5.5 GiB at 3 GiB of KV; the repacked ones load
        # 91.34 GiB and leave 10.5 GiB. Six GiB of KV crosses the 3 GiB reserve on the
        # pinned weights and stays above it on the repacked ones.
        defaults = config.load(DEFAULTS)
        defaults["cache"]["kv_cache_memory_bytes"] = SIX_GIB
        with self.assertRaisesRegex(ValueError, "derived_checkpoint"):
            config.validate(defaults)
        defaults["cache"]["kv_cache_memory_bytes"] = 3 * 2**30
        config.validate(defaults)
        axl = config.load(AXL)
        axl["runtime"]["derived_checkpoint"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "derived_checkpoint"):
            config.validate(axl)
        pinned_size = copy.deepcopy(axl)
        pinned_size["cache"]["kv_cache_memory_bytes"] = 3 * 2**30
        config.validate(pinned_size)  # a smaller KV needs no derived checkpoint


if __name__ == "__main__":
    unittest.main()
