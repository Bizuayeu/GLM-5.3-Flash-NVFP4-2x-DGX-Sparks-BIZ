import copy
import unittest
from pathlib import Path

from glm53_setup import startup_config as config


class ExpertParallelConfigTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(
            Path(__file__).resolve().parents[1] / "examples/startup.example.toml"
        )

    def test_opt_in_changes_only_expert_partitioning_on_both_ranks(self):
        for rank in (0, 1):
            before = config.serve_args(self.profile, rank, "/hf/model")
            trial = copy.deepcopy(self.profile)
            trial["runtime"]["expert_parallel"] = True
            config.validate(trial)
            after = config.serve_args(trial, rank, "/hf/model")
            self.assertEqual(after, before + ["--enable-expert-parallel"])
            self.assertEqual(after[after.index("--tensor-parallel-size") + 1], "2")
            self.assertNotIn("--data-parallel-size", after)
            self.assertNotEqual(
                config.fingerprint(trial), config.fingerprint(self.profile)
            )

    def test_independent_ep_scope_rejects_unqualified_combinations(self):
        for section, key, value in (
            ("lpa", "enabled", True),
            ("mtp", "enabled", True),
            ("cache", "prefix_caching", True),
            ("cache", "fused_unpack", True),
            ("runtime", "enforce_eager", False),
            ("context", "max_num_seqs", 4),
        ):
            trial = copy.deepcopy(self.profile)
            trial["runtime"]["expert_parallel"] = True
            trial[section][key] = value
            with self.assertRaisesRegex(ValueError, "EP"):
                config.validate(trial)


if __name__ == "__main__":
    unittest.main()
