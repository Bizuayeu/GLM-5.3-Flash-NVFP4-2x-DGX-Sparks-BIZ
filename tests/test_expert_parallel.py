import copy
import unittest
from pathlib import Path

from glm53_setup import server_config as config


class ExpertParallelConfigTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        # EP experiments isolate partitioning from the distributed combination.
        self.profile["mtp"]["enabled"] = False
        self.profile["lpa"]["enabled"] = False
        self.profile["cache"]["prefix_caching"] = False
        self.profile["cache"]["fused_unpack"] = False
        self.profile["runtime"]["fa2_attention"] = False
        self.profile["runtime"]["mla_decode_cpb"] = False

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
            ("runtime", "decode_graphs", True),
            ("context", "max_num_seqs", 4),
        ):
            trial = copy.deepcopy(self.profile)
            trial["runtime"]["expert_parallel"] = True
            trial[section][key] = value
            with self.assertRaisesRegex(ValueError, "EP"):
                config.validate(trial)

    def test_expert_observer_is_explicit_and_separate_from_other_workers(self):
        self.profile["validation"]["expert_worker"] = True
        self.profile["context"]["max_num_seqs"] = 2
        config.validate(self.profile)
        args = config.serve_args(self.profile, 0, "/hf/model")
        self.assertEqual(
            args[args.index("--worker-extension-cls") + 1],
            "glm53_setup.validation.expert_worker.ExpertFixtureWorker",
        )
        self.assertEqual(
            config.environment(self.profile, 0)["VLLM_SERVER_DEV_MODE"], "1"
        )
        for section, key, value in (
            ("validation", "component_worker", True),
            ("runtime", "pipeline_parallel_size", 2),
            ("mtp", "enabled", True),
            ("lpa", "enabled", True),
            ("cache", "prefix_caching", True),
            ("runtime", "decode_graphs", True),
        ):
            trial = copy.deepcopy(self.profile)
            trial[section][key] = value
            with self.assertRaises(ValueError):
                config.validate(trial)


if __name__ == "__main__":
    unittest.main()
