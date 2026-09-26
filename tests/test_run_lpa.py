import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup.validation import run_lpa

KEYS = [
    "baseline_token_equal",
    "oracle_token_equal",
    "max_shared_logprob_error",
    "oracle_full_mlp_equal",
    "oracle_full_mlp_logprob_error",
    "finite",
    "state_finite",
    "oracle_active_state_equal",
    "state_comparisons",
    "passed",
]


def mode(tokens=(1, 2), logprobs=None, errors=({"finite": True, "max_abs": 0},)):
    return {
        "token_ids": list(tokens),
        "logprobs": logprobs or [{"1": -0.5, "9": -2.0}, {"2": -0.25}],
        "workers": [{"state_errors": [dict(error) for error in errors]}],
    }


def modes(**changed):
    names = ("capture", "off", "oracle_full_mlp", "oracle", "restored")
    return {name: changed.get(name) or mode() for name in names}


class AssessCaseTests(unittest.TestCase):
    """One replay length: what makes the oracle replay a pass."""

    def test_identical_replays_pass_with_the_keys_in_record_order(self):
        verdict = run_lpa.assess_case(modes())
        self.assertEqual(list(verdict), KEYS)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["max_shared_logprob_error"], 0)
        self.assertEqual(verdict["state_comparisons"], 2)

    def test_logprob_error_is_taken_over_tokens_both_runs_report(self):
        oracle = mode(logprobs=[{"1": -0.75}, {"2": -0.25, "7": -9.0}])
        verdict = run_lpa.assess_case(modes(oracle=oracle))
        self.assertEqual(verdict["max_shared_logprob_error"], 0.25)
        self.assertEqual(verdict["oracle_full_mlp_logprob_error"], 0)
        self.assertTrue(verdict["passed"])  # logprob error is recorded, not judged

    def test_no_shared_token_leaves_the_oracle_error_unknown(self):
        oracle = mode(logprobs=[{"8": -0.5}, {"8": -0.5}])
        self.assertIsNone(
            run_lpa.assess_case(modes(oracle=oracle))["max_shared_logprob_error"]
        )

    def test_each_failed_condition_fails_the_case(self):
        cases = {
            "baseline_token_equal": modes(capture=mode(tokens=(1, 3))),
            "oracle_token_equal": modes(oracle=mode(tokens=(1, 3))),
            "oracle_full_mlp_equal": modes(oracle_full_mlp=mode(tokens=(1, 3))),
            "finite": modes(restored=mode(logprobs=[{"1": math.nan}, {"2": 0.0}])),
            "state_finite": modes(off=mode(errors=({"finite": False, "max_abs": 0},))),
            "oracle_active_state_equal": modes(
                oracle=mode(errors=({"finite": True, "max_abs": 1e-3},))
            ),
        }
        for key, replay in cases.items():
            with self.subTest(condition=key):
                verdict = run_lpa.assess_case(replay)
                self.assertFalse(verdict[key])
                self.assertFalse(verdict["passed"])

    def test_a_replay_with_no_state_compared_proves_nothing(self):
        empty = mode(errors=())
        verdict = run_lpa.assess_case(modes(oracle=empty, oracle_full_mlp=empty))
        self.assertEqual(verdict["state_comparisons"], 0)
        self.assertTrue(verdict["oracle_active_state_equal"])
        self.assertFalse(verdict["passed"])

    def test_state_off_the_oracle_modes_may_differ(self):
        # Only the oracle replays must reproduce the captured state exactly.
        off = mode(errors=({"finite": True, "max_abs": 0.5},))
        self.assertTrue(run_lpa.assess_case(modes(off=off, capture=off))["passed"])


class ConfigureKwargsTests(unittest.TestCase):
    def args(self, *more):
        return run_lpa.parser().parse_args(
            ["--fixture", "/f", "--output", "/o", "--cut", "3", *more]
        )

    def test_each_replay_mode_configures_the_worker(self):
        args = self.args()
        self.assertEqual(
            run_lpa.configure_kwargs("capture", args, 127),
            {
                "mode": "capture",
                "cut": 3,
                "prompt_length": 127,
                "tail": 1,
                "profile": False,
                "verify_state": True,
                "skip_mlp": True,
                "skip_mla_queries": False,
                "allow_mtp": False,
            },
        )
        self.assertTrue(run_lpa.configure_kwargs("off", args, 1)["profile"])

    def test_the_full_mlp_oracle_is_the_oracle_with_its_mlp_kept(self):
        kwargs = run_lpa.configure_kwargs("oracle_full_mlp", self.args(), 4)
        self.assertEqual((kwargs["mode"], kwargs["skip_mlp"]), ("oracle", False))
        self.assertTrue(run_lpa.configure_kwargs("oracle", self.args(), 4)["skip_mlp"])

    def test_the_switches_come_from_the_command_line(self):
        kwargs = run_lpa.configure_kwargs(
            "oracle", self.args("--skip-mla-queries", "--mtp", "2"), 4
        )
        self.assertTrue(kwargs["skip_mla_queries"])
        self.assertIs(kwargs["allow_mtp"], True)


def split_modes(**changed):
    names = ("capture", "off", "split-self", "split", "restored")
    return {name: changed.get(name) or mode() for name in names}


class SplitCaseTests(unittest.TestCase):
    """The split-state check: split-self is off; split is recorded, not judged."""

    def test_split_self_equal_to_off_passes_whatever_split_does(self):
        split = mode(
            tokens=(1, 3),
            logprobs=[{"1": -0.75}, {"3": -1.0}],
            errors=({"finite": True, "max_abs": 0.5},),
        )
        verdict = run_lpa.assess_split_case(split_modes(split=split))
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["split_self_logprob_error"], 0)
        self.assertEqual(verdict["split_token_agreement"], 0.5)
        self.assertEqual(verdict["split_max_state_error"], 0.5)
        self.assertEqual(verdict["split_self_state_comparisons"], 1)

    def test_each_failed_condition_fails_the_split_case(self):
        cases = {
            "baseline_token_equal": split_modes(restored=mode(tokens=(1, 3))),
            "split_self_token_equal": split_modes(**{"split-self": mode((1, 3))}),
            "split_self_state_equal": split_modes(
                **{"split-self": mode(errors=({"finite": True, "max_abs": 1e-6},))}
            ),
            "finite": split_modes(split=mode(logprobs=[{"1": math.nan}, {"2": 0}])),
            "state_finite": split_modes(
                split=mode(errors=({"finite": False, "max_abs": 0},))
            ),
        }
        for key, replay in cases.items():
            with self.subTest(condition=key):
                verdict = run_lpa.assess_split_case(replay)
                self.assertFalse(verdict[key])
                self.assertFalse(verdict["passed"])
        empty = split_modes(**{"split-self": mode(errors=())})
        self.assertFalse(run_lpa.assess_split_case(empty)["passed"])

    def test_the_split_sequence_and_its_worker_settings(self):
        args = run_lpa.parser().parse_args(
            ["--fixture", "/f", "--output", "/o", "--cut", "0", "--split"]
        )
        self.assertEqual(
            run_lpa.replay_modes(args),
            ("capture", "off", "split-self", "split", "off"),
        )
        split = run_lpa.configure_kwargs("split", args, 5)
        self.assertEqual(split["mode"], "split")
        self.assertEqual(
            split["predictor_path"],
            str((Path("/o") / run_lpa.SPLIT_PROJECTOR).resolve()),
        )
        self.assertNotIn(
            "predictor_path", run_lpa.configure_kwargs("split-self", args, 5)
        )
        plain = run_lpa.parser().parse_args(["--fixture", "/f", "--output", "/o"])
        self.assertEqual(
            run_lpa.replay_modes(plain),
            ("capture", "off", "oracle_full_mlp", "oracle", "off"),
        )

    def test_split_refuses_mtp_and_query_skipping(self):
        for more in (["--mtp", "1"], ["--skip-mla-queries"]):
            args = run_lpa.parser().parse_args(
                ["--fixture", "/f", "--output", "/o", "--split", *more]
            )
            with self.subTest(more=more), self.assertRaises(ValueError):
                run_lpa.replay_modes(args)


class EngineRefused(Exception):
    pass


def refusing_vllm():
    """vLLM stand-ins whose engine constructor raises, as a failed load does."""

    def refuse(**kwargs):
        raise EngineRefused("engine could not load")

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = refuse, object
    config = types.ModuleType("vllm.config")
    compilation = types.ModuleType("vllm.config.compilation")
    compilation.CompilationMode = types.SimpleNamespace(NONE=0)
    return {"vllm": vllm, "vllm.config": config, "vllm.config.compilation": compilation}


class FailureRecordTests(unittest.TestCase):
    def test_an_engine_that_fails_to_load_leaves_a_failed_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with (
                patch.object(run_lpa, "read_fixture"),
                patch.dict(sys.modules, refusing_vllm()),
                self.assertRaises(EngineRefused),
            ):
                run_lpa.main(["--fixture", tmp, "--output", str(output)])
            record = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "EngineRefused('engine could not load')")
        self.assertEqual(record["cases"], [])


if __name__ == "__main__":
    unittest.main()
