import contextlib
import io
import unittest

from glm53_setup.validation import apc_history
from glm53_setup.validation.apc_history import (
    answer_matches,
    common_prefix,
    expected_omission,
    history_cases,
)


class HistoryCaseTests(unittest.TestCase):
    def test_partial_or_under_repeated_timings_cannot_pose_as_full_functional_validation(
        self,
    ):
        base = [
            "--config",
            "unused",
            "--corpus",
            "unused",
            "--corpus-sha256",
            "0" * 64,
            "--output",
            "unused",
            "--block-tokens",
            "4352",
        ]
        for options in (
            ["--case-ids", "edit-50"],
            ["--timing-only", "--case-ids", "edit-50", "--repeats", "1"],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                apc_history.main(base + options)
            self.assertEqual(caught.exception.code, 2)

    def test_edits_change_the_answer_before_the_common_question(self):
        cases, *_ = history_cases(["early", "middle", "late", "end"])
        self.assertEqual(len(cases), 7)
        for case in cases:
            if case["id"].startswith("edit"):
                self.assertNotEqual(case["expected"], case["prime_expected"])
                self.assertEqual(
                    case["prime"]["messages"][-1], case["request"]["messages"][-1]
                )
                self.assertNotEqual(
                    case["prime"]["messages"][1], case["request"]["messages"][1]
                )
        self.assertEqual(common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(common_prefix([1, 2], [1, 2, 3]), 2)

    def test_stale_or_mixed_answer_is_not_accepted_as_format_variation(self):
        self.assertTrue(answer_matches("**MIA110001**", "MIA110001"))
        self.assertFalse(answer_matches("MIA550005", "MIA110001"))
        self.assertFalse(answer_matches("MIA110001 or MIA550005", "MIA110001"))
        for malformed in (
            "MIA1100019",
            "XMIA110001",
            "MIA110001_suffix",
            "MIA110001-9",
        ):
            self.assertFalse(answer_matches(malformed, "MIA110001"))


class ExpectedOmissionTests(unittest.TestCase):
    def test_every_eligible_row_is_skipped_on_each_mla_layer_after_the_cut(self):
        self.assertEqual(
            expected_omission(32, 7), {"35": 7, "39": 7, "43": 7}
        )  # The calibration cut; 44 is the model's last layer.
        self.assertEqual(expected_omission(40, 7), {"43": 7})
        self.assertEqual(expected_omission(0, 1, layers=4), {"3": 1})


class PolicyHitTests(unittest.TestCase):
    """The admission check both ranks must agree on, lifted out of main().

    It used to be a closure over the profile and the collective RPC, so only a
    running two-rank pair could reach it. The arithmetic it performs -- how
    many positions are eligible for approximation once the tail and the cached
    prefix are removed -- decides what every APC/LPA history reading means.
    """

    PROFILE = {"lpa": {"tail": 256, "cut": 32, "break_even_tokens": 512}}

    def worker(self, *, cached, prompt, skipped=None, limit=None):
        return {
            "policy": {
                "policy": {
                    "cached_tokens": cached,
                    "prompt_tokens": prompt,
                    "shared_cache_limit": limit,
                }
            },
            "lpa": {"mla_queries_skipped": skipped} if skipped is not None else None,
        }

    def call(self, workers, length, mode):
        return apc_history.policy_hits(self.PROFILE, lambda _: workers, length, mode)

    def test_off_mode_skips_nothing_however_long_the_prompt(self):
        pair = [self.worker(cached=64, prompt=8192) for _ in range(2)]
        hit, workers = self.call(pair, 8192, "off")
        self.assertEqual(hit, 64)
        self.assertEqual(len(workers), 2)

    def test_auto_mode_approximates_what_the_tail_and_the_cache_leave(self):
        # 8192 - 256 tail - 64 already cached = 7872 eligible, above break-even,
        # on every fourth layer after the cut.
        skipped = {str(layer): 7872 for layer in range(33, 45) if layer % 4 == 3}
        pair = [
            self.worker(cached=64, prompt=8192, skipped=skipped, limit=64)
            for _ in range(2)
        ]
        self.assertEqual(self.call(pair, 8192, "auto")[0], 64)

    def test_a_prompt_under_the_break_even_point_is_left_exact(self):
        pair = [self.worker(cached=0, prompt=600, skipped={}) for _ in range(2)]
        self.assertEqual(self.call(pair, 600, "auto")[0], 0)

    def test_a_scheduler_that_counted_different_tokens_is_refused(self):
        pair = [self.worker(cached=0, prompt=999) for _ in range(2)]
        with self.assertRaises(ValueError):
            self.call(pair, 1000, "off")

    def test_ranks_that_disagree_on_the_joint_hit_are_refused(self):
        pair = [self.worker(cached=64, prompt=8192), self.worker(cached=0, prompt=8192)]
        with self.assertRaises(ValueError):
            self.call(pair, 8192, "off")

    def test_a_single_rank_is_never_a_pair(self):
        with self.assertRaises(ValueError):
            self.call([self.worker(cached=0, prompt=100)], 100, "off")

    def test_work_actually_skipped_must_match_what_the_policy_promised(self):
        pair = [
            self.worker(cached=0, prompt=8192, skipped={"35": 1}, limit=0)
            for _ in range(2)
        ]
        with self.assertRaises(ValueError):
            self.call(pair, 8192, "auto")


if __name__ == "__main__":
    unittest.main()
