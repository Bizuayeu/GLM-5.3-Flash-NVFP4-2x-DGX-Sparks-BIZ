import contextlib
import io
import unittest

from glm53_setup.validation import apc_history
from glm53_setup.validation.apc_history import (
    answer_matches,
    common_prefix,
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
