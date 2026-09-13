import unittest

from glm53_setup.validation.apc_history import (
    answer_matches,
    common_prefix,
    history_cases,
)


class HistoryCaseTests(unittest.TestCase):
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
