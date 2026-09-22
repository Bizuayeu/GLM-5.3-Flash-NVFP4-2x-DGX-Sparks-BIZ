import unittest

from glm53_setup.validation import reference_check


class CaseRowTests(unittest.TestCase):
    def test_a_case_passes_only_within_tolerance_and_with_an_empty_row_at_zero(self):
        row = reference_check.case_row(
            2051, error=0.5, tolerance=1.0, empty_row_zero=True
        )
        self.assertEqual(
            row,
            {
                "width": 2051,
                "max_abs_error": 0.5,
                "tolerance": 1.0,
                "empty_row_zero": True,
                "passed": True,
            },
        )
        self.assertTrue(reference_check.case_row(64, 1.0, 1.0, True)["passed"])
        self.assertFalse(reference_check.case_row(64, 1.01, 1.0, True)["passed"])
        self.assertFalse(reference_check.case_row(64, 0.0, 1.0, False)["passed"])


class VerdictTests(unittest.TestCase):
    def passing(self, **overrides):
        values = {
            "cache_error": 0.0,
            "cases": [{"passed": True}, {"passed": True}],
            "sensitivity": 16.0,
            "integration_error": 0.0,
        }
        values.update(overrides)
        return reference_check.verdict(**values)

    def test_every_condition_is_necessary(self):
        self.assertTrue(self.passing())
        # The cache decode must be exact, not merely close.
        self.assertFalse(self.passing(cache_error=1e-9))
        self.assertFalse(self.passing(cases=[{"passed": True}, {"passed": False}]))
        # An omitted tail must move the output by more than one unit.
        self.assertFalse(self.passing(sensitivity=1.0))
        self.assertFalse(self.passing(integration_error=1e-9))
        self.assertTrue(self.passing(cases=[]))


if __name__ == "__main__":
    unittest.main()
