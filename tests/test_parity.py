import importlib.util
import math
import unittest

from glm53_setup.validation.parity import bf16_bound, judge, judge_tail


class Bf16BoundTests(unittest.TestCase):
    def test_two_bf16_ulps_at_the_largest_reference_magnitude(self):
        self.assertEqual(bf16_bound(4.0), 2 * 2**-7 * 4.0)
        self.assertEqual(bf16_bound(0.25), 2 * 2**-7)  # never below magnitude one

    def test_a_nonfinite_reference_does_not_widen_the_bound(self):
        self.assertEqual(bf16_bound(math.nan), 2 * 2**-7)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
    def test_the_constant_is_torchs_bf16_epsilon(self):
        import torch

        self.assertEqual(bf16_bound(1.0), 2 * torch.finfo(torch.bfloat16).eps)


class JudgeTests(unittest.TestCase):
    def test_judge_names_every_failure(self):
        good = {"finite": True, "max_abs_error": 0.01, "tolerance": 0.02}
        self.assertEqual(judge(good), {"passed": True, "reasons": []})
        self.assertEqual(judge({**good, "empty_row_zero": True})["passed"], True)
        self.assertEqual(judge({**good, "empty_row_zero": None})["passed"], True)
        bad = {
            "finite": False,
            "max_abs_error": 0.03,
            "tolerance": 0.02,
            "empty_row_zero": False,
        }
        self.assertEqual(
            judge(bad)["reasons"],
            ["non-finite", "error-above-bound", "empty-row-nonzero"],
        )

    def test_an_error_that_is_not_a_number_is_above_the_bound(self):
        case = {"finite": True, "max_abs_error": math.nan, "tolerance": 0.02}
        self.assertEqual(judge(case)["reasons"], ["error-above-bound"])

    def test_the_tail_must_be_within_bound_and_visibly_matter(self):
        tail = {"native_error": 0.01, "tolerance": 0.02, "omission_difference": 0.5}
        self.assertEqual(judge_tail(tail)["reasons"], ["tail-insensitive"])
        self.assertTrue(judge_tail({**tail, "omission_difference": 2})["passed"])
        self.assertEqual(
            judge_tail({**tail, "native_error": math.nan, "omission_difference": 2})[
                "reasons"
            ],
            ["error-above-bound"],
        )


if __name__ == "__main__":
    unittest.main()
