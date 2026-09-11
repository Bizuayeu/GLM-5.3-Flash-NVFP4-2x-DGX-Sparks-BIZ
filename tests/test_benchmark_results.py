import unittest

from tools.assess_benchmark import assess_result


class BenchmarkResultsTests(unittest.TestCase):
    def test_successful_cli_exit_does_not_make_zero_completions_valid(self):
        data = {"completed": 0, "total_input_tokens": 0, "total_output_tokens": 0}
        self.assertFalse(assess_result(data, 3, 64)["passed"])

    def test_all_requests_and_expected_output_are_required(self):
        data = {
            "completed": 3,
            "total_input_tokens": 96,
            "total_output_tokens": 192,
            "mean_ttft_ms": 20.0,
            "mean_tpot_ms": 50.0,
            "mean_e2el_ms": 3200.0,
            "output_throughput": 19.0,
        }
        self.assertTrue(assess_result(data, 3, 64)["passed"])
        self.assertFalse(assess_result({**data, "completed": 2}, 3, 64)["passed"])
        self.assertFalse(
            assess_result({**data, "total_output_tokens": 191}, 3, 64)["passed"]
        )
        self.assertFalse(
            assess_result({**data, "output_throughput": float("nan")}, 3, 64)["passed"]
        )
