import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import assess_benchmark
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


PASSING = {
    "completed": 3,
    "total_input_tokens": 96,
    "total_output_tokens": 192,
    "mean_ttft_ms": 20.0,
    "mean_tpot_ms": 50.0,
    "mean_e2el_ms": 3200.0,
    "output_throughput": 19.0,
}


class AssessBenchmarkMainTests(unittest.TestCase):
    def run_main(self, data, *counts):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            argv = ["assess_benchmark.py", str(path), *counts]
            out, err = io.StringIO(), io.StringIO()
            with (
                patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
                self.assertRaises(SystemExit) as caught,
            ):
                assess_benchmark.main()
        return caught.exception.code, out.getvalue()

    def test_a_complete_result_exits_zero(self):
        code, out = self.run_main(PASSING, "--requests", "3", "--output-tokens", "64")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["passed"])

    def test_an_incomplete_result_exits_one(self):
        code, out = self.run_main(
            {**PASSING, "completed": 2}, "--requests", "3", "--output-tokens", "64"
        )
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["checks"]["all_requests_completed"])

    def test_missing_expected_counts_are_a_usage_error(self):
        code, out = self.run_main(PASSING, "--requests", "3")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_nonpositive_expected_counts_are_refused(self):
        with self.assertRaises(ValueError):
            self.run_main(PASSING, "--requests", "0", "--output-tokens", "64")
