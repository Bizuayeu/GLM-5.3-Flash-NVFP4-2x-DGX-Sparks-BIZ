import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from glm53_setup.validation.summarize_fixture import (
    LABELS,
    assess_directory,
    assess_outputs,
    main,
)


class SummaryTests(unittest.TestCase):
    def records(self):
        row = {"token_ids": [7] * 16, "logprobs": [{"7": -0.5}] * 16}
        return {
            label: [copy.deepcopy(row)] * (2 if label == "a-b-batch" else 1)
            for label in LABELS
        }

    def test_command_json_is_not_treated_as_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, data in self.records().items():
                (root / (label + ".json")).write_text(json.dumps(data))
            (root / "command.json").write_text(json.dumps(["docker", "run"]))
            self.assertTrue(assess_directory(root)["passed"])

    def test_missing_output_and_probability_drift_are_not_success(self):
        data = self.records()
        data["forced-7"][0]["logprobs"][0] = {"7": -2.0}
        self.assertFalse(assess_outputs(data)["passed"])
        data = self.records()
        data["long"][0]["token_ids"] = []
        self.assertFalse(assess_outputs(data)["passed"])
        data.pop("long")
        with self.assertRaises(ValueError):
            assess_outputs(data)

    def test_boundary_prefill_is_checked_when_present(self):
        data = self.records()
        data["long-forced"] = copy.deepcopy(data["long"])
        self.assertTrue(assess_outputs(data)["passed"])
        data["long-forced"][0]["logprobs"][0] = {"7": -3.0}
        self.assertFalse(assess_outputs(data)["passed"])


ROW = {"token_ids": [7] * 16, "logprobs": [{"7": -0.5}] * 16}


class MainTests(unittest.TestCase):
    def run_main(self, records):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for label, rows in records.items():
                (directory / (label + ".json")).write_text(json.dumps(rows))
            out = io.StringIO()
            with (
                contextlib.redirect_stdout(out),
                self.assertRaises(SystemExit) as caught,
            ):
                main([str(directory)])
            saved = json.loads((directory / "assessment.json").read_text())
        self.assertEqual(json.loads(out.getvalue()), saved)
        return caught.exception.code, saved

    def records(self):
        return {label: [ROW] * (2 if label == "a-b-batch" else 1) for label in LABELS}

    def test_a_consistent_record_set_exits_zero_and_saves_the_assessment(self):
        code, saved = self.run_main(self.records())
        self.assertEqual(code, 0)
        self.assertTrue(saved["passed"])
        self.assertIsNone(saved["boundary_prefill"])

    def test_a_batch_mismatch_exits_one(self):
        records = self.records()
        records["b"] = [{"token_ids": [8] * 16, "logprobs": [{"8": -0.5}] * 16}]
        code, saved = self.run_main(records)
        self.assertEqual(code, 1)
        self.assertFalse(saved["checks"]["batch_b_tokens_equal"])

    def test_the_optional_long_forced_record_is_assessed_when_present(self):
        records = self.records()
        records["long-forced"] = [ROW]
        code, saved = self.run_main(records)
        self.assertEqual(code, 0)
        self.assertTrue(saved["checks"]["boundary_prefill_matches_decode"])

    def test_a_missing_record_fails_before_any_assessment_is_saved(self):
        records = self.records()
        del records["long"]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for label, rows in records.items():
                (directory / (label + ".json")).write_text(json.dumps(rows))
            with self.assertRaises(FileNotFoundError):
                main([str(directory)])
            self.assertFalse((directory / "assessment.json").exists())
