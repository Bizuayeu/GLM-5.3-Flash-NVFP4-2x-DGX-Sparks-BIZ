import copy
import json
import tempfile
import unittest
from pathlib import Path

from glm53_setup.validation.summarize_fixture import (
    LABELS,
    assess_directory,
    assess_outputs,
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


if __name__ == "__main__":
    unittest.main()
