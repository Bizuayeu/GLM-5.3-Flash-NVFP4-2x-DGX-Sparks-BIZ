"""The base image's probe: what its payload must show before preparation passes."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import images

GB10 = {
    "capability": [12, 1],
    "gpu_mismatches": 0,
    "glm_architectures": ["Glm5NextForCausalLM"],
}


class ProbeVerdictTests(unittest.TestCase):
    def test_a_gb10_with_exact_integers_and_the_glm_model_passes(self):
        self.assertIs(images.probe_verdict(GB10), True)

    def test_each_missing_guarantee_fails(self):
        for name, change in {
            "another GPU": {"capability": [9, 0]},
            "wrong integer results": {"gpu_mismatches": 3},
            "no GLM architecture": {"glm_architectures": []},
        }.items():
            with self.subTest(case=name):
                self.assertIs(images.probe_verdict({**GB10, **change}), False)


class PrepareRunTests(unittest.TestCase):
    def prepare(self, payload):
        def run(args, **kwargs):
            stdout = json.dumps(payload) if args[:2] == ["docker", "run"] else "[]"
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        lock = {"image": "base@sha256:" + "0" * 64, "platform": "linux/arm64"}
        record = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with (
            patch.object(images, "load_lock", return_value=lock),
            patch.object(images.subprocess, "run", side_effect=run),
        ):
            try:
                images.run(record)
            finally:
                status = json.loads((record / "prepare-status.json").read_text())
        return status

    def test_a_passing_probe_is_recorded_as_prepared_not_validated(self):
        status = self.prepare(GB10)
        self.assertEqual(status["status"], "prepared_not_inference_validated")

    def test_a_failing_probe_is_recorded_and_raised(self):
        with self.assertRaisesRegex(RuntimeError, "GPU or model registration"):
            self.prepare({**GB10, "gpu_mismatches": 1})


if __name__ == "__main__":
    unittest.main()
