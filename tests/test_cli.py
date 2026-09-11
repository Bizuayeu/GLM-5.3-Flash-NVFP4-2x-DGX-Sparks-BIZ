import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PublicCliTests(unittest.TestCase):
    def invoke(self, *args, cwd=ROOT):
        return subprocess.run(
            [sys.executable, "-m", "glm53_setup", *args],
            cwd=cwd,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            check=False,
        )

    def test_version_and_help_do_not_require_gpu_packages(self):
        self.assertIn("0.1.0b1", self.invoke("--version").stdout)
        for command, option in [
            ("download", "--background"),
            ("verify-download", "--hf"),
            ("prepare-image", "--record-dir"),
            ("build-reference", "--plan"),
            ("service", "--site"),
            ("fixture-build", "--source"),
            ("fixture-run", "--fixture"),
            ("fixture-assess", "directory"),
            ("inspect-runtime", "snapshot"),
            ("probe-attention", "--output"),
            ("test-reference", "--output"),
            ("patch-reference", "--check"),
        ]:
            with self.subTest(command=command):
                result = self.invoke(command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(option, result.stdout)

    def test_checkout_resources_do_not_depend_on_callers_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.invoke("build-reference", "--plan", cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("sha256:", result.stdout)
            self.assertIn("Dockerfile.reference", result.stdout)


if __name__ == "__main__":
    unittest.main()
