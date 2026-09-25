import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from glm53_setup import server
from tools import check_prefix_cache

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
        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "version"
        ]
        result = self.invoke("--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), version)
        for command, option in [
            ("download", "--background"),
            ("verify-download", "--hf"),
            ("prepare-image", "--record-dir"),
            ("build-reference", "--plan"),
            ("server", "--config"),
            ("cluster", "--remote-config"),
            ("fixture-build", "--source"),
            ("fixture-run", "--fixture"),
            ("fixture-assess", "directory"),
            ("inspect-runtime", "snapshot"),
            ("probe-attention", "--output"),
            ("test-reference", "--output"),
            ("patch-reference", "--check"),
            ("lpa-fixture", "--fixture"),
            ("apc-lpa-fixture", "--fixture"),
            ("apc-lpa-benchmark", "--cached-prefix-tokens"),
            ("apc-history", "--block-tokens"),
            ("lpa-corpus", "--source-byte-limit"),
            ("lpa-train", "--captures"),
            ("freedombench", "--benchmark-dir"),
            ("profile-assess", "--prefill-control"),
            ("indexer-overlap", "capture"),
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


class HelpTextTests(unittest.TestCase):
    def test_server_help_is_the_launcher_docstring(self):
        description = server.parser().description
        self.assertEqual(description, server.__doc__)
        self.assertTrue(description.startswith("The launcher of the TP=2 serving pair"))
        self.assertNotIn("experiment", description.lower())
        for path in re.findall(r"docs/[\w.-]+\.md", description):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_check_prefix_cache_usage_is_one_command_line(self):
        # argparse re-wraps --help, so the docstring is where the line lives.
        lines = [
            line.strip()
            for line in check_prefix_cache.__doc__.splitlines()
            if "tools/check_prefix_cache.py" in line
        ]
        self.assertEqual(len(lines), 1, lines)
        self.assertRegex(
            lines[0],
            r"^python tools/check_prefix_cache\.py --base-url \S+ --model \S+$",
        )


if __name__ == "__main__":
    unittest.main()
