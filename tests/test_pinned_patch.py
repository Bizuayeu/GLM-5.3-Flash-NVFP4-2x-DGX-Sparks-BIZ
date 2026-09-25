import contextlib
import hashlib
import importlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from glm53_setup.__main__ import COMMANDS
from glm53_setup.runtime import pinned_patch

ROOT = Path(__file__).resolve().parents[1]
TARGET = "v1/example.py"
RECORD = "glm53-example-patch.json"


def patch_text(text):
    if text.startswith("# patched"):
        raise ValueError("Example patch already applied")
    return "# patched\n" + pinned_patch.replace_once(text, "a = 1\n", "a = 2\n")


class ReplaceOnceTests(unittest.TestCase):
    def test_replaces_exactly_one_unique_anchor(self):
        self.assertEqual(pinned_patch.replace_once("x\ny\n", "y\n", "z\n"), "x\nz\n")
        for text in ("x\n", "y\ny\n"):
            with self.assertRaises(ValueError):
                pinned_patch.replace_once(text, "y\n", "z\n")


class PinnedPatchCommandTests(unittest.TestCase):
    SOURCE = b"import os\na = 1\n"

    def package(self, directory):
        package = Path(directory) / "vllm"
        path = package / TARGET
        path.parent.mkdir(parents=True)
        path.write_bytes(self.SOURCE)
        return package

    def prepare(self, package, sha256=None):
        return pinned_patch.prepare(
            package,
            TARGET,
            sha256 or hashlib.sha256(self.SOURCE).hexdigest(),
            "example source hash mismatch",
            patch_text,
        )

    def run_main(self, package, *flags, sha256=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pinned_patch.main(
                ["--package", str(package), *flags],
                doc="example",
                target=TARGET,
                sha256=sha256 or hashlib.sha256(self.SOURCE).hexdigest(),
                prepare=lambda p: self.prepare(p, sha256),
                record=RECORD,
            )
        return json.loads(out.getvalue())

    def test_prepare_checks_the_pinned_hash_and_returns_bytes_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.package(directory)
            self.assertEqual(self.prepare(package), b"# patched\nimport os\na = 2\n")
            self.assertEqual((package / TARGET).read_bytes(), self.SOURCE)
            with self.assertRaisesRegex(ValueError, "example source hash mismatch"):
                self.prepare(package, "0" * 64)

    def test_check_reports_without_writing_and_the_default_writes_a_record(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.package(directory)
            report = self.run_main(package, "--check")
            self.assertTrue(report["check_only"])
            self.assertEqual((package / TARGET).read_bytes(), self.SOURCE)
            self.assertFalse((package.parent / RECORD).exists())
            report = self.run_main(package)
            self.assertEqual(
                set(report), {"source_sha256", "patched_sha256", "check_only"}
            )
            self.assertFalse(report["check_only"])
            patched = (package / TARGET).read_bytes()
            self.assertTrue(patched.startswith(b"# patched"))
            self.assertEqual(
                report["patched_sha256"], hashlib.sha256(patched).hexdigest()
            )
            self.assertEqual(
                json.loads((package.parent / RECORD).read_text(encoding="utf-8")),
                report,
            )
            # Running again meets the patched file, whose hash is no longer the pin.
            with self.assertRaises(ValueError):
                self.run_main(package)


DOCKERFILE = ROOT / "docker/Dockerfile.reference"

# Patches no reference image build runs, slated for deletion:
# patch_pipeline_layout is unneeded for the full-model split.
CANDIDATE_IMAGE_ONLY = {"patch_pipeline_layout"}


class ImageBuildContractTests(unittest.TestCase):
    def test_every_runtime_patch_is_built_into_the_image_or_named_as_left_out(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        built = set(
            re.findall(
                r"^RUN python3 -m glm53_setup\.runtime\.(patch_\w+)$", dockerfile, re.M
            )
        )
        # A subcommand of the package CLI resolves through its command table.
        for name in re.findall(r"^RUN python3 -m glm53_setup (\S+)$", dockerfile, re.M):
            if COMMANDS[name].startswith("runtime.patch_"):
                built.add(COMMANDS[name].removeprefix("runtime."))
        present = {
            path.stem for path in (ROOT / "glm53_setup/runtime").glob("patch_*.py")
        }
        self.assertEqual(built - present, set())
        self.assertEqual(built & CANDIDATE_IMAGE_ONLY, set())
        self.assertEqual(present - built, CANDIDATE_IMAGE_ONLY)

    def test_every_patch_the_image_build_runs_exposes_the_shared_surface(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        modules = re.findall(
            r"python3 -m (glm53_setup\.runtime\.patch_\w+)", dockerfile
        )
        for name in modules:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertTrue(callable(module.main))
                self.assertTrue(callable(module.prepare))
                out = io.StringIO()
                with (
                    contextlib.redirect_stdout(out),
                    self.assertRaises(SystemExit) as e,
                ):
                    module.main(["--help"])
                self.assertEqual(e.exception.code, 0)
                # Every build patch names the package it edits and has a dry mode;
                # patch_apc_lpa spells them --vllm-package/--dry-run.
                self.assertRegex(out.getvalue(), r"--(vllm-)?package")
                self.assertRegex(out.getvalue(), r"--(check|dry-run)")


if __name__ == "__main__":
    unittest.main()
