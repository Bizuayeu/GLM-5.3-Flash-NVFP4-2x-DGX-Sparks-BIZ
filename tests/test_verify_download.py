import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup import verify_download
from glm53_setup.config import MODEL, REVISION
from glm53_setup.download import STATUS_FILE


class VerifyDownloadTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.state = self.root / "state"
        self.state.mkdir()
        self.output = self.root / "out"
        patcher = patch.object(verify_download, "STATE", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def download(self, status, **extra):
        record = {"model": MODEL, "revision": REVISION, "status": status, **extra}
        (self.state / STATUS_FILE).write_text(json.dumps(record))

    def run_main(self, *extra, returncode=0, sleep=None):
        argv = ["--hf", "hf", "--output", str(self.output), *extra]
        with (
            patch.object(
                verify_download.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=returncode),
            ) as run,
            patch.object(verify_download.time, "sleep", side_effect=sleep) as slept,
            self.assertRaises(SystemExit) as caught,
        ):
            verify_download.main(argv)
        saved = json.loads((self.output / "checksum-status.json").read_text())
        return caught.exception.code, saved, run, slept

    def test_a_download_of_another_revision_exits_one(self):
        for key in ("model", "revision"):
            with self.subTest(key=key):
                self.download("complete", **{key: "other"})
                code, saved, run, _ = self.run_main()
                self.assertEqual(code, 1)
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["reason"], "revision mismatch")
                run.assert_not_called()

    def test_paused_exits_two_and_failed_or_unfinished_without_wait_exit_one(self):
        for status, code, saved_status in (
            ("paused", 2, "paused"),
            ("failed", 1, "failed"),
            ("downloading", 1, "failed"),
        ):
            with self.subTest(status=status):
                self.download(status)
                actual, saved, run, _ = self.run_main()
                self.assertEqual(actual, code)
                self.assertEqual(saved["status"], saved_status)
                self.assertEqual(saved["reason"], "download not complete")
                run.assert_not_called()

    def test_a_paused_download_still_exits_two_while_waiting(self):
        self.download("paused")
        code, _, _, slept = self.run_main("--wait")
        self.assertEqual(code, 2)
        slept.assert_not_called()

    def test_wait_polls_until_the_download_completes(self):
        self.download("downloading")
        code, saved, run, slept = self.run_main(
            "--wait", sleep=lambda seconds: self.download("complete")
        )
        self.assertEqual(code, 0)
        self.assertEqual(slept.call_count, 1)
        self.assertEqual(saved["status"], "complete")
        run.assert_called_once()

    def test_the_cache_verify_return_code_is_propagated(self):
        for returncode, status in ((0, "complete"), (3, "failed")):
            with self.subTest(returncode=returncode):
                self.download("complete")
                code, saved, run, _ = self.run_main(returncode=returncode)
                self.assertEqual(code, returncode)
                self.assertEqual(saved["status"], status)
                self.assertEqual(saved["exit_code"], returncode)
                command = run.call_args.args[0]
                self.assertEqual(command[:4], ["hf", "cache", "verify", MODEL])
                self.assertIn("--fail-on-missing-files", command)
                self.assertIn("--fail-on-extra-files", command)

    def test_output_is_required(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            verify_download.main(["--hf", "hf"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
