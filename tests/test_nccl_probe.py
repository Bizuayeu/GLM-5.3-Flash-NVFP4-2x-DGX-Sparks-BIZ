import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tools import nccl_probe


class NcclProbeCliTests(unittest.TestCase):
    def test_help_does_not_import_gpu_dependencies(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as result,
        ):
            nccl_probe.main(["--help"])
        self.assertEqual(result.exception.code, 0)

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rank.json"
            output.write_text("previous evidence", encoding="utf-8")
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as result,
            ):
                nccl_probe.main(
                    ["--rank", "0", "--head", "10.53.0.1", "--output", str(output)]
                )
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous evidence")
