import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import verify_download
from glm53_setup.config import MODEL, REVISION


class TransferStateTests(unittest.TestCase):
    def test_wait_does_not_resume_or_hang_on_paused_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {"status": "paused", "model": MODEL, "revision": REVISION}
            (root / "download-status.json").write_text(json.dumps(state))
            with (
                patch.object(verify_download, "STATE", root),
                patch.object(verify_download.subprocess, "run") as run,
                self.assertRaises(SystemExit) as raised,
            ):
                verify_download.main(
                    ["--hf", "unused", "--output", str(root / "out"), "--wait"]
                )
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()
            self.assertEqual(
                json.loads((root / "out/checksum-status.json").read_text())["status"],
                "paused",
            )


if __name__ == "__main__":
    unittest.main()
