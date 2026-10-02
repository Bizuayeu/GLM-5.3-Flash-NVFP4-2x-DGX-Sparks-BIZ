import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import config

REVISION = "0" * 40
IMAGE = "nvcr.io/nvidia/vllm@sha256:" + "0" * 64


class LoadLockTests(unittest.TestCase):
    def load(self, **lock):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.lock.json"
            values = {"model": "m", "revision": REVISION, "image": IMAGE, **lock}
            path.write_text(json.dumps(values), encoding="utf-8")
            with patch.object(config, "LOCK_PATH", path):
                return config.load_lock()

    def test_a_short_revision_is_refused(self):
        with self.assertRaisesRegex(
            ValueError, "^Model revision must be a full commit hash$"
        ):
            self.load(revision="0" * 7)

    def test_an_image_without_a_digest_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^Base image must be digest-pinned$"):
            self.load(image="nvcr.io/nvidia/vllm:latest")


if __name__ == "__main__":
    unittest.main()
