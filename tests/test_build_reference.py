import contextlib
import io
import json
import unittest
from unittest.mock import patch

from glm53_setup import build_reference
from glm53_setup.config import ROOT

LOCK = {
    "platform": "linux/arm64",
    "image": "nvcr.io/example/base@sha256:" + "0" * 64,
    "reference_candidate": {"tag": "glm53-reference:test"},
}


class BuildCommandTests(unittest.TestCase):
    def test_the_build_is_pinned_to_the_lock_and_rooted_at_the_checkout(self):
        command = build_reference.build_command(LOCK)
        self.assertEqual(command[:2], ["docker", "build"])
        self.assertEqual(command[command.index("--platform") + 1], "linux/arm64")
        self.assertEqual(
            command[command.index("--build-arg") + 1], "BASE_IMAGE=" + LOCK["image"]
        )
        self.assertEqual(command[command.index("-t") + 1], "glm53-reference:test")
        self.assertEqual(
            command[command.index("-f") + 1], str(ROOT / "docker/Dockerfile.reference")
        )
        self.assertEqual(command[-1], str(ROOT))

    def test_plan_prints_the_command_and_builds_nothing(self):
        out = io.StringIO()
        with (
            patch.object(build_reference, "load_lock", return_value=LOCK),
            patch.object(build_reference.subprocess, "run") as run,
            contextlib.redirect_stdout(out),
        ):
            build_reference.main(["--plan"])
        run.assert_not_called()
        self.assertEqual(
            json.loads(out.getvalue()), build_reference.build_command(LOCK)
        )


if __name__ == "__main__":
    unittest.main()
