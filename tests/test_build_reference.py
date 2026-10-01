import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
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


def inspect_output(layers):
    """`docker image inspect` stdout for an image with this many layers."""
    diff_ids = ["sha256:" + f"{i:064x}" for i in range(layers)]
    return json.dumps([{"Id": "sha256:" + "f" * 64, "RootFS": {"Layers": diff_ids}}])


class LayerBudgetTests(unittest.TestCase):
    def test_counts_the_layers_and_the_headroom_to_the_overlay2_limit(self):
        for layers, headroom, warning in (
            (48, 77, False),
            (123, 2, False),
            (124, 1, True),
            (126, -1, True),
        ):
            with self.subTest(layers=layers):
                self.assertEqual(
                    build_reference.layer_budget(json.loads(inspect_output(layers))),
                    {
                        "layers": layers,
                        "limit": 125,
                        "headroom": headroom,
                        "warning": warning,
                    },
                )

    def test_record_dir_keeps_the_layer_count_and_warns_near_the_limit(self):
        for layers, warned in ((48, False), (124, True)):
            with (
                self.subTest(layers=layers),
                tempfile.TemporaryDirectory() as tmp,
                patch.object(build_reference, "load_lock", return_value=LOCK),
                patch.object(
                    build_reference.subprocess,
                    "run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0),
                        subprocess.CompletedProcess([], 0, inspect_output(layers)),
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as err,
            ):
                build_reference.main(["--record-dir", tmp])
                record = json.loads(
                    (Path(tmp) / "image-layers.json").read_text(encoding="utf-8")
                )
                self.assertEqual(record["layers"], layers)
                self.assertEqual(record["warning"], warned)
                self.assertEqual("125" in err.getvalue(), warned)


if __name__ == "__main__":
    unittest.main()
