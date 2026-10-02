import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import server
from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/server.example.toml"
MANIFEST = {"profile": {"frozen": True}, "fingerprint": "f" * 64}


class FreezeTests(unittest.TestCase):
    def freeze(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with (
            patch.object(server.settings, "freeze", return_value=MANIFEST),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            server.main(["freeze", "--config", str(EXAMPLE), *extra])
        return out.getvalue()

    def test_a_new_manifest_is_written_under_a_created_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "launch" / "frozen.json"
            printed = json.loads(self.freeze("--output", str(output)))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), MANIFEST)
        self.assertEqual(
            printed, {"fingerprint": MANIFEST["fingerprint"], "output": str(output)}
        )

    def test_an_existing_manifest_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "frozen.json"
            output.write_text("earlier launch\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.freeze("--output", str(output))
            self.assertEqual(output.read_text(encoding="utf-8"), "earlier launch\n")

    def test_freeze_needs_an_output_and_a_toml_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            launch = Path(tmp) / "launch.json"
            launch.write_text(
                json.dumps(config.freeze(config.load(EXAMPLE), {})), encoding="utf-8"
            )
            output = Path(tmp) / "frozen.json"
            for extra in ((), ("--output", str(output), "--launch", str(launch))):
                with self.subTest(extra=extra):
                    with self.assertRaises(SystemExit) as caught:
                        self.freeze(*extra)
                    self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())


class SnapshotTests(unittest.TestCase):
    def test_a_download_state_naming_another_snapshot_is_refused(self):
        profile = config.load(EXAMPLE)
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(server, "read_json", return_value={}),
                patch.object(
                    server.host,
                    "snapshot_from_state",
                    return_value=Path(tmp) / "other-snapshot",
                ),
                patch.object(server.host, "run", side_effect=AssertionError),
                self.assertRaisesRegex(ValueError, "pinned HF cache snapshot"),
            ):
                server.preflight(profile, EXAMPLE, 0, check_memory=False)


class RankTests(unittest.TestCase):
    def test_only_nonnegative_digit_ranks_are_accepted(self):
        self.assertEqual(server.rank_number("0"), 0)
        self.assertEqual(server.rank_number("2"), 2)
        for text in ("-1", "a", "1.5", "", " 1"):
            with self.subTest(text=text):
                with self.assertRaises(argparse.ArgumentTypeError):
                    server.rank_number(text)

    def test_the_parser_refuses_a_non_digit_rank(self):
        err = io.StringIO()
        with (
            contextlib.redirect_stderr(err),
            self.assertRaises(SystemExit) as caught,
        ):
            server.parser().parse_args(["plan", "--rank", "x"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("rank must be a nonnegative integer", err.getvalue())


if __name__ == "__main__":
    unittest.main()
