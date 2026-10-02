"""Contracts for the coordinator CLI's entry layer, and the two RPC gaps.

``cluster.main`` carried four subcommands in one run: each built its own
argument validation, its own SSH backend and its own result shaping. Only
``job`` was reachable from a test.

The RPC surface is left whole -- ``rpc(action, rank, value)`` is already
callable per action -- but two of its eight actions had no test. ``prepare``
is the one the 2026-09-20 launch refusal went through, so what it accepts
today is pinned here before 1.6.1 changes it.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from glm53_setup import cluster, server
from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def rooted(root):
    """Redirect the rank state and the attempt records under ``root``."""
    with (
        patch.object(server, "STATE", root / "state"),
        patch.object(cluster, "RECORDS", root / "records"),
    ):
        yield


def profile():
    return config.load(ROOT / "examples/server.example.toml")


class ActionTableTests(unittest.TestCase):
    def test_every_accepted_action_has_exactly_one_table_entry(self):
        choices = next(
            action.choices
            for action in cluster.parser()._actions
            if getattr(action, "dest", None) == "action"
        )
        self.assertEqual(sorted(choices), sorted(cluster.ACTIONS))
        for name, handler in cluster.ACTIONS.items():
            with self.subTest(action=name):
                self.assertTrue(callable(handler))

    def test_each_action_reaches_its_own_handler(self):
        for action in cluster.ACTIONS:
            with self.subTest(action=action):
                handler = MagicMock()
                with patch.dict(cluster.ACTIONS, {action: handler}):
                    cluster.main([action])
                handler.assert_called_once()

    def test_the_transported_actions_refuse_an_incomplete_invocation(self):
        # Each names the options it needs; a shared message would send the
        # operator to the wrong list.
        for action, fragment in (
            ("resume", "resume requires --output"),
            ("switch", "switch requires --config"),
        ):
            with self.subTest(action=action):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        cluster.main([action])
                self.assertIn(fragment, stderr.getvalue())

    def test_remote_paths_must_be_absolute_linux_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = [
                "switch",
                "--config",
                str(ROOT / "examples/server.example.toml"),
                "--remote-config",
                "relative/server.toml",
                "--hosts",
                "a",
                "b",
                "--checkout",
                "/srv/glm53",
                "--output",
                str(Path(tmp) / "out"),
            ]
            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                cluster.main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(
                "Remote paths must be absolute Linux paths", stderr.getvalue()
            )
            self.assertFalse((Path(tmp) / "out").exists())

    def test_a_nonpositive_readiness_timeout_is_refused(self):
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            cluster.main(
                [
                    "resume",
                    "--output",
                    "/tmp/x",
                    "--hosts",
                    "a",
                    "b",
                    "--checkout",
                    "/srv/glm53",
                    "--ready-timeout",
                    "0",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "resume requires --output, --hosts, --checkout and a positive timeout",
            stderr.getvalue(),
        )

    def test_resume_needs_one_host_per_recorded_rank(self):
        # A three-rank record resumed with two hosts used to reach rank 2 and
        # raise IndexError inside the SSH backend instead of naming the misuse.
        for status, key in (
            ("readiness-unconfirmed", "assets"),
            ("recovery-readiness-unconfirmed", "recovery_assets"),
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                server.write_json(
                    out / "result.json", {"status": status, key: [{}, {}, {}]}
                )
                stderr = io.StringIO()
                with (
                    contextlib.redirect_stderr(stderr),
                    patch.object(cluster, "resume") as resume,
                    self.assertRaises(SystemExit) as raised,
                ):
                    cluster.main(
                        [
                            "resume",
                            "--output",
                            str(out),
                            "--hosts",
                            "a",
                            "b",
                            "--checkout",
                            "/srv/glm53",
                        ]
                    )
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(
                    "resume needs one --hosts entry per recorded rank (3)",
                    stderr.getvalue(),
                )
                resume.assert_not_called()


class ActionOutcomeTests(unittest.TestCase):
    """What each action leaves on disk and prints once its work returns or raises."""

    def test_resume_saves_the_carried_record_and_prints_its_status(self):
        for source in (None, ROOT / "examples/server.example.toml"):
            with (
                self.subTest(config=source),
                tempfile.TemporaryDirectory() as tmp,
            ):
                out = Path(tmp)
                recorded = {"status": "readiness-unconfirmed", "assets": [{}, {}]}
                server.write_json(out / "result.json", recorded)
                seen = []

                def resume(backend, report, *, config, save):
                    seen.append((backend.hosts, report, config))
                    save({**report, "status": "saved-midway"})
                    seen.append(server.read_json(out / "result.json")["status"])
                    return {**report, "status": "complete"}

                printed = io.StringIO()
                with (
                    patch.object(cluster, "resume", resume),
                    contextlib.redirect_stdout(printed),
                ):
                    cluster.main(
                        [
                            "resume",
                            "--output",
                            str(out),
                            "--hosts",
                            "a",
                            "b",
                            "--checkout",
                            "/srv/glm53",
                            *(["--config", str(source)] if source else []),
                        ]
                    )
                text = source.read_text(encoding="utf-8") if source else None
                self.assertEqual(seen, [(["a", "b"], recorded, text), "saved-midway"])
                self.assertEqual(
                    server.read_json(out / "result.json"),
                    {**recorded, "status": "complete"},
                )
                self.assertEqual(
                    json.loads(printed.getvalue()),
                    {"status": "complete", "recovered": False, "output": str(out)},
                )

    def test_a_supervisor_whose_launch_raises_records_the_failure(self):
        with tempfile.TemporaryDirectory() as tmp, rooted(Path(tmp)):
            identity = cluster.rpc(
                "reserve",
                1,
                {"manifest": {"fingerprint": "f"}, "config_path": "/srv/s.toml"},
            )
            record = Path(identity["record"])
            with (
                patch.object(server, "main", side_effect=RuntimeError("oom")),
                self.assertRaisesRegex(RuntimeError, "oom"),
            ):
                cluster.main(["job", "--record", str(record)])
            self.assertEqual(
                server.read_json(record / "finished.json"),
                {"status": "failed", "error": "RuntimeError"},
            )
            self.assertIn("pid", server.read_json(record / "job.json"))

    def test_a_failed_switch_leaves_its_failure_beside_the_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            printed = io.StringIO()
            with (
                patch.object(
                    cluster, "switch", side_effect=ValueError("assets differ")
                ),
                contextlib.redirect_stdout(printed),
                self.assertRaisesRegex(ValueError, "assets differ"),
            ):
                cluster.main(
                    [
                        "switch",
                        "--config",
                        str(ROOT / "examples/server.example.toml"),
                        "--remote-config",
                        "/srv/glm53/state/server.toml",
                        "--hosts",
                        "a",
                        "b",
                        "--checkout",
                        "/srv/glm53",
                        "--output",
                        str(out),
                    ]
                )
            self.assertEqual(
                server.read_json(out / "failure.json"),
                {"error": "ValueError", "message": "assets differ"},
            )
            self.assertTrue((out / "launch.json").exists())
            self.assertEqual(printed.getvalue(), "")

    def test_the_transport_needs_at_least_two_hosts(self):
        args = cluster.parser().parse_args(
            ["resume", "--hosts", "only", "--checkout", "/srv/glm53"]
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            cluster.ssh_backend(args)

    def test_a_record_that_is_not_waiting_on_readiness_is_not_resumed(self):
        # The host count is checked only for a resumable record; any other status
        # is refused by resume itself, before a single SSH call.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            server.write_json(out / "result.json", {"status": "complete"})
            with (
                patch.object(cluster.subprocess, "run") as run,
                self.assertRaisesRegex(ValueError, "Only a recorded, unconfirmed"),
            ):
                cluster.main(
                    [
                        "resume",
                        "--output",
                        str(out),
                        "--hosts",
                        "a",
                        "b",
                        "--checkout",
                        "/srv/glm53",
                    ]
                )
            run.assert_not_called()
            self.assertEqual(
                json.loads((out / "result.json").read_text(encoding="utf-8")),
                {"status": "complete"},
            )


class RemoteProcedureGapTests(unittest.TestCase):
    """The two actions no test reached: ``current`` and ``prepare``."""

    def test_current_reports_nothing_when_this_rank_has_no_recorded_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            with rooted(Path(tmp)):
                self.assertIsNone(cluster.rpc("current", 0, None))

    def test_current_reports_nothing_when_the_recorded_container_is_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            server.write_json(
                root / "state/startup-rank0.json",
                {"name": "owned", "fingerprint": "f", "record": str(root)},
            )
            with (
                rooted(root),
                patch.object(
                    server, "inspect_owned", return_value={"State": {"Running": False}}
                ),
            ):
                self.assertIsNone(cluster.rpc("current", 0, None))

    def test_a_running_rank_without_a_recorded_config_path_cannot_be_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            server.write_json(
                root / "state/startup-rank0.json",
                {"name": "owned", "fingerprint": "f", "record": str(root)},
            )
            with (
                rooted(root),
                patch.object(
                    server, "inspect_owned", return_value={"State": {"Running": True}}
                ),
            ):
                with self.assertRaises(ValueError):
                    cluster.rpc("current", 0, None)

    def test_current_describes_the_running_launch_for_its_recovery(self):
        # This answer is what a switch restores if the candidate fails.
        running = profile()
        fingerprint = config.fingerprint(running)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            server.write_json(root / "settings.json", running)
            server.write_json(
                root / "state/startup-rank0.json",
                {
                    "name": "owned",
                    "fingerprint": fingerprint,
                    "record": str(root),
                    "config_path": "/srv/glm53/state/server.toml",
                },
            )
            with (
                rooted(root),
                patch.object(
                    server, "inspect_owned", return_value={"State": {"Running": True}}
                ) as inspect,
            ):
                self.assertEqual(
                    cluster.rpc("current", 0, None),
                    {
                        "name": "owned",
                        "fingerprint": fingerprint,
                        "launch": {
                            "manifest": {
                                "profile": running,
                                "fingerprint": fingerprint,
                            },
                            "config_path": "/srv/glm53/state/server.toml",
                        },
                    },
                )
            inspect.assert_called_once_with("owned", fingerprint)

    def test_current_refuses_settings_that_no_longer_match_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            server.write_json(root / "settings.json", profile())
            server.write_json(
                root / "state/startup-rank0.json",
                {
                    "name": "owned",
                    "fingerprint": "0" * 64,
                    "record": str(root),
                    "config_path": "/srv/glm53/state/server.toml",
                },
            )
            with (
                rooted(root),
                patch.object(
                    server, "inspect_owned", return_value={"State": {"Running": True}}
                ),
                self.assertRaisesRegex(ValueError, "no longer matches"),
            ):
                cluster.rpc("current", 0, None)

    def test_the_rpc_action_answers_one_json_operation_on_stdout(self):
        # The coordinator's SSH call writes the operation to stdin and reads stdout.
        request = json.dumps({"action": "current", "rank": 0, "value": None})
        stdout = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            rooted(Path(tmp)),
            patch("sys.stdin", io.StringIO(request)),
            contextlib.redirect_stdout(stdout),
        ):
            cluster.main(["rpc"])
        self.assertEqual(json.loads(stdout.getvalue()), None)

    def test_an_invalid_rank_is_refused_before_any_action_runs(self):
        # Rank 2 is a ring's third node (tests/test_cluster.py RingClusterTests).
        for rank in (-1, True, "0", None):
            with self.subTest(rank=rank):
                with self.assertRaises(ValueError):
                    cluster.rpc("current", rank, None)

    def test_an_unknown_operation_is_refused(self):
        with self.assertRaises(ValueError):
            cluster.rpc("no-such-operation", 0, None)

    def test_a_new_launch_needs_moe_order_marker_2_and_a_recovery_target_does_not(self):
        # Marker 1 also names the image whose sort mis-sized its buffer (46cd464).
        # prepare reaches this through launch_assets.inspect -> server.preflight.
        current = profile()
        current["runtime"]["canonical_moe_order"] = True
        for marker, recovery, expected in (
            ("2", False, True),
            ("2", True, True),
            ("1", False, False),
            ("1", True, True),
            (None, False, False),
            (None, True, False),
        ):
            with self.subTest(marker=marker, recovery=recovery):
                env = [f"GLM53_MOE_ORDER_API={marker}"] if marker else []
                image = {"Id": "sha256:" + "0" * 64, "Config": {"Env": env}}
                checks = config.image_capability_checks(
                    current, image, recovery=recovery
                )
                self.assertEqual(checks["moe_order_support"], expected)

    def test_prepare_tells_the_launch_checks_when_it_inspects_a_recovery_target(self):
        launch = {
            "manifest": config.freeze(profile(), {}),
            "config_path": "/srv/glm53/state/server.toml",
        }
        for value, expected in ((launch, False), ({**launch, "recovery": True}, True)):
            with self.subTest(recovery=expected):
                with patch.object(cluster.launch_assets, "inspect") as inspect:
                    cluster.rpc("prepare", 0, value)
                self.assertEqual(inspect.call_args.kwargs, {"recovery": expected})

    def test_the_ssh_backend_marks_a_recovery_target_inside_the_launch_it_sends(self):
        backend = cluster.SSHBackend.__new__(cluster.SSHBackend)
        sent = []
        backend.call = lambda action, rank, value=None: sent.append((action, value))
        launch = {"manifest": {}, "config_path": "/srv/server.toml"}
        backend.prepare(0, launch)
        backend.prepare(0, launch, recovery=True)
        backend.reserve(1, launch)
        backend.reserve(1, launch, recovery=True)
        self.assertEqual(
            sent,
            [
                ("prepare", launch),
                ("prepare", {**launch, "recovery": True}),
                ("reserve", launch),
                ("reserve", {**launch, "recovery": True}),
            ],
        )
        self.assertNotIn("recovery", launch)

    def test_a_reserved_recovery_attempt_starts_its_rank_as_a_recovery(self):
        for recovery in (False, True):
            with self.subTest(recovery=recovery), tempfile.TemporaryDirectory() as tmp:
                record = Path(tmp)
                launch = {"manifest": {}, "config_path": "/srv/server.toml"}
                if recovery:
                    launch["recovery"] = True
                identity = {
                    "rank": 1,
                    "run_id": "abc",
                    "record": str(record),
                    "launch": launch,
                }
                (record / "identity.json").write_text(json.dumps(identity))
                args = cluster.parser().parse_args(["job", "--record", str(record)])
                with (
                    patch.object(cluster, "owned_record", return_value=record),
                    patch.object(cluster.server, "main") as main,
                ):
                    cluster.act_job(None, args)
                self.assertEqual("--recovery" in main.call_args.args[0], recovery)

    def test_prepare_refuses_when_the_static_checks_did_not_pass(self):
        launch = {
            "manifest": config.freeze(profile(), {}),
            "config_path": "/srv/glm53/state/server.toml",
        }
        failed = {"passed": False, "checks": {}, "foreign_gpu_containers": []}
        with patch.object(server, "preflight", return_value=failed):
            with self.assertRaises(ValueError) as caught:
                cluster.rpc("prepare", 0, launch)
        self.assertIn("Static launch checks failed", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
