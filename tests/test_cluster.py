import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from glm53_setup import cluster, startup, startup_config


class ClusterOwnershipTests(unittest.TestCase):
    def test_cancelled_reserved_job_cannot_launch_after_transport_returns(self):
        profile = startup_config.load(
            Path(__file__).resolve().parents[1] / "examples/startup.example.toml"
        )
        launch = {
            "manifest": startup.freeze(profile, {}),
            "config_path": "/srv/glm53/state/startup.toml",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            identity = cluster.rpc("reserve", 0, launch)
            with patch.object(cluster, "inspect_attempt", return_value=None):
                cluster.rpc("stop", 0, identity)
            with patch.object(startup, "main") as start:
                cluster.main(["job", "--record", identity["record"]])
                start.assert_not_called()
            finished = startup.read_json(Path(identity["record"]) / "finished.json")
            self.assertEqual(finished["status"], "cancelled before launch")
            with self.assertRaises(ValueError):
                cluster.rpc("start", 0, identity)

    def test_swapped_attempt_record_is_not_allowed_to_stop_a_container(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            identity = cluster.rpc("reserve", 0, {"manifest": {"fingerprint": "test"}})
            identity["name"] = "unrelated-workload"
            with (
                patch.object(cluster, "inspect_attempt") as inspect,
                self.assertRaises(ValueError),
            ):
                cluster.rpc("stop", 0, identity)
            inspect.assert_not_called()

    def test_ssh_alias_cannot_be_a_command_line_option(self):
        with self.assertRaises(ValueError):
            cluster.SSHBackend(["-oProxyCommand=bad", "peer"], "/srv/model", None, 30)

    def test_transient_reads_retry_but_mutations_are_never_replayed(self):
        backend = cluster.SSHBackend(["head", "peer"], "/srv/model", None, 30)
        failed = subprocess.CompletedProcess(
            [], 255, stdout="", stderr="connect failed"
        )
        success = subprocess.CompletedProcess(
            [], 0, stdout='{"ready":false}', stderr=""
        )
        with (
            patch.object(
                cluster.subprocess, "run", side_effect=[failed, success]
            ) as run,
            patch.object(cluster.time, "sleep"),
        ):
            self.assertEqual(backend.call("poll", 0, {}), {"ready": False})
            self.assertEqual(run.call_count, 2)
        for action in ("start", "stop", "reserve"):
            with (
                patch.object(cluster.subprocess, "run", return_value=failed) as run,
                self.assertRaises(RuntimeError),
            ):
                backend.call(action, 0, {})
            self.assertEqual(run.call_count, 1)

    def test_read_retries_are_bounded_and_do_not_hide_asset_failures(self):
        backend = cluster.SSHBackend(["head", "peer"], "/srv/model", None, 30)
        for exit_code, attempts in ((255, 3), (1, 1)):
            failed = subprocess.CompletedProcess(
                [], exit_code, stdout="", stderr="failed"
            )
            with (
                patch.object(cluster.subprocess, "run", return_value=failed) as run,
                patch.object(cluster.time, "sleep"),
                self.assertRaises(RuntimeError),
            ):
                backend.call("prepare", 1, {})
            self.assertEqual(run.call_count, attempts)

    def test_readiness_resume_checks_saved_identities_and_assets_without_start(self):
        report = {
            "status": "readiness-unconfirmed",
            "error": "OperationFailure",
            "failure": {"reason": "ssh-unavailable"},
            "assets": [{"rank": 0}, {"rank": 1}],
            "new": [
                {
                    "rank": i,
                    "identity": {
                        "name": f"owned-{i}",
                        "fingerprint": "fixed",
                        "launch": "profile",
                    },
                }
                for i in (0, 1)
            ],
        }
        for changed in (False, True):
            backend = MagicMock()
            backend.current.side_effect = [
                {"name": "other" if changed else "owned-0", "fingerprint": "fixed"},
                {"name": "owned-1", "fingerprint": "fixed"},
            ]
            backend.prepare.side_effect = [{"rank": 0}, {"rank": 1}]
            if changed:
                with self.assertRaises(ValueError):
                    cluster.resume(backend, copy.deepcopy(report))
                backend.ready.assert_not_called()
            else:
                self.assertEqual(
                    cluster.resume(backend, copy.deepcopy(report))["status"], "complete"
                )
                backend.ready.assert_called_once()
            backend.start.assert_not_called()
            backend.stop.assert_not_called()

    def test_transport_timeout_never_exposes_subprocess_command(self):
        backend = cluster.SSHBackend(["head", "peer"], "/srv/model", None, 30)
        with (
            patch.object(
                cluster.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("secret-command", 120),
            ),
            patch.object(cluster.time, "sleep"),
            self.assertRaises(RuntimeError) as caught,
        ):
            backend.call("poll", 0, {})
        self.assertNotIn("secret-command", str(caught.exception))
