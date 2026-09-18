import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from glm53_setup import cluster, server, server_config


class ClusterOwnershipTests(unittest.TestCase):
    def test_cancelled_reserved_job_cannot_launch_after_transport_returns(self):
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        launch = {
            "manifest": server.freeze(profile, {}),
            "config_path": "/srv/glm53/state/server.toml",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            identity = cluster.rpc("reserve", 0, launch)
            with patch.object(cluster, "inspect_attempt", return_value=None):
                cluster.rpc("stop", 0, identity)
            with patch.object(server, "main") as start:
                cluster.main(["job", "--record", identity["record"]])
                start.assert_not_called()
            finished = server.read_json(Path(identity["record"]) / "finished.json")
            self.assertEqual(finished["status"], "cancelled before launch")
            with self.assertRaises(ValueError):
                cluster.rpc("start", 0, identity)

    def test_warmup_rpc_runs_only_for_the_owned_running_head_when_requested(self):
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        launch = {
            "manifest": server.freeze(profile, {}),
            "config_path": "/srv/glm53/state/server.toml",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
            patch.object(
                server, "warmup_running", return_value={"passed": True}
            ) as run,
        ):
            identity = cluster.rpc("reserve", 0, launch)
            self.assertEqual(cluster.rpc("warmup", 1, identity), {"skipped": True})
            (Path(tmp) / "state").mkdir()
            server.write_json(
                Path(tmp) / "state/startup-rank0.json", {"name": "someone-else"}
            )
            with self.assertRaises(ValueError):
                cluster.rpc("warmup", 0, identity)
            server.write_json(
                Path(tmp) / "state/startup-rank0.json", {"name": identity["name"]}
            )
            self.assertEqual(cluster.rpc("warmup", 0, identity), {"passed": True})
            run.assert_called_once()
            off = copy.deepcopy(launch)
            off["manifest"]["profile"]["generation"]["warmup"] = False
            off["manifest"] = server.freeze(off["manifest"]["profile"], {})
            quiet = cluster.rpc("reserve", 0, off)
            self.assertEqual(cluster.rpc("warmup", 0, quiet), {"skipped": True})

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

    def test_resuming_recovery_keeps_candidate_failure_and_marks_old_profile_recovered(
        self,
    ):
        rows = [
            {
                "rank": i,
                "identity": {
                    "name": f"old-{i}",
                    "fingerprint": "old",
                    "launch": "old-profile",
                },
            }
            for i in (0, 1)
        ]
        report = {
            "status": "recovery-readiness-unconfirmed",
            "error": "OperationFailure",
            "failure": {"reason": "candidate-failed"},
            "new": [],
            "recovery": rows,
            "recovery_assets": [{"rank": 0}, {"rank": 1}],
            "recovery_observation_failure": {"reason": "ssh-unavailable"},
        }
        backend = MagicMock()
        backend.current.side_effect = [
            {"name": f"old-{i}", "fingerprint": "old"} for i in (0, 1)
        ]
        backend.prepare.side_effect = report["recovery_assets"]
        result = cluster.resume(backend, report)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["recovered"])
        self.assertEqual(result["failure"]["reason"], "candidate-failed")
        backend.start.assert_not_called()
        backend.stop.assert_not_called()


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/server.example.toml"


class ProfileInstallTests(unittest.TestCase):
    """The rank writes the profile text it was switched to, and only that."""

    def running(self, tmp, text):
        config = Path(tmp) / "site/server.toml"
        config.parent.mkdir()
        launch = {
            "manifest": server.freeze(server_config.loads(text), {}),
            "config_path": str(config),
        }
        identity = cluster.rpc("reserve", 0, launch)
        (Path(tmp) / "state").mkdir()
        cluster.write_json(
            Path(tmp) / "state/startup-rank0.json", {"name": identity["name"]}
        )
        return config, identity

    def test_install_writes_the_running_profile_and_keeps_the_old_file(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            config, identity = self.running(tmp, text)
            config.write_text("old = true\n", encoding="utf-8")
            result = cluster.rpc("install", 0, {"identity": identity, "text": text})
            self.assertEqual(config.read_bytes(), text.encode())
            backup = Path(result["backup"])
            self.assertEqual(backup.parent, config.parent)
            self.assertTrue(backup.name.startswith("server.toml.bak-"))
            self.assertTrue(backup.name.endswith("-unparsed"))
            self.assertEqual(backup.read_text(encoding="utf-8"), "old = true\n")
            again = cluster.rpc("install", 0, {"identity": identity, "text": text})
            self.assertEqual(again, {"unchanged": True})
            self.assertEqual(len(list(config.parent.iterdir())), 2)

    def test_install_without_an_old_file_makes_no_backup(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            config, identity = self.running(tmp, text)
            result = cluster.rpc("install", 0, {"identity": identity, "text": text})
            self.assertEqual(result, {"written": True})
            self.assertEqual([p.name for p in config.parent.iterdir()], ["server.toml"])

    def test_install_rejects_text_that_is_not_the_running_profile(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        other = text.replace("seed = 42", "seed = 43")
        self.assertNotEqual(text, other)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            config, identity = self.running(tmp, text)
            with self.assertRaises(ValueError):
                cluster.rpc("install", 0, {"identity": identity, "text": other})
            self.assertFalse(config.exists())

    def test_install_accepts_a_launch_time_allocator_override(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            config, identity = self.running(tmp, text)
            identity["launch"]["manifest"] = server.freeze(
                server_config.loads(text),
                {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            )
            identity["fingerprint"] = identity["launch"]["manifest"]["fingerprint"]
            cluster.write_json(Path(identity["record"]) / "identity.json", identity)
            cluster.rpc("install", 0, {"identity": identity, "text": text})
            self.assertEqual(config.read_bytes(), text.encode())

    def test_install_needs_the_attempt_to_be_the_running_rank(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            config, identity = self.running(tmp, text)
            cluster.write_json(
                Path(tmp) / "state/startup-rank0.json", {"name": "someone-else"}
            )
            with self.assertRaises(ValueError):
                cluster.rpc("install", 0, {"identity": identity, "text": text})
            self.assertFalse(config.exists())


class SwitchCommandTests(unittest.TestCase):
    def run_switch(self, extra):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                cluster, "switch", return_value={"status": "complete"}
            ) as switch,
        ):
            cluster.main(
                [
                    "switch",
                    "--config",
                    str(EXAMPLE),
                    "--remote-config",
                    "/srv/glm53/state/server.toml",
                    "--hosts",
                    "head",
                    "peer",
                    "--checkout",
                    "/srv/glm53/source",
                    "--output",
                    str(Path(tmp) / "out"),
                    *extra,
                ]
            )
            return switch.call_args

    def test_switch_sends_the_profile_text_by_default(self):
        call = self.run_switch([])
        self.assertEqual(call.kwargs["config"], EXAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(
            call.args[1]["manifest"],
            server.freeze(server_config.loads(call.kwargs["config"])),
        )

    def test_no_send_config_keeps_the_remote_file_untouched(self):
        self.assertIsNone(self.run_switch(["--no-send-config"]).kwargs["config"])
