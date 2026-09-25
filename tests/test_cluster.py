import contextlib
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from glm53_setup import cluster, server, server_config, switch


@contextlib.contextmanager
def rooted(root):
    """Redirect the rank state and the attempt records under ``root``."""
    with (
        patch.object(server, "STATE", root / "state"),
        patch.object(cluster, "RECORDS", root / "records"),
    ):
        yield


class ClusterOwnershipTests(unittest.TestCase):
    def test_cancelled_reserved_job_cannot_launch_after_transport_returns(self):
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        launch = {
            "manifest": server_config.freeze(profile, {}),
            "config_path": "/srv/glm53/state/server.toml",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            rooted(Path(tmp)),
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

    def test_a_replayed_start_finds_the_attempt_in_flight_and_does_not_launch_twice(
        self,
    ):
        # The first start reached the rank and the connection dropped before the
        # reply (2026-09-20: `ssh-unavailable` at start failed a switch that had
        # already stopped both ranks). The replay must not raise and must not
        # spawn a second supervisor; a finished attempt still refuses.
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        launch = {
            "manifest": server_config.freeze(profile, {}),
            "config_path": "/srv/glm53/state/server.toml",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            rooted(Path(tmp)),
            patch.object(cluster.subprocess, "Popen") as spawn,
        ):
            identity = cluster.rpc("reserve", 0, launch)
            self.assertEqual(cluster.rpc("start", 0, identity), {"started": True})
            self.assertEqual(
                cluster.rpc("start", 0, identity), {"started": True, "replayed": True}
            )
            self.assertEqual(spawn.call_count, 1)
            record = Path(identity["record"])
            (record / "finished.json").write_text('{"status": "stopped"}')
            with self.assertRaises(ValueError):
                cluster.rpc("start", 0, identity)

    def test_warmup_rpc_runs_only_for_the_owned_running_head_when_requested(self):
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        launch = {
            "manifest": server_config.freeze(profile, {}),
            "config_path": "/srv/glm53/state/server.toml",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            rooted(Path(tmp)),
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
            off["manifest"] = server_config.freeze(off["manifest"]["profile"], {})
            quiet = cluster.rpc("reserve", 0, off)
            self.assertEqual(cluster.rpc("warmup", 0, quiet), {"skipped": True})

    def test_a_reserved_attempt_is_named_as_the_launcher_names_its_container(self):
        with tempfile.TemporaryDirectory() as tmp, rooted(Path(tmp)):
            identity = cluster.rpc("reserve", 1, {"manifest": {"fingerprint": "f"}})
        self.assertEqual(identity["name"], server.container_name(1, identity["run_id"]))

    def test_swapped_attempt_record_is_not_allowed_to_stop_a_container(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            rooted(Path(tmp)),
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

    def test_transient_reads_retry_and_only_harmless_mutations_are_replayed(self):
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
        # A mutation is replayed only where the rank makes the replay harmless:
        # stop is idempotent and start answers "already started" for an attempt
        # in flight. reserve would mint a second identity, install and warmup
        # would run twice.
        for action, attempts in (
            ("start", 3),
            ("stop", 3),
            ("reserve", 1),
            ("install", 1),
            ("warmup", 1),
        ):
            with (
                patch.object(cluster.subprocess, "run", return_value=failed) as run,
                patch.object(cluster.time, "sleep"),
                self.assertRaises(RuntimeError),
            ):
                backend.call(action, 0, {})
            self.assertEqual(run.call_count, attempts, action)

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

    def test_a_resumed_pair_gets_what_a_completed_switch_gives_it(self):
        # 2026-09-20: resume confirmed a pair whose observation was lost, and the
        # ladder and the profile file had to be supplied by hand.
        def unconfirmed(status, rows):
            return {
                "status": status,
                "error": "OperationFailure",
                "failure": {"reason": "ssh-unavailable"},
                "recovery_observation_failure": {"reason": "ssh-unavailable"},
                "assets": [{"rank": 0}, {"rank": 1}],
                "recovery_assets": [{"rank": 0}, {"rank": 1}],
                rows: [
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
                "new" if rows == "recovery" else "recovery": [],
            }

        def backend():
            fake = MagicMock()
            fake.current.side_effect = [
                {"name": f"owned-{i}", "fingerprint": "fixed"} for i in (0, 1)
            ]
            fake.prepare.side_effect = [{"rank": 0}, {"rank": 1}]
            fake.install.return_value = {"installed": True}
            fake.warmup.return_value = {"passed": True}
            return fake

        fake = backend()
        result = cluster.resume(
            fake, unconfirmed("readiness-unconfirmed", "new"), config="text"
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual([c.args[0] for c in fake.install.call_args_list], [0, 1])
        self.assertEqual(result["config"][1], {"rank": 1, "installed": True})
        fake.warmup.assert_called_once()
        self.assertEqual(result["warmup"], {"passed": True})

        fake = backend()
        result = cluster.resume(fake, unconfirmed("readiness-unconfirmed", "new"))
        fake.install.assert_not_called()  # no text given: the files stay as they are
        fake.warmup.assert_called_once()

        # A failed ladder is recorded and leaves the pair complete, as in a switch.
        fake = backend()
        fake.warmup.side_effect = RuntimeError("ladder")
        result = cluster.resume(fake, unconfirmed("readiness-unconfirmed", "new"))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["warmup"]["failed"])

        # The recovered old pair keeps its file, and a recovery is never warmed.
        fake = backend()
        result = cluster.resume(
            fake,
            unconfirmed("recovery-readiness-unconfirmed", "recovery"),
            config="text",
        )
        self.assertEqual(result["status"], "failed")
        fake.install.assert_not_called()
        fake.warmup.assert_not_called()

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
            "manifest": server_config.freeze(server_config.loads(text), {}),
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
            rooted(Path(tmp)),
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
            rooted(Path(tmp)),
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
            rooted(Path(tmp)),
        ):
            config, identity = self.running(tmp, text)
            with self.assertRaises(ValueError):
                cluster.rpc("install", 0, {"identity": identity, "text": other})
            self.assertFalse(config.exists())

    def test_install_accepts_a_launch_time_allocator_override(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        with (
            tempfile.TemporaryDirectory() as tmp,
            rooted(Path(tmp)),
        ):
            config, identity = self.running(tmp, text)
            identity["launch"]["manifest"] = server_config.freeze(
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
            rooted(Path(tmp)),
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
            server_config.freeze(server_config.loads(call.kwargs["config"])),
        )

    def test_no_send_config_keeps_the_remote_file_untouched(self):
        self.assertIsNone(self.run_switch(["--no-send-config"]).kwargs["config"])


class ReadinessPollTests(unittest.TestCase):
    def poll(self, tail, full):
        text = EXAMPLE.read_text(encoding="utf-8")
        calls = []

        def logs(args, **kwargs):
            calls.append(args)
            return tail if "--tail" in args else full

        response = MagicMock()
        response.__enter__.return_value.status = 200
        with (
            tempfile.TemporaryDirectory() as tmp,
            rooted(Path(tmp)),
        ):
            launch = {
                "manifest": server_config.freeze(server_config.loads(text), {}),
                "config_path": "/srv/glm53/state/server.toml",
            }
            identity = cluster.rpc("reserve", 0, launch)
            (Path(tmp) / "state").mkdir()
            cluster.write_json(
                Path(tmp) / "state/startup-rank0.json", {"name": identity["name"]}
            )
            with (
                patch.object(
                    cluster,
                    "inspect_attempt",
                    return_value={"State": {"Running": True}},
                ),
                patch.object(cluster.subprocess, "check_output", side_effect=logs),
                patch.object(
                    cluster.model_http, "open_response", return_value=response
                ),
            ):
                return cluster.rpc("poll", 0, identity), calls

    def test_a_long_running_head_is_ready_after_its_startup_line_left_the_tail(self):
        # The supervisor reads /metrics every two seconds; the access log pushes the
        # startup line out of the last 200 lines within minutes, and a resume
        # then never confirmed a healthy pair.
        noise = b"GET /metrics HTTP/1.1 200 OK\n" * 200
        result, calls = self.poll(noise, b"Application startup complete.\n" + noise)
        self.assertEqual(result, {"ready": True})
        self.assertEqual(len(calls), 2)

    def test_a_fresh_head_reads_only_the_tail(self):
        result, calls = self.poll(b"Application startup complete.\n", b"")
        self.assertEqual(result, {"ready": True})
        self.assertEqual(len(calls), 1)

    def test_a_head_that_never_logged_startup_is_not_ready(self):
        result, _ = self.poll(b"loading\n", b"loading\n")
        self.assertEqual(result, {"ready": False})


class LostPair:
    """Two ranks whose first readiness observation fails with `lost`."""

    def __init__(self, lost, new_start_fails=False):
        self.lost = lost
        self.new_start_fails = new_start_fails
        self.running = {
            rank: {"name": f"old-{rank}", "fingerprint": "old", "launch": "old"}
            for rank in (0, 1)
        }

    def current(self, rank):
        return self.running.get(rank)

    def prepare(self, rank, launch, *, recovery=False):
        return {"common": launch}

    def reserve(self, rank, launch, *, recovery=False):
        return {"name": f"{launch}-{rank}", "fingerprint": launch, "launch": launch}

    def stop(self, rank, identity):
        self.running.pop(rank, None)

    def start(self, rank, identity):
        if self.new_start_fails and identity["launch"] == "new":
            raise RuntimeError("launch failed")
        self.running[rank] = identity

    def ready(self, rows):
        lost, self.lost = self.lost, None
        if lost:
            raise lost

    def warmup(self, rows):
        return {"passed": True}


class SwitchVocabularyTests(unittest.TestCase):
    """What the transport raises the switch recognises; what it writes resume reads."""

    def test_the_words_are_the_ones_journals_and_docs_name(self):
        self.assertEqual(
            (
                switch.TRANSPORT_TIMEOUT,
                switch.SSH_UNAVAILABLE,
                switch.READINESS_UNCONFIRMED,
                switch.RECOVERY_READINESS_UNCONFIRMED,
            ),
            (
                "transport-timeout",
                "ssh-unavailable",
                "readiness-unconfirmed",
                "recovery-readiness-unconfirmed",
            ),
        )

    def lost(self, **run):
        backend = cluster.SSHBackend(["head", "peer"], "/srv/model", None, 30)
        with (
            patch.object(cluster.subprocess, "run", **run),
            patch.object(cluster.time, "sleep"),
            self.assertRaises(switch.OperationFailure) as caught,
        ):
            backend.call("poll", 1, {})
        return caught.exception

    def test_a_lost_observation_round_trips_from_the_transport_to_resume(self):
        losses = {
            switch.TRANSPORT_TIMEOUT: self.lost(
                side_effect=subprocess.TimeoutExpired([], 120)
            ),
            switch.SSH_UNAVAILABLE: self.lost(
                return_value=subprocess.CompletedProcess([], 255, "", "")
            ),
        }
        for reason, error in losses.items():
            self.assertEqual(error.evidence["reason"], reason)
            self.assertTrue(switch.observation_lost(error))
            for new_start_fails, status, resumed in (
                (False, switch.READINESS_UNCONFIRMED, "complete"),
                (True, switch.RECOVERY_READINESS_UNCONFIRMED, "failed"),
            ):
                with self.subTest(reason=reason, status=status):
                    pair = LostPair(error, new_start_fails)
                    reports = []
                    with self.assertRaises(RuntimeError):
                        switch.switch(
                            pair,
                            "new",
                            save=lambda r: reports.append(copy.deepcopy(r)),
                        )
                    self.assertEqual(reports[-1]["status"], status)
                    result = cluster.resume(pair, reports[-1])
                    self.assertEqual(result["status"], resumed)

    def test_only_a_lost_poll_is_a_lost_observation(self):
        failed = self.lost(return_value=subprocess.CompletedProcess([], 1, "", ""))
        self.assertEqual(failed.evidence["reason"], "remote-operation-failed")
        for error in (
            failed,
            switch.OperationFailure("stop", 0, switch.TRANSPORT_TIMEOUT),
            switch.OperationFailure("ready", 1, switch.SSH_UNAVAILABLE, 255),
            RuntimeError(switch.TRANSPORT_TIMEOUT),
        ):
            with self.subTest(error=str(error)):
                self.assertFalse(switch.observation_lost(error))

    def test_resume_refuses_every_other_status(self):
        rows = [
            {"rank": i, "identity": {"name": f"owned-{i}", "fingerprint": "fixed"}}
            for i in (0, 1)
        ]
        for status in ("complete", "failed", "", switch.TRANSPORT_TIMEOUT):
            backend = MagicMock()
            with (
                self.subTest(status=status),
                self.assertRaisesRegex(ValueError, "unconfirmed two-rank readiness"),
            ):
                cluster.resume(backend, {"status": status, "new": rows})
            backend.current.assert_not_called()
            backend.ready.assert_not_called()
