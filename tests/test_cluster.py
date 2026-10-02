import contextlib
import copy
import io
import json
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
                    switch.resume(backend, copy.deepcopy(report))
                backend.ready.assert_not_called()
            else:
                self.assertEqual(
                    switch.resume(backend, copy.deepcopy(report))["status"], "complete"
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
        result = switch.resume(
            fake, unconfirmed("readiness-unconfirmed", "new"), config="text"
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual([c.args[0] for c in fake.install.call_args_list], [0, 1])
        self.assertEqual(result["config"][1], {"rank": 1, "installed": True})
        fake.warmup.assert_called_once()
        self.assertEqual(result["warmup"], {"passed": True})

        fake = backend()
        result = switch.resume(fake, unconfirmed("readiness-unconfirmed", "new"))
        fake.install.assert_not_called()  # no text given: the files stay as they are
        fake.warmup.assert_called_once()

        # A failed ladder is recorded and leaves the pair complete, as in a switch.
        fake = backend()
        fake.warmup.side_effect = RuntimeError("ladder")
        result = switch.resume(fake, unconfirmed("readiness-unconfirmed", "new"))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["warmup"]["failed"])

        # The recovered old pair keeps its file, and a recovery is never warmed.
        fake = backend()
        result = switch.resume(
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
        result = switch.resume(backend, report)
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

    def test_install_writes_only_to_an_absolute_toml_path(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        for path in ("relative/server.toml", "/srv/glm53/state/server.json"):
            with (
                self.subTest(path=path),
                tempfile.TemporaryDirectory() as tmp,
                rooted(Path(tmp)),
            ):
                config, identity = self.running(tmp, text)
                identity["launch"]["config_path"] = path
                cluster.write_json(Path(identity["record"]) / "identity.json", identity)
                with self.assertRaisesRegex(ValueError, "absolute .toml path"):
                    cluster.rpc("install", 0, {"identity": identity, "text": text})
                self.assertFalse(config.exists())

    def test_install_refuses_an_identity_that_changed_after_reservation(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        with (
            tempfile.TemporaryDirectory() as tmp,
            rooted(Path(tmp)),
        ):
            config, identity = self.running(tmp, text)
            changed = {**identity, "name": "someone-else"}
            with self.assertRaisesRegex(ValueError, "Attempt identity changed"):
                cluster.rpc("install", 0, {"identity": changed, "text": text})
            self.assertFalse(config.exists())

    def test_an_attempt_record_outside_this_checkout_is_refused(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as elsewhere,
            rooted(Path(tmp)),
        ):
            config, identity = self.running(tmp, text)
            moved = {**identity, "record": elsewhere}
            cluster.write_json(Path(elsewhere) / "identity.json", moved)
            with self.assertRaisesRegex(ValueError, "belong to this checkout"):
                cluster.rpc("install", 0, {"identity": moved, "text": text})
            self.assertFalse(config.exists())

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
            contextlib.redirect_stdout(io.StringIO()) as printed,
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
            # The printed status is what an operator and a driver read.
            self.assertEqual(json.loads(printed.getvalue())["status"], "complete")
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


class AttemptStopTests(unittest.TestCase):
    """A stop signals the attempt's own supervisor and waits for it to exit."""

    PID = 4242

    @contextlib.contextmanager
    def supervised(self, cmdline=None):
        """A reserved attempt whose job.json names PID; its /proc cmdline is a file.

        ``cmdline`` is called with the resolved record (windows-latest TEMP is a
        short 8.3 path that ``resolve`` expands) and returns the file's bytes.
        """
        real = Path
        with tempfile.TemporaryDirectory() as tmp, rooted(Path(tmp)):
            identity = cluster.rpc("reserve", 0, {"manifest": {"fingerprint": "f"}})
            record = real(identity["record"]).resolve()
            cluster.write_json(record / "job.json", {"pid": self.PID})
            proc = real(tmp) / "cmdline"
            proc.write_bytes(
                cmdline(record)
                if cmdline
                else b"\0".join(
                    [b"python3", b"-m", b"glm53_setup.cluster", b"job", b"--record"]
                    + [str(record).encode(), b""]
                )
            )
            with (
                patch.object(
                    cluster,
                    "Path",
                    lambda *parts: (
                        proc if str(parts[0]).startswith("/proc/") else real(*parts)
                    ),
                ),
                patch.object(cluster.os, "kill") as kill,
                patch.object(cluster.time, "sleep") as sleep,
                patch.object(cluster, "inspect_attempt", return_value=None) as inspect,
            ):
                yield identity, proc, kill, sleep, inspect

    def test_a_pid_that_now_belongs_to_another_process_is_not_signalled(self):
        for name, cmdline in (
            ("another program", lambda record: b"sshd\0-D\0"),
            (
                "another attempt",
                lambda record: (
                    b"python3\0-m\0glm53_setup.cluster\0job\0--record\0"
                    + str(record.with_name("switch-other-r0")).encode()
                    + b"\0"
                ),
            ),
        ):
            with (
                self.subTest(name),
                self.supervised(cmdline) as (identity, _, kill, _, inspect),
            ):
                with self.assertRaisesRegex(ValueError, "no longer belongs"):
                    cluster.rpc("stop", 0, identity)
                kill.assert_not_called()
                inspect.assert_not_called()

    def test_the_supervisor_is_sent_sigterm_and_its_exit_awaited(self):
        with self.supervised() as (identity, proc, kill, sleep, inspect):
            sleep.side_effect = lambda seconds: proc.write_bytes(b"")
            inspect.return_value = {"State": {"Running": True}}
            with patch.object(cluster.host, "run") as run:
                self.assertEqual(cluster.rpc("stop", 0, identity), {"stopped": True})
            kill.assert_called_once_with(self.PID, cluster.signal.SIGTERM)
            self.assertEqual(sleep.call_count, 1)
            run.assert_called_once_with("docker", "stop", identity["name"])
            self.assertTrue((Path(identity["record"]) / "cancel.json").exists())

    def test_a_supervisor_that_never_exits_times_out_before_docker_is_read(self):
        with self.supervised() as (identity, _, kill, sleep, inspect):
            with self.assertRaisesRegex(TimeoutError, "did not terminate"):
                cluster.rpc("stop", 0, identity)
            kill.assert_called_once()
            self.assertEqual(sleep.call_count, 60)
            inspect.assert_not_called()

    def test_a_supervisor_already_gone_is_not_signalled_and_docker_is_read(self):
        with self.supervised() as (identity, proc, kill, sleep, inspect):
            proc.unlink()
            self.assertEqual(cluster.rpc("stop", 0, identity), {"stopped": True})
            kill.assert_not_called()
            sleep.assert_not_called()
            inspect.assert_called_once_with(identity)
            self.assertTrue((Path(identity["record"]) / "cancel.json").exists())

    def test_a_stop_without_a_record_stops_only_the_named_container(self):
        # A recovery stops a launch it did not reserve: it has a name, no record.
        identity = {"name": "glm53-old", "fingerprint": "f"}
        for running, stops in ((True, 1), (False, 0)):
            with (
                self.subTest(running=running),
                patch.object(
                    cluster,
                    "inspect_attempt",
                    return_value={"State": {"Running": running}},
                ),
                patch.object(cluster.host, "run") as run,
                patch.object(cluster.os, "kill") as kill,
            ):
                self.assertEqual(cluster.rpc("stop", 0, identity), {"stopped": True})
                self.assertEqual(run.call_count, stops)
                kill.assert_not_called()


class AttemptInspectionTests(unittest.TestCase):
    """An attempt is inspected only once Docker's inventory lists its name."""

    def test_a_listed_container_is_inspected_and_an_unlisted_one_is_absent(self):
        identity = {"name": "glm53-a", "fingerprint": "f"}
        for names, inspected in (("other\nglm53-a\n", True), ("other\n", False)):
            with (
                self.subTest(names=names),
                patch.object(cluster.host, "run", return_value=names) as run,
                patch.object(
                    server, "inspect_owned", return_value={"State": {}}
                ) as inspect,
            ):
                result = cluster.inspect_attempt(identity)
                run.assert_called_once_with(
                    "docker", "ps", "-a", "--format", "{{.Names}}"
                )
                if inspected:
                    self.assertEqual(result, {"State": {}})
                    inspect.assert_called_once_with("glm53-a", "f")
                else:
                    self.assertIsNone(result)
                    inspect.assert_not_called()


class BackendOperationTests(unittest.TestCase):
    """Each backend operation sends its own RPC action with the value it names."""

    def test_each_operation_names_its_action_and_value(self):
        backend = cluster.SSHBackend(["head", "peer"], "/srv/model", None, 30)
        launch, identity = {"manifest": {}}, {"name": "n"}
        rows = [
            {"rank": 1, "identity": {"name": "w"}},
            {"rank": 0, "identity": identity},
        ]
        for call, expected in (
            (lambda: backend.reserve(1, launch), ("reserve", 1, launch)),
            (
                lambda: backend.reserve(1, launch, recovery=True),
                ("reserve", 1, {**launch, "recovery": True}),
            ),
            (lambda: backend.start(1, identity), ("start", 1, identity)),
            (lambda: backend.stop(1, identity), ("stop", 1, identity)),
            (
                lambda: backend.install(1, identity, "text"),
                ("install", 1, {"identity": identity, "text": "text"}),
            ),
            # The warmup runs on the head, whatever order the rows come in.
            (lambda: backend.warmup(rows), ("warmup", 0, identity)),
        ):
            with self.subTest(expected[0]), patch.object(backend, "call") as sent:
                call()
                sent.assert_called_once_with(*expected)


class AttemptPollTests(unittest.TestCase):
    """What a poll answers before it reaches the head's log and API."""

    def poll(self, rank, *, finished=None, info=None, state=None):
        with tempfile.TemporaryDirectory() as tmp, rooted(Path(tmp)):
            identity = cluster.rpc(
                "reserve",
                rank,
                {"manifest": {"fingerprint": "f"}, "config_path": "/srv/s.toml"},
            )
            if finished is not None:
                cluster.write_json(Path(identity["record"]) / "finished.json", finished)
            if state is not None:
                server.state_path(rank).parent.mkdir()
                cluster.write_json(
                    server.state_path(rank),
                    {"name": identity["name"] if state == "owned" else state},
                )
            with patch.object(cluster, "inspect_attempt", return_value=info) as inspect:
                return cluster.rpc("poll", rank, identity), inspect

    def test_a_finished_supervisor_is_a_failed_attempt_without_reading_docker(self):
        result, inspect = self.poll(1, finished={"status": "failed", "error": "E"})
        self.assertEqual(
            result, {"failed": True, "finished": {"status": "failed", "error": "E"}}
        )
        inspect.assert_not_called()

    def test_a_container_not_yet_created_is_not_ready_and_a_stopped_one_failed(self):
        self.assertEqual(self.poll(1)[0], {"ready": False})
        stopped = {"State": {"Running": False}}
        self.assertEqual(self.poll(1, info=stopped)[0], {"failed": True})

    def test_a_worker_is_ready_once_its_state_names_this_attempt(self):
        running = {"State": {"Running": True}}
        for state, ready in ((None, False), ("someone-else", False), ("owned", True)):
            with self.subTest(state=state):
                result, _ = self.poll(1, info=running, state=state)
                self.assertEqual(result, {"ready": ready})


class HeadHealthTests(unittest.TestCase):
    """A started head is ready only on /health 200; a refused credential is raised."""

    def health(self, error):
        text = EXAMPLE.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp, rooted(Path(tmp)):
            launch = {
                "manifest": server_config.freeze(server_config.loads(text), {}),
                "config_path": "/srv/glm53/state/server.toml",
            }
            identity = cluster.rpc("reserve", 0, launch)
            (Path(tmp) / "state").mkdir()
            cluster.write_json(server.state_path(0), {"name": identity["name"]})
            with (
                patch.object(
                    cluster,
                    "inspect_attempt",
                    return_value={"State": {"Running": True}},
                ),
                patch.object(
                    cluster.subprocess,
                    "check_output",
                    return_value=b"Application startup complete.\n",
                ),
                patch.object(cluster.model_http, "open_response", side_effect=error),
            ):
                return cluster.rpc("poll", 0, identity)

    def test_an_authentication_failure_is_raised_not_read_as_not_ready(self):
        for code in (401, 403):
            with (
                self.subTest(code=code),
                self.assertRaises(cluster.model_http.ModelHTTPError) as caught,
            ):
                self.health(cluster.model_http.ModelHTTPError(code))
            self.assertEqual(caught.exception.code, code)

    def test_any_other_api_failure_is_not_ready(self):
        for code in (500, None):
            with self.subTest(code=code):
                error = cluster.model_http.ModelHTTPError(code)
                self.assertEqual(self.health(error), {"ready": False})


class ReadinessWaitTests(unittest.TestCase):
    """SSHBackend.ready polls every rank until all are ready, one fails, or time runs out."""

    @contextlib.contextmanager
    def clock(self):
        # The fake sleep advances the fake clock, so the deadline is reached in
        # a fixed number of rounds whatever else reads the time.
        now = [0.0]
        with (
            patch.object(cluster.time, "monotonic", lambda: now[0]),
            patch.object(
                cluster.time,
                "sleep",
                side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds),
            ) as sleep,
        ):
            yield sleep

    def setUp(self):
        self.backend = cluster.SSHBackend(["head", "peer"], "/srv/model", None, 30)
        self.rows = [{"rank": 1, "identity": "id-1"}, {"rank": 0, "identity": "id-0"}]

    def test_a_terminated_rank_is_named_by_its_rank_not_its_row(self):
        with (
            self.clock(),
            patch.object(
                self.backend,
                "call",
                side_effect=[{"ready": True}, {"failed": True}],
            ),
            self.assertRaises(switch.OperationFailure) as caught,
        ):
            self.backend.ready(self.rows)
        self.assertEqual(caught.exception.evidence["rank"], 0)
        self.assertEqual(caught.exception.evidence["reason"], switch.RANK_TERMINATED)

    def test_the_pair_is_ready_once_every_rank_is(self):
        statuses = [{"ready": False}, {"ready": True}, {"ready": True}, {"ready": True}]
        with (
            self.clock() as sleep,
            patch.object(self.backend, "call", side_effect=statuses) as call,
        ):
            self.assertIsNone(self.backend.ready(self.rows))
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(
            [c.args for c in call.call_args_list[:2]],
            [("poll", 1, "id-1"), ("poll", 0, "id-0")],
        )

    def test_a_pair_never_ready_runs_out_at_the_readiness_deadline(self):
        with (
            self.clock() as sleep,
            patch.object(self.backend, "call", return_value={"ready": False}),
            self.assertRaises(switch.OperationFailure) as caught,
        ):
            self.backend.ready(self.rows)
        self.assertEqual(
            caught.exception.evidence,
            {
                "action": "ready",
                "rank": None,
                "reason": switch.READINESS_DEADLINE,
                "exit_code": None,
            },
        )
        self.assertEqual(sleep.call_count, 6)  # 30 s timeout, 5 s between rounds


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
                    result = switch.resume(pair, reports[-1])
                    self.assertEqual(result["status"], resumed)

    def test_only_a_lost_poll_is_a_lost_observation(self):
        failed = self.lost(return_value=subprocess.CompletedProcess([], 1, "", ""))
        self.assertEqual(failed.evidence["reason"], switch.REMOTE_OPERATION_FAILED)
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
                self.assertRaisesRegex(ValueError, "unconfirmed readiness observation"),
            ):
                switch.resume(backend, {"status": status, "new": rows})
            backend.current.assert_not_called()
            backend.ready.assert_not_called()


RING = Path(__file__).resolve().parents[1] / "examples/server.tp3.example.toml"


class RingClusterTests(unittest.TestCase):
    """The coordinator addresses one host per node of the launch."""

    def switch_with(self, config, hosts):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                cluster, "switch", return_value={"status": "complete"}
            ) as switched,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            cluster.main(
                [
                    "switch",
                    "--config",
                    str(config),
                    "--remote-config",
                    "/srv/glm53/state/server.toml",
                    "--hosts",
                    *hosts,
                    "--checkout",
                    "/srv/glm53/source",
                    "--output",
                    str(Path(tmp) / "out"),
                ]
            )
            return switched.call_args

    def test_a_ring_switch_addresses_three_hosts(self):
        call = self.switch_with(RING, ["head", "peer", "third"])
        self.assertEqual(call.kwargs["nodes"], 3)
        self.assertEqual(call.args[0].hosts, ["head", "peer", "third"])
        self.assertEqual(self.switch_with(EXAMPLE, ["head", "peer"]).kwargs["nodes"], 2)

    def test_the_host_count_must_be_the_node_count(self):
        for config, hosts in (
            (RING, ["head", "peer"]),
            (EXAMPLE, ["head", "peer", "third"]),
        ):
            with self.subTest(config=config.name), self.assertRaises(SystemExit):
                self.switch_with(config, hosts)

    def test_readiness_needs_every_rank(self):
        backend = cluster.SSHBackend(["head", "peer", "third"], "/srv/model", None, 30)
        rows = [{"rank": r, "identity": {}} for r in (1, 0)]
        with self.assertRaisesRegex(ValueError, "every rank"):
            backend.ready(rows)

    def test_a_running_launch_of_another_size_is_refused_before_anything_stops(self):
        backend = cluster.SSHBackend(["head", "peer", "third"], "/srv/model", None, 30)
        running = {
            "name": "old-0",
            "fingerprint": "f",
            "launch": {"manifest": {"profile": server_config.load(EXAMPLE)}},
        }
        with (
            patch.object(backend, "call", return_value=running),
            self.assertRaisesRegex(ValueError, "2 nodes"),
        ):
            backend.current(0)
        with patch.object(backend, "call", return_value=None):
            self.assertIsNone(backend.current(2))

    def test_a_rank_rpc_accepts_the_third_rank(self):
        with tempfile.TemporaryDirectory() as tmp, rooted(Path(tmp)):
            self.assertIsNone(cluster.rpc("current", 2, None))
        for bad in (-1, "2", None):
            with self.subTest(rank=bad), self.assertRaises(ValueError):
                cluster.rpc("current", bad, None)
