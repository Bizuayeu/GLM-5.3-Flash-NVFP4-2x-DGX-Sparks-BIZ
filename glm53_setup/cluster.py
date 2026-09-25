"""Preflight both ranks, then perform an owned, recoverable two-rank switch."""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import host, launch_assets, model_http, server, server_config
from .config import ROOT
from .io import write_json
from .switch import (
    READINESS_UNCONFIRMED,
    RECOVERY_READINESS_UNCONFIRMED,
    SSH_UNAVAILABLE,
    TRANSPORT_TIMEOUT,
    OperationFailure,
    finish,
    switch,
)


def current(rank):
    path = ROOT / f"state/startup-rank{rank}.json"
    if not path.exists():
        return None
    state = server.read_json(path)
    info = server.inspect_owned(state["name"], state["fingerprint"])
    if not info["State"]["Running"]:
        return None
    if "config_path" not in state:
        raise ValueError("Running rank has no recorded configuration path for recovery")
    profile = server.read_json(Path(state["record"]) / "settings.json")
    manifest = {"profile": profile, "fingerprint": state["fingerprint"]}
    server.thaw(manifest)
    return {
        "name": state["name"],
        "fingerprint": state["fingerprint"],
        "launch": {"manifest": manifest, "config_path": state["config_path"]},
    }


def inspect_attempt(identity):
    # Missing is meaningful only after a successful Docker inventory query.
    names = host.run("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()
    return (
        server.inspect_owned(identity["name"], identity["fingerprint"])
        if identity["name"] in names
        else None
    )


def owned_record(identity):
    record = Path(identity["record"]).resolve()
    if not record.is_relative_to((ROOT / "records").resolve()):
        raise ValueError("Attempt record must belong to this checkout")
    if server.read_json(record / "identity.json") != identity:
        raise ValueError("Attempt identity changed")
    return record


def install(rank, value):
    """Write the profile text this rank was switched to at its recorded path."""
    identity, text = value["identity"], value["text"]
    owned_record(identity)
    state = ROOT / f"state/startup-rank{rank}.json"
    if not state.exists() or server.read_json(state)["name"] != identity["name"]:
        raise ValueError("This attempt is not the running rank")
    running = server.thaw(identity["launch"]["manifest"])
    profile = server_config.loads(text)
    # The launch may have frozen an allocator override from its environment.
    if "cuda_allocator_conf" in running["runtime"]:
        profile["runtime"].setdefault(
            "cuda_allocator_conf", running["runtime"]["cuda_allocator_conf"]
        )
    if profile != running:
        raise ValueError("Profile text is not the running profile")
    path = Path(identity["launch"]["config_path"])
    if not path.is_absolute() or path.suffix != ".toml":
        raise ValueError("Configuration path must be an absolute .toml path")
    result = {"written": True}
    if path.exists():
        old = path.read_bytes()
        if old == text.encode():
            return {"unchanged": True}
        try:
            mark = server_config.fingerprint(server_config.loads(old.decode()))[:8]
        except ValueError:  # TOMLDecodeError and UnicodeDecodeError included
            mark = "unparsed"
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = path.with_name(f"{path.name}.bak-{stamp}-{mark}")
        with backup.open("xb") as stream:
            stream.write(old)
        result["backup"] = str(backup)
    temporary = path.with_name(path.name + ".installing")
    temporary.write_bytes(text.encode())
    os.replace(temporary, path)
    return result


def rpc(action, rank, value):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("Invalid rank")
    if action == "current":
        return current(rank)
    if action == "prepare":
        return launch_assets.inspect(
            server.thaw(value["manifest"]),
            Path(value["config_path"]),
            rank,
            recovery=value.get("recovery", False),
        )
    if action == "reserve":
        run_id = uuid.uuid4().hex
        record = ROOT / "records" / f"switch-{run_id}-r{rank}"
        record.mkdir(parents=True)
        identity = {
            "rank": rank,
            "run_id": run_id,
            "name": f"glm53-startup-r{rank}-{run_id}",
            "fingerprint": value["manifest"]["fingerprint"],
            "record": str(record),
            "launch": value,
        }
        write_json(record / "identity.json", identity)
        write_json(record / "launch.json", value["manifest"])
        return identity
    if action == "start":
        record = owned_record(value)
        if (record / "finished.json").exists() or (record / "cancel.json").exists():
            raise ValueError(
                "This attempt was already started; reserve a fresh identity"
            )
        if (record / "supervisor.log").exists():
            # cc-defer: supervisor.log exists before Popen; a replay in that window
            # reports a launch that did not happen and the poll's deadline recovers
            # it. Write job.json from this side once Popen returns if that is seen.
            # A replay: the first start reached this rank and its reply was lost
            # on the way back. The supervisor it launched owns the attempt.
            return {"started": True, "replayed": True}
        with (record / "supervisor.log").open("x") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "glm53_setup.cluster",
                    "job",
                    "--record",
                    str(record),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return {"started": True}
    if action == "stop":
        if "record" in value:
            record = owned_record(value)
            write_json(record / "cancel.json", {"cancelled": True})
            job = record / "job.json"
            if job.exists():
                pid = server.read_json(job)["pid"]
                cmdline = Path(f"/proc/{pid}/cmdline")
                if cmdline.exists():
                    args = cmdline.read_bytes().split(b"\0")
                    if (
                        b"glm53_setup.cluster" not in args
                        or str(record).encode() not in args
                    ):
                        raise ValueError(
                            "Supervisor PID no longer belongs to this attempt"
                        )
                    os.kill(pid, signal.SIGTERM)
                    # job and server share a process; SIGTERM runs supervisor
                    # cleanup. Confirm it exited before checking Docker to avoid
                    # a delayed launch racing a recovery attempt.
                    for _ in range(60):
                        if not cmdline.exists() or not cmdline.read_bytes():
                            break
                        time.sleep(1)
                    else:
                        raise TimeoutError("Attempt supervisor did not terminate")
        info = inspect_attempt(value)
        if info is not None and info["State"]["Running"]:
            host.run("docker", "stop", value["name"])
        return {"stopped": True}
    if action == "poll":
        record = owned_record(value)
        finished = record / "finished.json"
        if finished.exists():
            return {"failed": True, "finished": server.read_json(finished)}
        info = inspect_attempt(value)
        if info is None:
            return {"ready": False}
        if not info["State"]["Running"]:
            return {"failed": True}
        state = ROOT / f"state/startup-rank{rank}.json"
        if not state.exists() or server.read_json(state)["name"] != value["name"]:
            return {"ready": False}
        if rank == 0:
            profile = server.thaw(value["launch"]["manifest"])
            # The tail answers a starting head cheaply. A head that has served
            # for minutes has pushed the line out of it (the supervisor reads
            # /metrics every two seconds), so a resume reads the whole log.
            started = False
            for window in (["--tail", "200"], []):
                logs = subprocess.check_output(
                    ["docker", "logs", *window, value["name"]],
                    stderr=subprocess.STDOUT,
                )
                if b"Application startup complete" in logs:
                    started = True
                    break
            if not started:
                return {"ready": False}
            try:
                with model_http.open_response(
                    f"http://127.0.0.1:{profile['api']['port']}", "/health", timeout=2
                ) as response:
                    return {"ready": response.status == 200}
            except model_http.ModelHTTPError as error:
                if error.code in (401, 403):
                    raise
                return {"ready": False}
        return {"ready": True}
    if action == "install":
        return install(rank, value)
    if action == "warmup":
        profile = server.thaw(value["launch"]["manifest"])
        if rank != 0 or not profile["generation"].get("warmup", False):
            return {"skipped": True}
        owned_record(value)
        state = ROOT / "state/startup-rank0.json"
        if server.read_json(state)["name"] != value["name"]:
            raise ValueError("Running rank 0 is not this attempt")
        return server.warmup_running(profile)
    raise ValueError("Unknown cluster operation")


def as_recovery(launch):
    """The same launch, marked as the pair a switch must be able to restore.

    The mark rides beside the manifest, never inside it: the fingerprint and the
    frozen launch stay what the pair was started with.
    """
    return {**launch, "recovery": True}


class SSHBackend:
    def __init__(self, hosts, checkout, ssh_config, timeout):
        if any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host) for host in hosts
        ):
            raise ValueError("Use explicit SSH hosts or aliases")
        self.hosts, self.checkout, self.ssh_config, self.timeout = (
            hosts,
            checkout,
            ssh_config,
            timeout,
        )

    def call(self, action, rank, value=None):
        command = (
            "cd "
            + shlex.quote(self.checkout)
            + " && python3 -m glm53_setup.cluster rpc"
        )
        args = ["ssh"] + (["-F", str(self.ssh_config)] if self.ssh_config else [])
        args += [
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            self.hosts[rank],
            command,
        ]
        # Reads, and the two mutations a rank makes harmless to replay: stop is
        # idempotent, start answers for an attempt already in flight. A switch has
        # stopped both ranks by the time it starts the new ones, so one dropped
        # connection there used to cost a recovery.
        attempts = 3 if action in ("current", "prepare", "poll", "start", "stop") else 1
        for attempt in range(attempts):
            try:
                response = subprocess.run(
                    args,
                    check=False,
                    input=json.dumps({"action": action, "rank": rank, "value": value}),
                    text=True,
                    capture_output=True,
                    # The ladder's long rung pays a full prefill (times by
                    # length in docs/benchmarks.md); the rest answer in seconds.
                    timeout=self.timeout if action == "warmup" else 120,
                )
            except subprocess.TimeoutExpired:
                if attempt + 1 == attempts:
                    raise OperationFailure(action, rank, TRANSPORT_TIMEOUT) from None
            else:
                if response.returncode == 0:
                    return json.loads(response.stdout)
                if response.returncode != 255 or attempt + 1 == attempts:
                    raise OperationFailure(
                        action,
                        rank,
                        SSH_UNAVAILABLE
                        if response.returncode == 255
                        else "remote-operation-failed",
                        response.returncode,
                    )
            time.sleep(1)

    def current(self, rank):
        return self.call("current", rank)

    def prepare(self, rank, launch, *, recovery=False):
        return self.call("prepare", rank, as_recovery(launch) if recovery else launch)

    def reserve(self, rank, launch, *, recovery=False):
        return self.call("reserve", rank, as_recovery(launch) if recovery else launch)

    def start(self, rank, identity):
        return self.call("start", rank, identity)

    def stop(self, rank, identity):
        return self.call("stop", rank, identity)

    def install(self, rank, identity, text):
        return self.call("install", rank, {"identity": identity, "text": text})

    def warmup(self, rows):
        head = next(r for r in rows if r["rank"] == 0)
        return self.call("warmup", 0, head["identity"])

    def ready(self, rows):
        if {r["rank"] for r in rows} != {0, 1}:
            raise ValueError("Recovery did not restore both ranks")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            statuses = [self.call("poll", r["rank"], r["identity"]) for r in rows]
            if any(s.get("failed") for s in statuses):
                index = next(
                    i for i, status in enumerate(statuses) if status.get("failed")
                )
                raise OperationFailure("ready", rows[index]["rank"], "rank-terminated")
            if all(s.get("ready") for s in statuses):
                return
            time.sleep(5)
        raise OperationFailure("ready", None, "readiness-deadline")


def resume(backend, report, *, config=None, save=lambda report: None):
    recovering = report["status"] == RECOVERY_READINESS_UNCONFIRMED
    rows = report["recovery"] if recovering else report["new"]
    if report["status"] not in (
        READINESS_UNCONFIRMED,
        RECOVERY_READINESS_UNCONFIRMED,
    ) or {r["rank"] for r in rows} != {0, 1}:
        raise ValueError(
            "Only a recorded, unconfirmed two-rank readiness observation can resume"
        )
    assets = report["recovery_assets"] if recovering else report["assets"]
    for row in rows:
        current_rank = backend.current(row["rank"])
        identity = row["identity"]
        if current_rank is None or any(
            current_rank[key] != identity[key] for key in ("name", "fingerprint")
        ):
            raise ValueError("Running identity changed; readiness cannot resume")
        if backend.prepare(row["rank"], identity["launch"]) != assets[row["rank"]]:
            raise ValueError("Assets changed since the recorded launch")
    backend.ready(rows)
    if recovering:
        report["prior_recovery_observation_failure"] = report.pop(
            "recovery_observation_failure"
        )
        report["recovered"] = True
        report["status"] = (
            "failed"  # The candidate still failed; the old profile recovered.
        )
    else:
        report["prior_observation_failure"] = report.pop("failure")
        report.pop("error")
        report["status"] = "complete"
        save(report)
        # The switch that lost its observation never reached its last steps.
        finish(backend, report, save=save, config=config)
    return report


def ssh_backend(args):
    """The transport both transported actions address the two ranks through."""
    return SSHBackend(args.hosts, args.checkout, args.ssh_config, args.ready_timeout)


def act_resume(cli, args):
    """Carry a recorded switch forward without replaying its start."""
    if not all((args.output, args.hosts, args.checkout)) or args.ready_timeout < 1:
        cli.error(
            "resume requires --output, --hosts, --checkout and a positive timeout"
        )
    result = resume(
        ssh_backend(args),
        server.read_json(args.output / "result.json"),
        config=args.config.read_text(encoding="utf-8") if args.config else None,
        save=lambda report: write_json(args.output / "result.json", report),
    )
    write_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "recovered": result.get("recovered", False),
                "output": str(args.output),
            }
        )
    )


def act_rpc(cli, args):
    """Serve one coordinator operation for the rank this runs on."""
    payload = json.load(sys.stdin)
    print(json.dumps(rpc(payload["action"], payload["rank"], payload["value"])))


def act_job(cli, args):
    """Run a reserved attempt's supervisor; the coordinator owns the identity."""
    identity = server.read_json(args.record / "identity.json")
    owned_record(identity)
    write_json(args.record / "job.json", {"pid": os.getpid()})
    if (args.record / "cancel.json").exists():
        write_json(args.record / "finished.json", {"status": "cancelled before launch"})
        return
    try:
        server.main(
            [
                "start",
                "--config",
                identity["launch"]["config_path"],
                "--launch",
                str(args.record / "launch.json"),
                "--rank",
                str(identity["rank"]),
                "--run-id",
                identity["run_id"],
                *(["--recovery"] if identity["launch"].get("recovery") else []),
            ]
        )
        write_json(args.record / "finished.json", {"status": "stopped"})
    except BaseException as error:
        write_json(
            args.record / "finished.json",
            {"status": "failed", "error": type(error).__name__},
        )
        raise


def act_switch(cli, args):
    """Stop the running pair and start the candidate, recoverably."""
    if (
        not all(
            (
                args.config,
                args.remote_config,
                args.hosts,
                args.checkout,
                args.output,
            )
        )
        or args.ready_timeout < 1
    ):
        cli.error(
            "switch requires --config, --remote-config, --hosts, --checkout, --output and positive readiness timeout"
        )
    if not args.remote_config.startswith("/") or not args.checkout.startswith("/"):
        cli.error("Remote paths must be absolute Linux paths")
    args.output.mkdir(parents=True, exist_ok=False)
    # Read once: the manifest and the text the ranks write come from the same
    # bytes. Text mode turns CRLF into LF on the way.
    text = args.config.read_text(encoding="utf-8")
    launch = {
        "manifest": server.freeze(server_config.loads(text)),
        "config_path": args.remote_config,
    }
    write_json(args.output / "launch.json", launch)
    try:
        result = switch(
            ssh_backend(args),
            launch,
            save=lambda r: write_json(args.output / "result.json", r),
            config=None if args.no_send_config else text,
        )
    except Exception as error:
        write_json(
            args.output / "failure.json",
            {"error": type(error).__name__, "message": str(error)},
        )
        raise
    sent = result.get("config")
    print(
        json.dumps(
            {
                "status": result["status"],
                "config": "not-sent"
                if sent is None
                else "failed"
                if isinstance(sent, dict)
                else "installed",
                "output": str(args.output),
            }
        )
    )


# Insertion order is the order argparse prints in --help.
ACTIONS = {
    "switch": act_switch,
    "resume": act_resume,
    "rpc": act_rpc,
    "job": act_job,
}


def parser():
    """The coordinator's argument interface; each action validates its own set."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=list(ACTIONS))
    cli.add_argument("--config", type=Path)
    cli.add_argument(
        "--remote-config", help="Same absolute configuration path on both Linux ranks"
    )
    cli.add_argument(
        "--no-send-config",
        action="store_true",
        help="Leave the --remote-config files as they are (default: both ranks "
        "write the --config text there once the new pair is complete)",
    )
    cli.add_argument("--hosts", nargs=2)
    cli.add_argument(
        "--checkout", help="Same audited absolute Linux checkout on both ranks"
    )
    cli.add_argument("--ssh-config", type=Path)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--record", type=Path)
    cli.add_argument("--ready-timeout", type=int, default=1800)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    return ACTIONS[args.action](cli, args)


if __name__ == "__main__":
    main()
