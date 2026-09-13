"""Preflight both ranks, then perform an owned, recoverable experimental switch."""

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

from . import launch_assets, model_http, service, startup, startup_config
from .config import ROOT
from .io import write_json
from .switch import switch


def current(rank):
    path = ROOT / f"state/startup-rank{rank}.json"
    if not path.exists():
        return None
    state = startup.read_json(path)
    info = startup.inspect_owned(state["name"], state["fingerprint"])
    if not info["State"]["Running"]:
        return None
    if "config_path" not in state:
        raise ValueError("Running rank has no recorded configuration path for recovery")
    profile = startup.read_json(Path(state["record"]) / "settings.json")
    manifest = {"profile": profile, "fingerprint": state["fingerprint"]}
    startup.thaw(manifest)
    return {
        "name": state["name"],
        "fingerprint": state["fingerprint"],
        "launch": {"manifest": manifest, "config_path": state["config_path"]},
    }


def inspect_attempt(identity):
    # Missing is meaningful only after a successful Docker inventory query.
    names = service.run("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()
    return (
        startup.inspect_owned(identity["name"], identity["fingerprint"])
        if identity["name"] in names
        else None
    )


def owned_record(identity):
    record = Path(identity["record"]).resolve()
    if not record.is_relative_to((ROOT / "records").resolve()):
        raise ValueError("Attempt record must belong to this checkout")
    if startup.read_json(record / "identity.json") != identity:
        raise ValueError("Attempt identity changed")
    return record


def rpc(action, rank, value):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("Invalid rank")
    if action == "current":
        return current(rank)
    if action == "prepare":
        return launch_assets.inspect(
            startup.thaw(value["manifest"]), Path(value["config_path"]), rank
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
        if (record / "job.json").exists() or (record / "cancel.json").exists():
            raise ValueError(
                "This attempt was already started; reserve a fresh identity"
            )
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
                pid = startup.read_json(job)["pid"]
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
                    # job and startup share a process; SIGTERM runs supervisor
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
            service.run("docker", "stop", value["name"])
        return {"stopped": True}
    if action == "poll":
        record = owned_record(value)
        finished = record / "finished.json"
        if finished.exists():
            return {"failed": True, "finished": startup.read_json(finished)}
        info = inspect_attempt(value)
        if info is None:
            return {"ready": False}
        if not info["State"]["Running"]:
            return {"failed": True}
        state = ROOT / f"state/startup-rank{rank}.json"
        if not state.exists() or startup.read_json(state)["name"] != value["name"]:
            return {"ready": False}
        if rank == 0:
            profile = startup.thaw(value["launch"]["manifest"])
            logs = subprocess.check_output(
                ["docker", "logs", "--tail", "200", value["name"]],
                stderr=subprocess.STDOUT,
            )
            if b"Application startup complete" not in logs:
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
    raise ValueError("Unknown cluster operation")


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
        response = subprocess.run(
            args,
            check=False,
            input=json.dumps({"action": action, "rank": rank, "value": value}),
            text=True,
            capture_output=True,
            timeout=120,
        )
        if response.returncode:
            raise RuntimeError(
                f"Rank {rank} {action} failed (SSH/process exit {response.returncode})"
            )
        return json.loads(response.stdout)

    def current(self, rank):
        return self.call("current", rank)

    def prepare(self, rank, launch):
        return self.call("prepare", rank, launch)

    def reserve(self, rank, launch):
        return self.call("reserve", rank, launch)

    def start(self, rank, identity):
        return self.call("start", rank, identity)

    def stop(self, rank, identity):
        return self.call("stop", rank, identity)

    def ready(self, rows):
        if {r["rank"] for r in rows} != {0, 1}:
            raise ValueError("Recovery did not restore both ranks")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            statuses = [self.call("poll", r["rank"], r["identity"]) for r in rows]
            if any(s.get("failed") for s in statuses):
                raise RuntimeError("New rank terminated before readiness")
            if all(s.get("ready") for s in statuses):
                return
            time.sleep(5)
        raise TimeoutError("New model readiness deadline exceeded")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["switch", "rpc", "job"])
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--remote-config", help="Same absolute configuration path on both Linux ranks"
    )
    parser.add_argument("--hosts", nargs=2)
    parser.add_argument(
        "--checkout", help="Same audited absolute Linux checkout on both ranks"
    )
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument("--experimental", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "rpc":
        payload = json.load(sys.stdin)
        print(json.dumps(rpc(payload["action"], payload["rank"], payload["value"])))
        return
    if args.action == "job":
        identity = startup.read_json(args.record / "identity.json")
        owned_record(identity)
        write_json(args.record / "job.json", {"pid": os.getpid()})
        if (args.record / "cancel.json").exists():
            write_json(
                args.record / "finished.json", {"status": "cancelled before launch"}
            )
            return
        try:
            startup.main(
                [
                    "start",
                    "--experimental",
                    "--config",
                    identity["launch"]["config_path"],
                    "--launch",
                    str(args.record / "launch.json"),
                    "--rank",
                    str(identity["rank"]),
                    "--run-id",
                    identity["run_id"],
                ]
            )
            write_json(args.record / "finished.json", {"status": "stopped"})
        except BaseException as error:
            write_json(
                args.record / "finished.json",
                {"status": "failed", "error": type(error).__name__},
            )
            raise
        return
    if (
        not all(
            (
                args.config,
                args.remote_config,
                args.hosts,
                args.checkout,
                args.output,
                args.experimental,
            )
        )
        or args.ready_timeout < 1
    ):
        parser.error(
            "switch requires --config, --remote-config, --hosts, --checkout, --output, --experimental and positive readiness timeout"
        )
    if not args.remote_config.startswith("/") or not args.checkout.startswith("/"):
        parser.error("Remote paths must be absolute Linux paths")
    args.output.mkdir(parents=True, exist_ok=False)
    launch = {
        "manifest": startup.freeze(startup_config.load(args.config)),
        "config_path": args.remote_config,
    }
    write_json(args.output / "launch.json", launch)
    backend = SSHBackend(args.hosts, args.checkout, args.ssh_config, args.ready_timeout)
    try:
        result = switch(
            backend, launch, save=lambda r: write_json(args.output / "result.json", r)
        )
    except Exception as error:
        write_json(
            args.output / "failure.json",
            {"error": type(error).__name__, "message": str(error)},
        )
        raise
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
