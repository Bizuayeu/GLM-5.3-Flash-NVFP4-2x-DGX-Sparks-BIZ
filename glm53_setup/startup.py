"""Run a serial TP=2 reference experiment with a configurable lifetime."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import service
from . import startup_config as settings
from .config import MODEL_LAYERS, ROOT, load_lock
from .io import write_json

LABEL = "glm53.experiment.startup"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def projector_path(profile, config_path):
    return (config_path.parent / profile["lpa"]["projector"]).resolve()


def model_path(profile, cache):
    lock = load_lock()
    if profile["mtp"]["enabled"]:
        return cache / profile["mtp"]["view"] / lock["revision"]
    return (
        cache
        / "hub"
        / ("models--" + lock["model"].replace("/", "--"))
        / "snapshots"
        / lock["revision"]
    )


def command(profile, config_path, rank, name, cache=None):
    settings.validate(profile)
    cache = cache or Path.home() / ".cache/huggingface"
    model = model_path(profile, cache)
    limit = f"{profile['resources']['container_memory_gib']}g"
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--init",
        "--restart",
        "no",
        "--label",
        f"{LABEL}={settings.fingerprint(profile)}",
        "--label",
        f"glm53.setup.rank={rank}",
        "--gpus",
        "all",
        "--network",
        "host",
        "--memory",
        limit,
        "--memory-swap",
        limit,
        "--shm-size",
        "2g",
        "--cap-add",
        "IPC_LOCK",
        "--ulimit",
        "memlock=-1:-1",
        "--device",
        "/dev/infiniband:/dev/infiniband",
        "-v",
        f"{cache.resolve()}:/hf:ro",
        "-v",
        f"{ROOT / 'state/tp2-runtime-cache'}:/root/.cache",
    ]
    if profile["lpa"]["enabled"]:
        target = "/lpa/projector.pt"
        args += ["-v", f"{projector_path(profile, config_path)}:{target}:ro"]
    if profile["profiling"]["enabled"]:
        args += ["-v", f"{ROOT / 'records/profiles' / name}:/profiles"]
    for key, value in settings.environment(profile, rank).items():
        args += ["-e", f"{key}={value}"]
    return args + [
        "--entrypoint",
        "vllm",
        settings.selected_image(profile),
        *settings.serve_args(
            profile, rank, "/hf/" + model.relative_to(cache).as_posix()
        ),
    ]


def available_gib():
    rows = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return int(rows["MemAvailable"].split()[0]) / 1024**2


def inspect_owned(name, fingerprint=None):
    info = json.loads(service.run("docker", "inspect", name))[0]
    owner = (info["Config"].get("Labels") or {}).get(LABEL)
    if not owner or (fingerprint and owner != fingerprint):
        raise ValueError("Container does not belong to this startup profile")
    return info


def preflight(profile, config_path, rank):
    cache = Path.home() / ".cache/huggingface"
    lock = load_lock()
    source = service.snapshot_from_state(
        read_json(ROOT / "state/download-status.json"), lock
    )
    expected = model_path(
        {**profile, "mtp": {**profile["mtp"], "enabled": False}}, cache
    )
    if source.resolve() != expected.resolve():
        raise ValueError("Download state must identify the pinned HF cache snapshot")
    model = model_path(profile, cache)
    metadata = read_json(model / "config.json")
    checks = service.fabric_checks(settings.site(profile, rank))
    checks["full_model"] = metadata["text_config"][
        "num_hidden_layers"
    ] == MODEL_LAYERS and not metadata.get("_test_fixture_only")
    if profile["mtp"]["enabled"]:
        view = metadata.get("_local_mtp_metadata", {})
        checks["mtp_view"] = (
            view.get("source_revision") == lock["revision"]
            and view.get("weight_bytes_modified") is False
        )
    if profile["lpa"]["enabled"]:
        with projector_path(profile, config_path).open("rb") as stream:
            checks["projector_sha256"] = (
                hashlib.file_digest(stream, "sha256").hexdigest()
                == profile["lpa"]["projector_sha256"]
            )
    image = json.loads(
        service.run("docker", "image", "inspect", settings.selected_image(profile))
    )[0]
    checks["image_id"] = image["Id"] == settings.selected_image(profile)
    if profile["validation"]["component_worker"]:
        checks["component_worker"] = "GLM53_COMPONENT_API=1" in (
            image["Config"].get("Env") or []
        )
    if profile["cache"]["fused_unpack"]:
        checks["fused_unpack_support"] = "GLM53_FUSED_UNPACK_SUPPORTED=1" in (
            image["Config"].get("Env") or []
        )
    if not profile["runtime"]["enforce_eager"]:
        checks["decode_graph_support"] = "GLM53_DECODE_GRAPH_API=1" in (
            image["Config"].get("Env") or []
        )
    if profile["lpa"]["enabled"]:
        checks["lpa_worker"] = "GLM53_LPA_API=2" in (image["Config"].get("Env") or [])
    checks["reference_attention"] = (
        "GLM53_REFERENCE_ATTENTION=1" in image["Config"]["Env"]
    )
    checks["startup_memory"] = (
        available_gib() >= profile["resources"]["minimum_available_gib"]
    )
    # This is an explicit experiment, not the routine service qualification path.
    return {
        "scope": "experimental-reference",
        "checks": checks,
        "passed": all(checks.values()),
    }


def supervise(profile, name, record):
    seconds = profile["resources"]["run_seconds"]
    deadline = time.monotonic() + seconds if seconds else None
    try:
        with (record / "resources.jsonl").open("a", encoding="utf-8") as log:
            while True:
                info = inspect_owned(name)
                if not info["State"]["Running"]:
                    break
                available = available_gib()
                log.write(
                    json.dumps({"epoch": time.time(), "available_gib": available})
                    + "\n"
                )
                log.flush()
                if available < profile["resources"]["reserve_gib"] or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    write_json(
                        record / "stop-reason.json",
                        {
                            "reason": "memory-reserve"
                            if available < profile["resources"]["reserve_gib"]
                            else "run-deadline"
                        },
                    )
                    break
                time.sleep(2)  # Same sampling cadence as the measured experiments.
    finally:
        info = inspect_owned(name)
        if info["State"]["Running"]:
            service.run("docker", "stop", name)
        write_json(record / "container-inspect.json", inspect_owned(name))
        with (record / "server.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                ["docker", "logs", name],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )


def post(profile, path, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{profile['api']['port']}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=profile["generation"]["timeout_seconds"]
    ) as response:
        return json.load(response)


def ask(profile, request, sender=post):
    body = settings.request_body(profile, request)
    if not profile["lpa"]["enabled"]:
        return sender(profile, "/v1/chat/completions", body)
    # Only text/tool chat fields whose tokenization was exercised are accepted.
    allowed = {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "seed",
        "reasoning_effort",
        "chat_template_kwargs",
        "stream",
        "stop",
    }
    if body.keys() - allowed:
        raise ValueError(
            "Unsupported LPA request fields; tokenization must stay identical"
        )
    encoded = sender(
        profile,
        "/tokenize",
        {
            "model": body["model"],
            "messages": body["messages"],
            "tools": body.get("tools"),
            "add_generation_prompt": True,
            "chat_template_kwargs": body["chat_template_kwargs"],
        },
    )
    length = len(encoded["tokens"])
    if not length or length + body["max_tokens"] > profile["context"]["max_model_len"]:
        raise ValueError("Prompt plus max_tokens exceeds configured context")
    rpc = {
        "method": "lpa_configure",
        "kwargs": settings.lpa_request(profile, length),
        "timeout": profile["generation"]["timeout_seconds"],
    }
    try:
        sender(profile, "/collective_rpc", rpc)
        result = sender(profile, "/v1/chat/completions", body)
        if result["usage"]["prompt_tokens"] != length:
            raise ValueError("Tokenization differs from serving; discard this result")
        return result
    finally:
        # Leave the worker in native mode after successful or failed generation.
        # Concurrent direct clients remain unsupported; the CLI holds a host lock.
        rpc["kwargs"]["mode"] = "off"
        sender(profile, "/collective_rpc", rpc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["plan", "preflight", "start", "stop", "status", "ask"]
    )
    parser.add_argument("--config", type=Path, default=ROOT / "state/startup.toml")
    parser.add_argument("--rank", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Required for start; does not qualify routine service",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--request", type=Path)
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    state = ROOT / f"state/startup-rank{args.rank}.json"
    # Recovery must work even if the operator has just mistyped the TOML.
    if args.action == "stop":
        if os.name != "posix":
            parser.error("Run stop on the Linux model host")
        current = read_json(state)
        inspect_owned(current["name"])
        print(service.run("docker", "stop", current["name"]))
        return
    profile = settings.load(args.config)
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "scope": "experimental-reference",
                    "fingerprint": settings.fingerprint(profile),
                    "command": command(
                        profile,
                        args.config,
                        args.rank,
                        f"glm53-startup-r{args.rank}-RUN",
                    ),
                    "generation": profile["generation"],
                    "resources": profile["resources"],
                    "lpa": profile["lpa"],
                },
                indent=2,
            )
        )
        return
    if os.name != "posix":
        parser.error("Run this action on the Linux model host; plan works on Windows")
    if args.action in ("status", "ask"):
        current = read_json(state)
        info = inspect_owned(current["name"])
        if args.action == "status":
            print(
                json.dumps(
                    {
                        "name": current["name"],
                        "state": info["State"],
                        "settings_changed": current["fingerprint"]
                        != settings.fingerprint(profile),
                    },
                    indent=2,
                )
            )
        else:
            if args.rank != 0 or not info["State"]["Running"]:
                parser.error("ask requires the running rank 0")
            inspect_owned(current["name"], settings.fingerprint(profile))
            if bool(args.prompt) == bool(args.request):
                parser.error("Supply one --prompt or --request JSON")
            import fcntl

            with (ROOT / "state/startup-request.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                body = (
                    read_json(args.request)
                    if args.request
                    else {"messages": [{"role": "user", "content": args.prompt}]}
                )
                print(json.dumps(ask(profile, body), ensure_ascii=False, indent=2))
        return
    if args.action == "start" and not args.experimental:
        parser.error("start requires --experimental; routine service remains gated")
    if args.action == "start":

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        for signum in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, interrupted)
    result = preflight(profile, args.config, args.rank)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(2)
    if args.action == "preflight":
        return
    if state.exists() and inspect_owned(read_json(state)["name"])["State"]["Running"]:
        raise ValueError("The previous rank is still running; stop it first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"glm53-startup-r{args.rank}-{stamp.lower()}"
    record = ROOT / "records" / (stamp + f"-startup-r{args.rank}")
    record.mkdir(parents=True)
    (ROOT / "state/tp2-runtime-cache").mkdir(parents=True, exist_ok=True)
    if profile["profiling"]["enabled"]:
        (ROOT / "records/profiles" / name).mkdir(parents=True)
    cmd = command(profile, args.config, args.rank, name)
    write_json(record / "preflight.json", result)
    write_json(record / "settings.json", profile)
    write_json(record / "command.json", cmd)
    print(service.run(*cmd), flush=True)
    try:
        write_json(
            state,
            {
                "name": name,
                "fingerprint": settings.fingerprint(profile),
                "record": str(record),
            },
        )
    except BaseException:
        inspect_owned(name)
        service.run("docker", "stop", name)
        raise
    print(
        "Supervising in foreground; Ctrl+C, low memory or an enabled deadline stops this rank.",
        flush=True,
    )
    supervise(profile, name, record)


if __name__ == "__main__":
    main()
