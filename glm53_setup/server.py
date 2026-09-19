"""Run a serial TP=2 reference experiment with a configurable lifetime."""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from . import agreement, capacity, host, model_http, mojibake, warmup
from . import server_config as settings
from .config import MODEL_LAYERS, ROOT, load_lock
from .host import available_gib
from .io import read_json, write_json

LABEL = "glm53.experiment.startup"
# Where the pinned image keeps the GLM model sources that overlays replace.
VLLM_MODEL_DIR = "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia"
# Where the reference image installs this package (build-reference); a module
# newer than the image is bind-mounted there so a worker extension can import it.
IMAGE_PACKAGE_DIR = "/opt/glm53/glm53_setup"
# The backend patch imports the reference attention from this copy.
IMAGE_REFERENCE = "/usr/local/lib/python3.12/dist-packages/glm53_reference.py"


def projector_path(profile, config_path):
    return (config_path.parent / profile["lpa"]["projector"]).resolve()


def model_path(profile, cache):
    derived = settings.derived_checkpoint(profile)
    if derived:
        # Undeclared modules resolve unquantized under MIXED_PRECISION, so the
        # BF16 draft layer needs no metadata view.
        return Path(derived["path"])
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
    derived = settings.derived_checkpoint(profile)
    if derived:
        args += ["-v", f"{derived['path']}:/derived:ro"]
        for overlay in derived["overlays"]:
            target = f"{VLLM_MODEL_DIR}/{overlay['target']}"
            args += ["-v", f"{overlay['source']}:{target}:ro"]
    if profile["lpa"]["enabled"]:
        target = "/lpa/projector.pt"
        args += ["-v", f"{projector_path(profile, config_path)}:{target}:ro"]
    if profile["validation"].get("memory_probe"):
        # The probe is newer than the image; mount the checkout's copy.
        # So is the stable top-k the probe can switch in.
        for name in ("memory_probe.py", "stable_topk.py"):
            source = ROOT / "glm53_setup/runtime" / name
            args += ["-v", f"{source}:{IMAGE_PACKAGE_DIR}/runtime/{name}:ro"]
    if profile["runtime"].get("fa2_attention"):
        # The FA2 path and its dispatch are newer than the image, and so is the
        # fused unpack that takes its element count at run time: the image's
        # copy compiles one kernel per size, which FA2's varying row counts leak.
        runtime = ROOT / "glm53_setup/runtime"
        reference = runtime / "reference_attention.py"
        args += [
            "-v",
            f"{runtime / 'fused_unpack.py'}:{IMAGE_PACKAGE_DIR}/runtime/fused_unpack.py:ro",
            "-v",
            f"{runtime / 'fa2_attention.py'}:{IMAGE_PACKAGE_DIR}/runtime/fa2_attention.py:ro",
            "-v",
            f"{reference}:{IMAGE_PACKAGE_DIR}/runtime/reference_attention.py:ro",
            "-v",
            f"{reference}:{IMAGE_REFERENCE}:ro",
        ]
    if profile["profiling"]["enabled"]:
        args += ["-v", f"{ROOT / 'records/profiles' / name}:/profiles"]
    for key, value in settings.environment(profile, rank).items():
        args += ["-e", f"{key}={value}"]
    return args + [
        "--entrypoint",
        "vllm",
        settings.selected_image(profile),
        *settings.serve_args(
            profile,
            rank,
            "/derived" if derived else "/hf/" + model.relative_to(cache).as_posix(),
        ),
    ]


def inspect_owned(name, fingerprint=None):
    info = json.loads(host.run("docker", "inspect", name))[0]
    owner = (info["Config"].get("Labels") or {}).get(LABEL)
    if not owner or (fingerprint and owner != fingerprint):
        raise ValueError("Container does not belong to this server profile")
    return info


def image_capability_checks(profile, image):
    """Each enabled feature must find its API marker baked into the image env."""
    runtime, validation = profile["runtime"], profile["validation"]
    required = [
        (
            "pipeline_support",
            "GLM53_PIPELINE_API=1",
            runtime["pipeline_parallel_size"] == 2,
        ),
        (
            "expert_parallel_support",
            "GLM53_EXPERT_PARALLEL_API=1",
            runtime["expert_parallel"] or validation["expert_worker"],
        ),
        ("component_worker", "GLM53_COMPONENT_API=1", validation["component_worker"]),
        (
            "fused_unpack_support",
            "GLM53_FUSED_UNPACK_SUPPORTED=1",
            profile["cache"]["fused_unpack"],
        ),
        (
            "decode_graph_support",
            "GLM53_DECODE_GRAPH_API=1",
            settings.decode_graphs(profile),
        ),
        (
            "async_index_check_support",
            "GLM53_ASYNC_INDEX_CHECK_API=1",
            settings.asynchronous_index_checks(profile),
        ),
        ("lpa_worker", "GLM53_LPA_API=2", profile["lpa"]["enabled"]),
        ("apc_lpa_support", "GLM53_APC_LPA_API=1", settings.apc_lpa_enabled(profile)),
        ("reference_attention", "GLM53_REFERENCE_ATTENTION=1", True),
        (
            "moe_order_support",
            "GLM53_MOE_ORDER_API=1",
            runtime.get("canonical_moe_order", False),
        ),
    ]
    env = image["Config"].get("Env") or []
    return {key: marker in env for key, marker, enabled in required if enabled}


def derived_checks(profile, metadata):
    """Fail closed unless the checkpoint and each overlay are the declared ones."""
    derived = settings.derived_checkpoint(profile)
    if not derived:
        return {}
    quantization = metadata.get("quantization_config") or {}
    image = settings.selected_image(profile)

    def overlay_matches(overlay):
        source = Path(overlay["source"])
        if not source.is_file():
            return False
        content = source.read_bytes()
        base = host.run(
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "sha256sum",
            image,
            f"{VLLM_MODEL_DIR}/{overlay['target']}",
        )
        return (
            hashlib.sha256(content).hexdigest() == overlay["sha256"]
            and overlay["marker"].encode() in content
            and base.split()[:1] == [overlay["base_sha256"]]
        )

    draft = f".layers.{MODEL_LAYERS}."
    return {
        "derived_checkpoint": quantization.get("quant_algo") == "MIXED_PRECISION"
        and (quantization.get("producer") or {}).get("requant_target")
        == derived["requant_target"],
        "derived_mtp_draft_unquantized": not profile["mtp"]["enabled"]
        or not any(draft in key for key in quantization.get("quantized_layers", {})),
        "derived_overlays": all(map(overlay_matches, derived["overlays"])),
    }


def preflight(profile, config_path, rank, *, check_memory=True):
    cache = Path.home() / ".cache/huggingface"
    lock = load_lock()
    source = host.snapshot_from_state(
        read_json(ROOT / "state/download-status.json"), lock
    )
    # The pinned snapshot must be on the host whatever is served from it.
    pinned = dict(profile["runtime"])
    pinned.pop("derived_checkpoint", None)
    expected = model_path(
        {**profile, "runtime": pinned, "mtp": {**profile["mtp"], "enabled": False}},
        cache,
    )
    if source.resolve() != expected.resolve():
        raise ValueError("Download state must identify the pinned HF cache snapshot")
    model = model_path(profile, cache)
    metadata = read_json(model / "config.json")
    checks = host.fabric_checks(settings.site(profile, rank))
    checks["full_model"] = metadata["text_config"][
        "num_hidden_layers"
    ] == MODEL_LAYERS and not metadata.get("_test_fixture_only")
    checks.update(derived_checks(profile, metadata))
    if profile["mtp"]["enabled"] and not settings.derived_checkpoint(profile):
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
        host.run("docker", "image", "inspect", settings.selected_image(profile))
    )[0]
    checks["image_id"] = image["Id"] == settings.selected_image(profile)
    checks.update(image_capability_checks(profile, image))
    # Any pair of this launcher carries LABEL, including the old pair that is
    # still running while cluster switch prepares the new profile.
    foreign = host.foreign_gpu_containers(host.running_containers(), LABEL)
    checks["exclusive_gpu"] = not foreign
    if check_memory:
        checks["startup_memory"] = (
            available_gib() >= profile["resources"]["minimum_available_gib"]
        )
    # Reference-profile preflight; qualification status lives in the documents.
    return {
        "scope": "experimental-reference",
        "checks": checks,
        "foreign_gpu_containers": foreign,
        "passed": all(checks.values()),
    }


def freeze(profile, environ=None):
    resolved = settings.resolve_launch(profile, environ)
    return {"profile": resolved, "fingerprint": settings.fingerprint(resolved)}


def thaw(manifest):
    if not isinstance(manifest, dict) or manifest.keys() != {"profile", "fingerprint"}:
        raise ValueError("Invalid frozen launch manifest")
    settings.validate(manifest["profile"])
    if manifest["fingerprint"] != settings.fingerprint(manifest["profile"]):
        raise ValueError("Frozen launch manifest no longer matches this checkout/lock")
    return manifest["profile"]


PROGRESS_SIGNALS = (
    "vllm:num_requests_running",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


def parse_metrics(text, names):
    """Sum each named Prometheus sample over its label sets; absent names are omitted."""
    totals = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name = head.split("{", 1)[0]
        if name in names:
            try:
                totals[name] = totals.get(name, 0.0) + float(value)
            except ValueError:
                continue
    return totals


def metrics_text(profile, timeout=2):
    with model_http.open_response(
        f"http://127.0.0.1:{profile['api']['port']}", "/metrics", timeout=timeout
    ) as response:
        return response.read().decode("utf-8", "replace")


def progress_sample(profile):
    """The four signals a wedged engine stops moving; None when unobservable."""
    try:
        sample = parse_metrics(metrics_text(profile), PROGRESS_SIGNALS)
    except Exception:  # noqa: BLE001 - a read that fails or stalls is not a stalled engine
        return None
    return sample if PROGRESS_SIGNALS[0] in sample else None


def supervise(profile, name, record, rank):
    seconds = profile["resources"]["run_seconds"]
    deadline = time.monotonic() + seconds if seconds else None
    # Only the head serves /metrics; /health answers 200 while the engine is
    # wedged, so progress is read from the request counters instead.
    stall = profile["resources"].get("stall_seconds", 0) if rank == 0 else 0
    last, moved = None, (time.monotonic() if stall else None)
    try:
        with (record / "resources.jsonl").open("a", encoding="utf-8") as log:
            while True:
                info = inspect_owned(name)
                if not info["State"]["Running"]:
                    break
                available = available_gib()
                entry = {"epoch": time.time(), "available_gib": available}
                entry.update(host.memory_sample())
                entry.update(host.container_memory_sample(info.get("Id", "")))
                reason = None
                if available < profile["resources"]["reserve_gib"]:
                    reason = {"reason": "memory-reserve"}
                elif deadline is not None and time.monotonic() >= deadline:
                    reason = {"reason": "run-deadline"}
                elif stall:
                    sample = progress_sample(profile)
                    if sample is not None:
                        entry["progress"] = sample
                        if sample[PROGRESS_SIGNALS[0]] <= 0 or sample != last:
                            last, moved = sample, time.monotonic()
                        elif time.monotonic() - moved >= stall:
                            reason = {
                                "reason": "engine-stall",
                                "stall_seconds": stall,
                                "progress": sample,
                            }
                log.write(json.dumps(entry) + "\n")
                log.flush()
                if reason is not None:
                    write_json(record / "stop-reason.json", reason)
                    break
                time.sleep(2)  # Same sampling cadence as the measured experiments.
    finally:
        info = inspect_owned(name)
        if info["State"]["Running"]:
            host.run("docker", "stop", name)
        write_json(record / "container-inspect.json", inspect_owned(name))
        with (record / "server.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                ["docker", "logs", name],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )


@contextlib.contextmanager
def request_lock():
    """Hold the host lock that serializes direct clients of the running head."""
    import fcntl

    with (ROOT / "state/startup-request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def running_head(profile):
    """Return rank 0's recorded state and container info owned by this profile."""
    state = read_json(ROOT / "state/startup-rank0.json")
    return state, inspect_owned(state["name"], settings.fingerprint(profile))


def post(profile, path, body):
    return model_http.post_json(
        f"http://127.0.0.1:{profile['api']['port']}",
        path,
        body,
        timeout=profile["generation"]["timeout_seconds"],
    )


def collective_rpc(profile, method, **kwargs):
    body = {"method": method, "kwargs": kwargs, "timeout": 600}
    return post(profile, "/collective_rpc", body)["results"]


def container_logs(name):
    return subprocess.run(
        ["docker", "logs", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.decode("utf-8", "replace")


def dev_endpoints(profile):
    return (
        profile["api"].get("dev_endpoints", False)
        or profile["lpa"]["enabled"]
        or profile["validation"]["component_worker"]
        or profile["validation"]["expert_worker"]
    )


def capacity_report(profile, name):
    """Decompose the KV boot line for the running head; read-only."""
    layout = None
    if profile["lpa"]["enabled"] and dev_endpoints(profile):
        layout = collective_rpc(profile, "apc_cache_layout")[0]
    return capacity.summarize(
        profile, container_logs(name), metrics_text(profile), layout
    )


def warmup_report(profile, name):
    """Run the request ladder against the running head."""

    def count_tokens(text):
        return post(
            profile,
            "/tokenize",
            {"model": profile["api"]["served_model_name"], "prompt": text},
        )["count"]

    def reset():
        if post(profile, "/reset_prefix_cache", {}) != {"success": True}:
            raise ValueError("Prefix cache reset was not acknowledged")

    return warmup.run(
        profile,
        ask=lambda request: ask(profile, request),
        count_tokens=count_tokens,
        logs=lambda: container_logs(name),
        clock=time.monotonic,
        reset=reset if dev_endpoints(profile) else None,
    )


def warmup_running(profile):
    """Ladder for the running rank 0 owned by this profile; records the result."""
    state, info = running_head(profile)
    if not info["State"]["Running"]:
        raise ValueError("warmup requires the running rank 0")
    with request_lock():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record = ROOT / "records" / (stamp + "-warmup-r0")
        record.mkdir(parents=True)
        result = warmup_report(profile, state["name"])
    result["record"] = str(record)
    write_json(record / "result.json", result)
    return result


def mojibake_running(profile):
    """Japanese/Korean broken-character check on the running rank 0; recorded."""
    state, info = running_head(profile)
    if not info["State"]["Running"]:
        raise ValueError("mojibake requires the running rank 0")
    with request_lock():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record = ROOT / "records" / (stamp + "-mojibake-r0")
        record.mkdir(parents=True)
        result = mojibake.run(lambda request: ask(profile, request))
    result["record"] = str(record)
    write_json(record / "result.json", result)
    return result


def agreement_senders(profile, sender=post):
    """Tokenize and completions senders for the running rank 0.

    The text is tokenized once, then sent back as token ids, so the scored
    positions are exactly the tokens the server saw. LPA's native mode
    rewrites prefill outside the shared cache, so the reading is only taken
    with LPA off or in its APC-first form (the same guard as ``ask``).
    """
    if profile["lpa"]["enabled"] and not settings.apc_lpa_enabled(profile):
        raise ValueError(
            "agreement requires LPA off or APC-first; native LPA rewrites prefill"
        )
    model = profile["api"]["served_model_name"]

    def tokenize(text):
        return sender(profile, "/tokenize", {"model": model, "prompt": text})["tokens"]

    def complete(token_ids, top_k):
        return sender(
            profile,
            "/v1/completions",
            {
                "model": model,
                "prompt": token_ids,
                "max_tokens": 1,
                "temperature": 0,
                "seed": profile["runtime"]["seed"],
                "prompt_logprobs": top_k,
            },
        )

    return tokenize, complete


def agreement_running(profile, reference=None):
    """Teacher-forced reading on the running rank 0; compared with a saved run."""
    state, info = running_head(profile)
    if not info["State"]["Running"]:
        raise ValueError("agreement requires the running rank 0")
    tokenize, complete = agreement_senders(profile)
    with request_lock():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record = ROOT / "records" / (stamp + "-agreement-r0")
        record.mkdir(parents=True)
        result = agreement.run(tokenize, complete)
    if reference is not None:
        result["reference"] = str(reference)
        result["comparison"] = agreement.compare_records(read_json(reference), result)
    result["record"] = str(record)
    write_json(record / "result.json", result)
    return result


def ask(profile, request, sender=post):
    body = settings.request_body(profile, request)
    if not profile["lpa"]["enabled"] or settings.apc_lpa_enabled(profile):
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
        "action",
        choices=[
            "plan",
            "freeze",
            "assets",
            "preflight",
            "start",
            "stop",
            "status",
            "ask",
            "capacity",
            "warmup",
            "mojibake",
            "agreement",
        ],
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Saved agreement result.json to compare this run against",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "state/server.toml")
    parser.add_argument("--rank", type=int, choices=[0, 1], default=0)
    parser.add_argument("--prompt")
    parser.add_argument("--run-id", help="Unique coordinator-owned launch ID")
    parser.add_argument("--request", type=Path)
    parser.add_argument(
        "--launch",
        type=Path,
        help="Shared frozen JSON from server freeze; host allocator environment is ignored",
    )
    parser.add_argument("--output", type=Path, help="New output file for server freeze")
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    state = ROOT / f"state/startup-rank{args.rank}.json"
    # Recovery must work even if the operator has just mistyped the TOML.
    if args.action == "stop":
        if os.name != "posix":
            parser.error("Run stop on the Linux model host")
        current = read_json(state)
        inspect_owned(current["name"])
        print(host.run("docker", "stop", current["name"]))
        return
    profile = (
        thaw(read_json(args.launch)) if args.launch else settings.load(args.config)
    )
    if args.action == "freeze":
        if not args.output or args.launch:
            parser.error("freeze requires --output and a TOML --config")
        manifest = freeze(profile)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        print(
            json.dumps(
                {"fingerprint": manifest["fingerprint"], "output": str(args.output)}
            )
        )
        return
    if (
        args.action in ("start", "preflight", "assets")
        and not args.launch
        and "PYTORCH_CUDA_ALLOC_CONF" in os.environ
    ):
        parser.error(
            "Freeze the launch-origin allocator once with server freeze and pass the same --launch JSON to both ranks"
        )
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
    if args.action in ("capacity", "warmup", "mojibake", "agreement"):
        if args.rank != 0:
            parser.error(f"{args.action} requires rank 0")
        if args.action in ("warmup", "mojibake", "agreement"):
            if args.action == "warmup":
                result = warmup_running(profile)
            elif args.action == "agreement":
                result = agreement_running(profile, args.reference)
                # Per-position rows stay in the record; the terminal gets the rates.
                for row in result["texts"]:
                    for key in ("rows", "ranks", "logprobs", "prompt_token_ids"):
                        row.pop(key, None)
                result.pop("first_raw_response", None)
            else:
                result = mojibake_running(profile)
                # The answers stay in the record; the terminal gets the verdicts.
                for row in result["runs"]:
                    row.pop("content", None)
                    row.pop("reasoning", None)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result["passed"]:
                raise SystemExit(1)
            return
        current, info = running_head(profile)
        if not info["State"]["Running"]:
            parser.error("capacity requires the running rank 0")
        print(json.dumps(capacity_report(profile, current["name"]), indent=2))
        return
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
            with request_lock():
                body = (
                    read_json(args.request)
                    if args.request
                    else {"messages": [{"role": "user", "content": args.prompt}]}
                )
                print(json.dumps(ask(profile, body), ensure_ascii=False, indent=2))
        return
    if args.action == "start":

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        for signum in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, interrupted)
    result = preflight(
        profile, args.config, args.rank, check_memory=args.action != "assets"
    )
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(2)
    if args.action in ("preflight", "assets"):
        return
    if state.exists() and inspect_owned(read_json(state)["name"])["State"]["Running"]:
        raise ValueError("The previous rank is still running; stop it first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.run_id:
        import re

        if not re.fullmatch(r"[0-9a-f]{32}", args.run_id):
            parser.error("run-id must be a 32-character hexadecimal launch ID")
    name = f"glm53-startup-r{args.rank}-{args.run_id or stamp.lower()}"
    record = ROOT / "records" / (stamp + f"-server-r{args.rank}")
    record.mkdir(parents=True)
    (ROOT / "state/tp2-runtime-cache").mkdir(parents=True, exist_ok=True)
    if profile["profiling"]["enabled"]:
        (ROOT / "records/profiles" / name).mkdir(parents=True)
    cmd = command(profile, args.config, args.rank, name)
    write_json(record / "preflight.json", result)
    write_json(record / "settings.json", profile)
    write_json(record / "command.json", cmd)
    print(host.run(*cmd), flush=True)
    try:
        write_json(
            state,
            {
                "name": name,
                "fingerprint": settings.fingerprint(profile),
                "record": str(record),
                "config_path": str(args.config),
            },
        )
    except BaseException:
        inspect_owned(name)
        host.run("docker", "stop", name)
        raise
    print(
        "Supervising in foreground; Ctrl+C, low memory or an enabled deadline stops this rank.",
        flush=True,
    )
    supervise(profile, name, record, args.rank)


if __name__ == "__main__":
    main()
