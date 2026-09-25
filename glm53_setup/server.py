"""The launcher of the TP=2 serving pair: plan and start a rank, supervise it,
and query or check the running head (docs/server-configuration.md)."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

from . import agreement, capacity, host, model_http, mojibake, warmup
from . import server_config as settings
from .config import (
    DEFAULT_PROFILE,
    MODEL_LAYERS,
    MTP_VIEW_KEY,
    RECORDS,
    ROOT,
    STATE,
    load_lock,
)
from .download import STATUS_FILE
from .host import available_gib
from .io import read_json, write_json
from .runtime.patch_nope_reference import REFERENCE_FILE

LABEL = "glm53.experiment.startup"
# The image's site directory; site runs the import lines of .pth files found here.
SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"
# Where the pinned image keeps the GLM model sources that overlays replace.
VLLM_MODEL_DIR = f"{SITE_PACKAGES}/vllm/models/glm5next/nvidia"
# Where the reference image installs this package (build-reference); a module
# newer than the image is bind-mounted there so a worker extension can import it.
IMAGE_PACKAGE_DIR = "/opt/glm53/glm53_setup"
# The backend patch imports the reference attention from this copy.
IMAGE_REFERENCE = f"{SITE_PACKAGES}/{REFERENCE_FILE}"


def hf_cache():
    """The host's shared Hugging Face cache, which the download fills."""
    return Path.home() / ".cache/huggingface"


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
    cache = cache or hf_cache()
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
        f"{STATE / 'tp2-runtime-cache'}:/root/.cache",
    ]
    if profile["nodes"][rank].get("cpuset_cpus") is not None:
        args += ["--cpuset-cpus", profile["nodes"][rank]["cpuset_cpus"]]
    derived = settings.derived_checkpoint(profile)
    if derived:
        args += ["-v", f"{derived['path']}:/derived:ro"]
        for overlay in derived["overlays"]:
            target = f"{VLLM_MODEL_DIR}/{overlay['target']}"
            args += ["-v", f"{overlay['source']}:{target}:ro"]
    if profile["lpa"]["enabled"]:
        target = settings.LPA_PROJECTOR
        args += ["-v", f"{projector_path(profile, config_path)}:{target}:ro"]
    if settings.optional(profile, "validation", "memory_probe"):
        # The probe is newer than the image; mount the checkout's copy.
        source = ROOT / "glm53_setup/runtime/memory_probe.py"
        args += ["-v", f"{source}:{IMAGE_PACKAGE_DIR}/runtime/memory_probe.py:ro"]
    if settings.optional(profile, "runtime", "inductor_deterministic"):
        # Keeps TORCHINDUCTOR_DETERMINISTIC on through Dynamo's state restore,
        # which turns it off after the first compiled frame (torch 2.12.1 and
        # 2.13, pytorch/pytorch#198563). The .pth runs the module at
        # interpreter start in every container process.
        runtime = ROOT / "glm53_setup/runtime"
        args += [
            "-v",
            f"{runtime / 'inductor_pin.py'}:{IMAGE_PACKAGE_DIR}/runtime/inductor_pin.py:ro",
            "-v",
            f"{runtime / 'inductor_pin_pth.txt'}:{SITE_PACKAGES}/glm53-inductor-pin.pth:ro",
        ]
    if settings.optional(profile, "runtime", "fa2_attention"):
        # cc-defer: redundant on images that carry GLM53_FA2_ATTENTION_API=1; drop
        # the mounts once no image without it can be a recovery target. That is now
        # checked: preflight requires the marker of an FA2 profile, recovery
        # included (fa2_attention_support).
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
        args += ["-v", f"{RECORDS / 'profiles' / name}:/profiles"]
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


def verify_cpu_set(profile, rank, name):
    """Read back an explicitly requested Docker placement before recording a start."""
    requested = settings.cpuset_cpus(profile, rank)
    if requested is None:
        return
    info = inspect_owned(name, settings.fingerprint(profile))
    actual = info["HostConfig"].get("CpusetCpus") or ""
    # Docker may write the list in its own form; compare the CPUs, not the text.
    if not actual or settings.cpu_list(actual, "Docker CPU set") != requested:
        raise ValueError("Docker CPU set does not match the configured node")


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


def preflight(profile, config_path, rank, *, check_memory=True, recovery=False):
    cache = hf_cache()
    lock = load_lock()
    source = host.snapshot_from_state(read_json(STATE / STATUS_FILE), lock)
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
    requested_cpus = settings.cpuset_cpus(profile, rank)
    if requested_cpus is not None:
        try:
            checks["cpu_set_available"] = requested_cpus <= os.sched_getaffinity(0)
        except (AttributeError, OSError):
            checks["cpu_set_available"] = False
    checks["full_model"] = metadata["text_config"][
        "num_hidden_layers"
    ] == MODEL_LAYERS and not metadata.get("_test_fixture_only")
    checks.update(derived_checks(profile, metadata))
    if profile["mtp"]["enabled"] and not settings.derived_checkpoint(profile):
        view = metadata.get(MTP_VIEW_KEY, {})
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
    checks.update(settings.image_capability_checks(profile, image, recovery=recovery))
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
        "warnings": settings.capability_warnings(profile, image, recovery=recovery),
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


def api_origin(profile):
    """The running head's API as a client on the same host reaches it."""
    return f"http://127.0.0.1:{profile['api']['port']}"


def metrics_text(profile, timeout=2, errors="replace"):
    with model_http.open_response(
        api_origin(profile), "/metrics", timeout=timeout
    ) as response:
        return response.read().decode("utf-8", errors)


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
    stall = settings.optional(profile, "resources", "stall_seconds") if rank == 0 else 0
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

    with (STATE / "startup-request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def running_head(profile):
    """Return rank 0's recorded state and container info owned by this profile."""
    state = read_json(state_path(0))
    return state, inspect_owned(state["name"], settings.fingerprint(profile))


def post(profile, path, body):
    return model_http.post_json(
        api_origin(profile),
        path,
        body,
        timeout=profile["generation"]["timeout_seconds"],
    )


def collective_rpc(profile, method, **kwargs):
    body = {"method": method, "kwargs": kwargs, "timeout": 600}
    return post(profile, "/collective_rpc", body)["results"]


def reset_prefix_cache(profile):
    """Empty the running head's prefix cache; a reset it does not acknowledge fails."""
    if post(profile, "/reset_prefix_cache", {}) != {"success": True}:
        raise ValueError("Prefix cache reset was not acknowledged")


def container_logs(name):
    return subprocess.run(
        ["docker", "logs", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.decode("utf-8", "replace")


def capacity_report(profile, name):
    """Decompose the KV boot line for the running head; read-only."""
    layout = None
    if profile["lpa"]["enabled"] and settings.dev_mode(profile):
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

    return warmup.run(
        profile,
        ask=lambda request: ask(profile, request),
        count_tokens=count_tokens,
        logs=lambda: container_logs(name),
        clock=time.monotonic,
        reset=(lambda: reset_prefix_cache(profile))
        if settings.dev_mode(profile)
        else None,
    )


def warmup_running(profile):
    """Ladder for the running rank 0 owned by this profile; records the result."""
    state, info = running_head(profile)
    if not info["State"]["Running"]:
        raise ValueError("warmup requires the running rank 0")
    with request_lock():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record = RECORDS / (stamp + "-warmup-r0")
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
        record = RECORDS / (stamp + "-mojibake-r0")
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
        record = RECORDS / (stamp + "-agreement-r0")
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


def state_path(rank):
    """Where this rank records the container it owns."""
    return STATE / f"startup-rank{rank}.json"


def container_name(rank, run_id):
    """The container a launch of this rank runs as; the coordinator polls it by name."""
    return f"glm53-startup-r{rank}-{run_id}"


def report_verdict(result):
    """Print a recorded verdict; a failed check leaves a non-zero status."""
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def act_plan(cli, args, profile):
    """Show what a launch would run, without touching the host."""
    print(
        json.dumps(
            {
                "scope": "experimental-reference",
                "fingerprint": settings.fingerprint(profile),
                "command": command(
                    profile, args.config, args.rank, container_name(args.rank, "RUN")
                ),
                "generation": profile["generation"],
                "resources": profile["resources"],
                "lpa": profile["lpa"],
            },
            indent=2,
        )
    )


def act_freeze(cli, args, profile):
    """Write the shared launch manifest both ranks will be started from."""
    if not args.output or args.launch:
        cli.error("freeze requires --output and a TOML --config")
    manifest = freeze(profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"fingerprint": manifest["fingerprint"], "output": str(args.output)})
    )


def act_stop(cli, args, profile):
    """Stop this rank's container; runs without a loadable profile."""
    if os.name != "posix":
        cli.error("Run stop on the Linux model host")
    current = read_json(state_path(args.rank))
    inspect_owned(current["name"])
    print(host.run("docker", "stop", current["name"]))


def act_status(cli, args, profile):
    """Report this rank's container state and whether the settings moved."""
    current = read_json(state_path(args.rank))
    info = inspect_owned(current["name"])
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


def act_ask(cli, args, profile):
    """Send one request to the running head under the host request lock."""
    current = read_json(state_path(args.rank))
    info = inspect_owned(current["name"])
    if args.rank != 0 or not info["State"]["Running"]:
        cli.error("ask requires the running rank 0")
    inspect_owned(current["name"], settings.fingerprint(profile))
    if bool(args.prompt) == bool(args.request):
        cli.error("Supply one --prompt or --request JSON")
    with request_lock():
        body = (
            read_json(args.request)
            if args.request
            else {"messages": [{"role": "user", "content": args.prompt}]}
        )
        print(json.dumps(ask(profile, body), ensure_ascii=False, indent=2))


def act_capacity(cli, args, profile):
    """Decompose the running head's KV boot line; read-only."""
    current, info = running_head(profile)
    if not info["State"]["Running"]:
        cli.error("capacity requires the running rank 0")
    print(json.dumps(capacity_report(profile, current["name"]), indent=2))


def act_warmup(cli, args, profile):
    """Run the post-readiness request ladder against the running head."""
    report_verdict(warmup_running(profile))


def act_mojibake(cli, args, profile):
    """Check the running head for Japanese/Korean broken characters."""
    result = mojibake_running(profile)
    # The answers stay in the record; the terminal gets the verdicts.
    for row in result["runs"]:
        row.pop("content", None)
        row.pop("reasoning", None)
    report_verdict(result)


def act_agreement(cli, args, profile):
    """Score the running head against a saved reading."""
    result = agreement_running(profile, args.reference)
    # Per-position rows stay in the record; the terminal gets the rates.
    for row in result["texts"]:
        for key in ("rows", "ranks", "logprobs", "prompt_token_ids"):
            row.pop(key, None)
    result.pop("first_raw_response", None)
    report_verdict(result)


def act_launch(cli, args, profile):
    """Preflight this rank, and for ``start`` keep supervising it."""
    if args.action == "start":

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        for signum in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, interrupted)
    result = preflight(
        profile,
        args.config,
        args.rank,
        check_memory=args.action != "assets",
        recovery=args.recovery,
    )
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(2)
    if args.action in ("preflight", "assets"):
        return
    start_rank(cli, args, profile, result)


def start_rank(cli, args, profile, result):
    """Launch this rank's container and supervise it in the foreground."""
    state = state_path(args.rank)
    if state.exists() and inspect_owned(read_json(state)["name"])["State"]["Running"]:
        raise ValueError("The previous rank is still running; stop it first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.run_id and not re.fullmatch(r"[0-9a-f]{32}", args.run_id):
        cli.error("run-id must be a 32-character hexadecimal launch ID")
    name = container_name(args.rank, args.run_id or stamp.lower())
    record = RECORDS / (stamp + f"-server-r{args.rank}")
    record.mkdir(parents=True)
    (STATE / "tp2-runtime-cache").mkdir(parents=True, exist_ok=True)
    if profile["profiling"]["enabled"]:
        (RECORDS / "profiles" / name).mkdir(parents=True)
    cmd = command(profile, args.config, args.rank, name)
    write_json(record / "preflight.json", result)
    write_json(record / "settings.json", profile)
    write_json(record / "command.json", cmd)
    print(host.run(*cmd), flush=True)
    try:
        verify_cpu_set(profile, args.rank, name)
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
        # docker run succeeded with this unique name. Stop it even if the
        # read-back inspect itself failed, rather than leaving an untracked rank.
        host.run("docker", "stop", name)
        raise
    print(
        "Supervising in foreground; Ctrl+C, low memory, an enabled deadline or an "
        "engine stall stops this rank.",
        flush=True,
    )
    supervise(profile, name, record, args.rank)


Action = namedtuple(
    "Action", ("handler", "needs_profile", "windows", "rank0", "launch_path")
)


def entry(
    handler, *, needs_profile=True, windows=False, rank0=False, launch_path=False
):
    """One action and the preconditions ``main`` enforces before calling it.

    ``needs_profile`` false reaches the handler without reading the operator's
    TOML; ``windows`` true runs away from the model host; ``rank0`` refuses a
    peer rank; ``launch_path`` refuses an allocator inherited from the host.
    """
    return Action(handler, needs_profile, windows, rank0, launch_path)


# Insertion order is the order argparse prints in --help.
ACTIONS = {
    "plan": entry(act_plan, windows=True),
    "freeze": entry(act_freeze, windows=True),
    "assets": entry(act_launch, launch_path=True),
    "preflight": entry(act_launch, launch_path=True),
    "start": entry(act_launch, launch_path=True),
    "stop": entry(act_stop, needs_profile=False),
    "status": entry(act_status),
    "ask": entry(act_ask),
    "capacity": entry(act_capacity, rank0=True),
    "warmup": entry(act_warmup, rank0=True),
    "mojibake": entry(act_mojibake, rank0=True),
    "agreement": entry(act_agreement, rank0=True),
}


def parser():
    """The launcher's argument interface; ``main`` adds the table's guards."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=list(ACTIONS))
    cli.add_argument(
        "--reference",
        type=Path,
        help="Saved agreement result.json to compare this run against",
    )
    cli.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    cli.add_argument("--rank", type=int, choices=[0, 1], default=0)
    cli.add_argument("--prompt")
    cli.add_argument("--run-id", help="Unique coordinator-owned launch ID")
    cli.add_argument("--request", type=Path)
    cli.add_argument(
        "--launch",
        type=Path,
        help="Shared frozen JSON from server freeze; host allocator environment is ignored",
    )
    cli.add_argument("--output", type=Path, help="New output file for server freeze")
    cli.add_argument(
        "--recovery",
        action="store_true",
        help="Restart a pair as it was launched: the coordinator passes this when a "
        "switch restores the previous profile. A new launch does not use it",
    )
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    args.config = args.config.resolve()
    action = ACTIONS[args.action]
    # Recovery must work even if the operator has just mistyped the TOML.
    if not action.needs_profile:
        return action.handler(cli, args, None)
    profile = (
        thaw(read_json(args.launch)) if args.launch else settings.load(args.config)
    )
    if (
        action.launch_path
        and not args.launch
        and "PYTORCH_CUDA_ALLOC_CONF" in os.environ
    ):
        cli.error(
            "Freeze the launch-origin allocator once with server freeze and pass the same --launch JSON to both ranks"
        )
    if args.recovery and not action.launch_path:
        cli.error("--recovery belongs to assets, preflight and start")
    if not action.windows and os.name != "posix":
        cli.error("Run this action on the Linux model host; plan works on Windows")
    if action.rank0 and args.rank != 0:
        cli.error(f"{args.action} requires rank 0")
    return action.handler(cli, args, profile)


if __name__ == "__main__":
    main()
