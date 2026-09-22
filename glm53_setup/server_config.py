"""Typed operator settings shared by the launcher and its chat client."""

import copy
import hashlib
import json
import math
import os
import re
import tomllib
from pathlib import Path, PurePosixPath

from . import host
from .config import MODEL_LAYERS, ROOT, load_lock


def load(path):
    with Path(path).open("rb") as stream:
        profile = tomllib.load(stream)
    validate(profile)
    return profile


def loads(text):
    profile = tomllib.loads(text)
    validate(profile)
    return profile


def decode_graphs(profile):
    """True when the profile asks for decode Graphs.

    ``runtime.decode_graphs`` is the switch; ``runtime.enforce_eager`` is the
    earlier spelling and, when both are present, may not contradict it.
    Absent both, the launch is eager.
    """
    runtime = profile["runtime"]
    if "decode_graphs" in runtime:
        if type(runtime["decode_graphs"]) is not bool:
            raise ValueError("runtime.decode_graphs must be true or false")
        if (
            "enforce_eager" in runtime
            and runtime["enforce_eager"] == runtime["decode_graphs"]
        ):
            raise ValueError("runtime.decode_graphs contradicts runtime.enforce_eager")
        return runtime["decode_graphs"]
    if type(runtime.get("enforce_eager", True)) is not bool:
        raise ValueError("runtime.enforce_eager must be true or false")
    return not runtime["enforce_eager"] if "enforce_eager" in runtime else False


# Keys a profile may omit, per category. The shipped example file is the
# schema; these are the entries whose absence is not a typo.
OPTIONAL_KEYS = {
    "server.runtime": frozenset(
        {
            "cuda_allocator_conf",
            "vision",
            "nccl_channels",
            "derived_checkpoint",
            "canonical_moe_order",
            "stable_indexer_topk",
            "decode_graphs",
            "enforce_eager",
            "fa2_attention",
            "prefix_page_dedup",
        }
    ),
    "server.cache": frozenset(
        {"prefix_cache_retention_interval", "mm_processor_cache_gb"}
    ),
    "server.api": frozenset({"prompt_tokens_details", "dev_endpoints"}),
    "server.validation": frozenset({"memory_probe"}),
    "server.resources": frozenset({"stall_seconds"}),
    "server.generation": frozenset({"warmup", "warmup_long_tokens"}),
    "server.nodes[]": frozenset({"additional_rails"}),
}


def optional_at(path):
    """Which keys may be absent at this point in the schema."""
    if path.startswith("server.nodes["):
        return OPTIONAL_KEYS["server.nodes[]"]
    return OPTIONAL_KEYS.get(path, frozenset())


def check_schema(profile):
    """Match the profile against the shipped example, shape for shape.

    No silent defaults: a typo or a missing category must not quietly change
    a launch, so every key is either present, or named as optional above.
    """
    with (ROOT / "examples/server.example.toml").open("rb") as stream:
        schema = tomllib.load(stream)

    def check(value, expected, path):
        if isinstance(expected, dict):
            optional = optional_at(path)
            if (
                not isinstance(value, dict)
                or value.keys() - optional != expected.keys() - optional
            ):
                raise ValueError(f"Unknown/missing settings in {path}")
            for key, item in expected.items():
                if key not in optional:
                    check(value[key], item, f"{path}.{key}")
        elif isinstance(expected, list):
            if not isinstance(value, list) or len(value) != len(expected):
                raise ValueError(f"Expected exactly two nodes in {path}")
            for index, item in enumerate(value):
                check(item, expected[index], f"{path}[{index}]")
        elif type(expected) is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Expected finite number in {path}")
        elif type(value) is not type(expected):
            raise ValueError(f"Invalid type in {path}")

    check(profile, schema, "server")


def check_optional_shapes(profile):
    """Type-check the keys a profile may omit, and the pairs they exclude."""
    if "prefix_cache_retention_interval" in profile["cache"]:
        interval = profile["cache"]["prefix_cache_retention_interval"]
        if interval != "dense" and (type(interval) is not int or interval < 0):
            raise ValueError(
                "cache.prefix_cache_retention_interval must be dense or a nonnegative integer; runtime validates numeric scheduler-block alignment"
            )
    if "cuda_allocator_conf" in profile["runtime"]:
        allocator = profile["runtime"]["cuda_allocator_conf"]
        if not isinstance(allocator, str) or any(c in allocator for c in "\x00\r\n"):
            raise ValueError(
                "runtime.cuda_allocator_conf must be a single-line string, including empty"
            )
    decode_graphs(profile)
    if type(profile["runtime"].get("vision", False)) is not bool:
        raise ValueError("runtime.vision must be true or false")
    if "nccl_channels" in profile["runtime"]:
        channels = profile["runtime"]["nccl_channels"]
        if type(channels) is not int or channels < 1:
            raise ValueError("runtime.nccl_channels must be a positive integer")
    if "derived_checkpoint" in profile["runtime"]:
        validate_derived(profile["runtime"]["derived_checkpoint"])
    if type(profile["runtime"].get("canonical_moe_order", True)) is not bool:
        raise ValueError("runtime.canonical_moe_order must be true or false")
    if type(profile["runtime"].get("stable_indexer_topk", True)) is not bool:
        raise ValueError("runtime.stable_indexer_topk must be true or false")
    if type(profile["runtime"].get("fa2_attention", False)) is not bool:
        raise ValueError("runtime.fa2_attention must be true or false")
    if type(profile["runtime"].get("prefix_page_dedup", False)) is not bool:
        raise ValueError("runtime.prefix_page_dedup must be true or false")
    if profile["runtime"].get("fa2_attention") and profile["lpa"]["enabled"]:
        # LPA's skip_mla_queries hooks the reference computation only.
        raise ValueError("runtime.fa2_attention excludes LPA")
    if "mm_processor_cache_gb" in profile["cache"]:
        size = profile["cache"]["mm_processor_cache_gb"]
        if type(size) not in (int, float) or not math.isfinite(size) or size < 0:
            raise ValueError(
                "cache.mm_processor_cache_gb must be a finite nonnegative number"
            )
    if type(profile["api"].get("dev_endpoints", False)) is not bool:
        raise ValueError("api.dev_endpoints must be true or false")
    if type(profile["generation"].get("warmup", False)) is not bool:
        raise ValueError("generation.warmup must be true or false")
    for section, key in (
        ("resources", "stall_seconds"),
        ("generation", "warmup_long_tokens"),
    ):
        value = profile[section].get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"{section}.{key} must be a nonnegative integer")


def check_pinned_identity(profile):
    """The schema version and the immutable image IDs this launch is pinned to."""
    if profile["schema_version"] != 1:
        raise ValueError("Unsupported profile schema_version")
    for key in ("reference_image", "lpa_image"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", profile["runtime"][key]):
            raise ValueError(f"runtime.{key} must be an immutable image ID")


def check_magnitudes(profile):
    """Sizes and counts that a launch cannot meaningfully take at zero."""
    for section, keys in {
        "context": ("max_model_len", "max_num_seqs", "max_num_batched_tokens"),
        "cache": ("kv_cache_memory_bytes", "block_size"),
        "generation": ("max_tokens", "timeout_seconds"),
        "resources": (
            "container_memory_gib",
            "minimum_available_gib",
            "reserve_gib",
        ),
    }.items():
        for key in keys:
            if profile[section][key] < 1:
                raise ValueError(f"{section}.{key} must be positive")
    if profile["resources"]["run_seconds"] < 0:
        raise ValueError("resources.run_seconds must be nonnegative (0 = no deadline)")


def check_expert_observer(profile):
    """The EP observer runs alone, on its own eager TP2 launch."""
    if profile["validation"]["expert_worker"] and (
        profile["validation"]["component_worker"]
        or profile["runtime"]["pipeline_parallel_size"] != 1
        or profile["lpa"]["enabled"]
        or profile["mtp"]["enabled"]
        or profile["cache"]["prefix_caching"]
        or decode_graphs(profile)
        or profile["context"]["max_num_seqs"] > 2
    ):
        raise ValueError(
            "EP observer requires independent eager TP2, no other worker/MTP/APC"
        )


def check_parallelism(profile):
    """Index checks, pipeline shape and expert parallelism, and what they exclude."""
    runtime = profile["runtime"]
    if runtime["index_checks"] not in ("auto", "sync", "async"):
        raise ValueError(
            "index_checks must be auto, sync or async; checks cannot be disabled"
        )
    if decode_graphs(profile) and runtime["index_checks"] == "sync":
        raise ValueError("Graph execution requires asynchronous index checks")
    if runtime["pipeline_parallel_size"] not in (1, 2):
        raise ValueError("Only PP sizes 1 and 2 are supported")
    split = runtime["pipeline_split_layer"]
    if not 4 <= split <= MODEL_LAYERS - 2:
        raise ValueError("PP stage boundaries must retain an MLA layer in both stages")
    if runtime["pipeline_parallel_size"] == 2 and (
        runtime["expert_parallel"]
        or decode_graphs(profile)
        or profile["lpa"]["enabled"]
        or profile["mtp"]["enabled"]
        or profile["cache"]["prefix_caching"]
        or profile["cache"]["fused_unpack"]
        or profile["context"]["max_num_seqs"] != 1
    ):
        # cc-defer: independent serial PP evaluation; extend only after the
        # matching optimization and batching combination is qualified.
        raise ValueError("PP2 requires eager, one sequence, no EP/LPA/MTP/fusion/APC")
    if profile["runtime"]["expert_parallel"] and (
        profile["lpa"]["enabled"]
        or profile["mtp"]["enabled"]
        or profile["cache"]["prefix_caching"]
        or profile["cache"]["fused_unpack"]
        or decode_graphs(profile)
        or profile["context"]["max_num_seqs"] > 2
    ):
        # cc-defer: independent EP with up to two sequences; extend combinations
        # only after their resource and quality gates pass.
        raise ValueError(
            "EP requires eager, at most two sequences, no LPA/MTP/fusion/APC"
        )


def check_worker_exclusivity(profile):
    """One worker extension class per launch, each with its own constraints."""
    if profile["validation"]["component_worker"] and (
        profile["lpa"]["enabled"]
        or profile["mtp"]["enabled"]
        or profile["context"]["max_num_seqs"] != 1
        or profile["cache"]["prefix_caching"]
        or decode_graphs(profile)
    ):
        raise ValueError(
            "Component validation requires eager, one sequence, no LPA/MTP/prefix cache"
        )
    if profile["lpa"]["enabled"] and profile["context"]["max_num_seqs"] != 1:
        raise ValueError(
            "LPA requires max_num_seqs=1; use a separate no-LPA throughput profile"
        )
    if type(profile["validation"].get("memory_probe", False)) is not bool:
        raise ValueError("validation.memory_probe must be true or false")
    if profile["validation"].get("memory_probe") and (
        profile["lpa"]["enabled"]
        or profile["validation"]["component_worker"]
        or profile["validation"]["expert_worker"]
    ):
        # One worker extension class per launch; the others carry their own.
        raise ValueError("validation.memory_probe excludes LPA and the other workers")


def check_generation(profile):
    """Memory share, sampling, and the room a reply needs inside the context."""
    if not 0 < profile["cache"]["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if profile["generation"]["temperature"] < 0:
        raise ValueError("temperature must be nonnegative")
    if profile["generation"]["max_tokens"] >= profile["context"]["max_model_len"]:
        raise ValueError("Reserve context space for the input prompt")
    if (
        profile["generation"].get("warmup_long_tokens", 0)
        + profile["generation"]["max_tokens"]
        >= profile["context"]["max_model_len"]
    ):
        raise ValueError("warmup_long_tokens plus max_tokens must fit the context")
    if profile["generation"]["reasoning_effort"] not in {"low", "high", "max"}:
        raise ValueError(
            "Use a supported reasoning_effort; thinking-off is unqualified"
        )


def check_speculation(profile):
    """Draft depth, and the local view the draft weights are served from."""
    depth = profile["mtp"]["num_speculative_tokens"]
    if type(depth) is not int or not 1 <= depth <= 5:
        # 1 and 3 are measured; 2, 4 and 5 are launchable for the depth sweep
        # (the draft is one layer, run k times, so acceptance falls with depth).
        raise ValueError("MTP depth must be an integer from 1 to 5")
    view = PurePosixPath(profile["mtp"]["view"])
    if view.is_absolute() or ".." in view.parts or not view.parts or ":" in str(view):
        raise ValueError("mtp.view must be a relative path inside the HF cache")


def check_lpa(profile):
    """Where the approximation cuts, and the projector it is pinned to."""
    lpa = profile["lpa"]
    if (
        not 0 <= lpa["cut"] < MODEL_LAYERS
        or not 1 <= lpa["tail"] <= profile["context"]["max_model_len"]
        or lpa["break_even_tokens"] < 0
    ):
        raise ValueError("Invalid LPA cut or tail")
    if not re.fullmatch(r"[0-9a-f]{64}", lpa["projector_sha256"]):
        raise ValueError("Invalid projector_sha256")
    if lpa["enabled"] and decode_graphs(profile):
        raise ValueError("LPA requires eager execution")


def check_graph_scope(profile):
    """What the decode Graph path has been qualified to cover."""
    if decode_graphs(profile) and profile["context"]["max_num_seqs"] != 1:
        # cc-defer: one sequence only (MTP k=3 and prefix caching were qualified on
        # the MTP fixture, records/20260918-stage1-graph); extend to batching after
        # a fixture with max_num_seqs > 1 shows the same eager/graph identity.
        raise ValueError("Graph experiments require one sequence")


def check_identifiers(profile):
    """The two sites, and the names the API is served under."""
    for rank in (0, 1):
        host.validate_site(site(profile, rank))
    for key in ("served_model_name", "reasoning_parser", "tool_call_parser"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", profile["api"][key]):
            raise ValueError(f"Invalid api.{key}")


# Order is part of the contract: the first raise is the sentence the operator
# reads, so a profile with two faults must report the one it met first.
VALIDATORS = (
    check_schema,
    check_optional_shapes,
    check_pinned_identity,
    check_magnitudes,
    check_expert_observer,
    check_parallelism,
    check_worker_exclusivity,
    check_generation,
    check_speculation,
    check_lpa,
    check_graph_scope,
    check_identifiers,
)


def validate(profile):
    for check in VALIDATORS:
        check(profile)


def derived_checkpoint(profile):
    """The derived-checkpoint table when present and switched on, else None."""
    derived = profile["runtime"].get("derived_checkpoint")
    if derived and derived.get("enabled", True):
        return derived
    return None


def validate_derived(derived):
    """A locally requantized checkpoint and the source overlays it needs to boot."""
    name = "runtime.derived_checkpoint"
    if not isinstance(derived, dict) or derived.keys() - {"enabled"} != {
        "path",
        "requant_target",
        "overlays",
    }:
        raise ValueError(f"Unknown/missing settings in {name}")
    if type(derived.get("enabled", True)) is not bool:
        raise ValueError(f"{name}.enabled must be true or false")

    def absolute(value):
        return isinstance(value, str) and PurePosixPath(value).is_absolute()

    target = derived["requant_target"]
    if not absolute(derived["path"]) or not isinstance(target, str) or not target:
        raise ValueError(f"{name} needs an absolute path and a requant_target")
    overlays = derived["overlays"]
    if not isinstance(overlays, list) or not overlays:
        raise ValueError(f"{name}.overlays must list at least one file")
    for overlay in overlays:
        if not isinstance(overlay, dict) or overlay.keys() != {
            "target",
            "source",
            "sha256",
            "base_sha256",
            "marker",
        }:
            raise ValueError(f"Unknown/missing settings in {name}.overlays")
        if (
            not isinstance(overlay["target"], str)
            or not re.fullmatch(r"[a-z_]+\.py", overlay["target"])
            or not absolute(overlay["source"])
            or not isinstance(overlay["marker"], str)
            or not overlay["marker"]
            or any(
                not isinstance(overlay[key], str)
                or not re.fullmatch(r"[0-9a-f]{64}", overlay[key])
                for key in ("sha256", "base_sha256")
            )
        ):
            raise ValueError(f"Invalid entry in {name}.overlays")
    if len({overlay["target"] for overlay in overlays}) != len(overlays):
        raise ValueError(f"{name}.overlays names a target twice")


def site(profile, rank):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    return {
        **profile["nodes"][rank],
        "rank": rank,
        "head_ip": profile["nodes"][0]["local_ip"],
        "api_port": profile["api"]["port"],
        "master_port": profile["api"]["master_port"],
    }


def fingerprint(profile):
    value = {"settings": profile, "lock": load_lock()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def selected_image(profile):
    return profile["runtime"][
        "lpa_image" if profile["lpa"]["enabled"] else "reference_image"
    ]


def environment(profile, rank):
    result = host.fabric_env(site(profile, rank))
    result.update(
        NCCL_SOCKET_FAMILY="AF_INET",
        NVIDIA_TF32_OVERRIDE="0",
        VLLM_BATCH_INVARIANT="0",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
        # Their defaults sit outside the mounted /root/.cache, so every container
        # recompiled its kernels while serving and the host RAM spike stopped rank0.
        TRITON_CACHE_DIR="/root/.cache/triton",
        TILELANG_CACHE_DIR="/root/.cache/tilelang",
        TORCHINDUCTOR_CACHE_DIR="/root/.cache/torchinductor",
        # Triton keeps compiled kernels there but tunes again on every launch, and
        # the KDA inverse kernel's pick (num_warps 2 or 4) decides which of two
        # numerical states a launch computes in. Kept, the first pick stays.
        TRITON_CACHE_AUTOTUNING="1",
    )
    if "cuda_allocator_conf" in profile["runtime"]:
        result["PYTORCH_CUDA_ALLOC_CONF"] = profile["runtime"]["cuda_allocator_conf"]
    if "nccl_channels" in profile["runtime"]:
        # Pin both bounds so the two ranks cannot settle on different counts.
        channels = str(profile["runtime"]["nccl_channels"])
        result["NCCL_MIN_NCHANNELS"] = channels
        result["NCCL_MAX_NCHANNELS"] = channels
    if "fa2_attention" in profile["runtime"]:
        result["GLM53_FA2_ATTENTION"] = str(int(profile["runtime"]["fa2_attention"]))
    if "canonical_moe_order" in profile["runtime"]:
        # Absent: the image decides (on where the patch is installed).
        result["GLM53_CANONICAL_MOE_ORDER"] = str(
            int(profile["runtime"]["canonical_moe_order"])
        )
    if "stable_indexer_topk" in profile["runtime"]:
        # Absent: the image decides (on where the patch is installed).
        result["GLM53_STABLE_INDEXER_TOPK"] = str(
            int(profile["runtime"]["stable_indexer_topk"])
        )
    if "prefix_page_dedup" in profile["runtime"]:
        # Absent: off, as the pinned pool behaves.
        result["GLM53_PREFIX_PAGE_DEDUP"] = str(
            int(profile["runtime"]["prefix_page_dedup"])
        )
    if (
        profile["lpa"]["enabled"]
        or profile["validation"]["component_worker"]
        or profile["validation"]["expert_worker"]
        or profile["api"].get("dev_endpoints", False)
    ):
        result["VLLM_SERVER_DEV_MODE"] = "1"
    if profile["cache"]["fused_unpack"]:
        result["GLM53_FUSED_UNPACK"] = "1"
    if asynchronous_index_checks(profile):
        result["GLM53_ASYNC_INDEX_CHECKS"] = "1"
    if apc_lpa_enabled(profile):
        lpa = profile["lpa"]
        result["GLM53_APC_LPA_CONFIG"] = json.dumps(
            {
                "cut": lpa["cut"],
                "tail": lpa["tail"],
                "break_even": lpa["break_even_tokens"],
                "projector_path": "/lpa/projector.pt",
                "projector_sha256": lpa["projector_sha256"],
                "skip_mla_queries": lpa["skip_mla_queries"],
            },
            sort_keys=True,
        )
    if profile["runtime"]["pipeline_parallel_size"] == 2:
        split = profile["runtime"]["pipeline_split_layer"]
        result["VLLM_PP_LAYER_PARTITION"] = f"{split},{MODEL_LAYERS - split}"
    return result


def resolve_launch(profile, environ=None):
    """Freeze the launch-origin allocator override into the shared profile once."""
    env = os.environ if environ is None else environ
    resolved = copy.deepcopy(profile)
    if "PYTORCH_CUDA_ALLOC_CONF" in env:
        resolved["runtime"]["cuda_allocator_conf"] = env["PYTORCH_CUDA_ALLOC_CONF"]
    validate(resolved)
    return resolved


def apc_lpa_enabled(profile):
    return profile["lpa"]["enabled"] and profile["cache"]["prefix_caching"]


def asynchronous_index_checks(profile):
    runtime = profile["runtime"]
    return runtime["index_checks"] == "async" or (
        runtime["index_checks"] == "auto" and decode_graphs(profile)
    )


def apply_scalar_settings(args, profile):
    """Substitute the named values into the placeholders host.serve_args left."""
    values = {
        "--served-model-name": profile["api"]["served_model_name"],
        "--reasoning-parser": profile["api"]["reasoning_parser"],
        "--tool-call-parser": profile["api"]["tool_call_parser"],
        "--gpu-memory-utilization": profile["cache"]["gpu_memory_utilization"],
        **{
            "--" + key.replace("_", "-"): value
            for key, value in profile["context"].items()
            if key != "chunked_prefill"
        },
    }
    for flag, value in values.items():
        args[args.index(flag) + 1] = str(value)


def apply_boolean_flags(args, profile):
    """Drop the flags this profile turns off; chunked prefill states both ways."""
    for section, key, flag in [
        ("runtime", "decode_graphs", "--enforce-eager"),
        ("context", "chunked_prefill", "--enable-chunked-prefill"),
        ("api", "auto_tool_choice", "--enable-auto-tool-choice"),
    ]:
        enabled = (
            not decode_graphs(profile)
            if key == "decode_graphs"
            else profile[section][key]
        )
        if not enabled:
            args.remove(flag)
            if key == "chunked_prefill":
                args.append("--no-enable-chunked-prefill")


def apply_vision(args, profile):
    """Accept image input, and bound what the startup profile encodes."""
    if not profile["runtime"].get("vision", False):
        return
    args.remove("--language-model-only")
    # Images only. Startup profiling encodes the largest item once, and a
    # 30,000-token video would otherwise set that peak.
    args += ["--limit-mm-per-prompt", json.dumps({"video": 0})]
    # vLLM defaults to 4 GiB, duplicated in the head's API and engine
    # processes; this host keeps about 1 GiB above the memory reserve.
    size = profile["cache"].get("mm_processor_cache_gb", 0.1)
    args += ["--mm-processor-cache-gb", str(size)]


def apply_cache(args, profile):
    """Prefix caching, the KV budget, and how long a prefix is retained."""
    if profile["cache"]["prefix_caching"]:
        args[args.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
    if profile["api"].get("prompt_tokens_details"):
        args.append("--enable-prompt-tokens-details")
    for key in ("kv_cache_memory_bytes", "block_size"):
        args += ["--" + key.replace("_", "-"), str(profile["cache"][key])]
    if "prefix_cache_retention_interval" in profile["cache"]:
        args += [
            "--prefix-cache-retention-interval",
            str(retention_interval(profile)),
        ]


def apply_determinism(args, profile):
    """The seed and the kernel choices an identical request repeats under."""
    args += [
        "--seed",
        str(profile["runtime"]["seed"]),
        "--kernel-config",
        json.dumps(
            {
                "moe_backend": "marlin",
                "linear_backend": "marlin",
                "enable_flashinfer_autotune": False,
                "enable_cutedsl_warmup": False,
                "enable_jit_warmup": False,
            }
        ),
    ]


def apply_speculation(args, profile):
    """The MTP draft, when this profile serves the local view."""
    if not profile["mtp"]["enabled"]:
        return
    args += [
        "--speculative-config",
        json.dumps(
            {
                "method": "mtp",
                "num_speculative_tokens": profile["mtp"]["num_speculative_tokens"],
                "moe_backend": "triton",
            }
        ),
    ]


def apply_decode_graphs(args, profile):
    """Capture decode as a Graph, at the sizes the draft depth implies."""
    if not decode_graphs(profile):
        return
    args += [
        "--compilation-config",
        json.dumps(
            {
                "mode": 0,  # CompilationMode.NONE in the pinned runtime.
                "cudagraph_mode": "FULL_DECODE_ONLY",
                # The pinned runtime rounds decode sizes up to a multiple of
                # num_speculative_tokens + 1 and rejects a list with none.
                "cudagraph_capture_sizes": [
                    1 + profile["mtp"]["num_speculative_tokens"]
                    if profile["mtp"]["enabled"]
                    else 1
                ],
            }
        ),
    ]


def apply_worker_extension(args, profile):
    """The one worker class this launch carries; validate() keeps them apart."""
    for enabled, extension in (
        (profile["lpa"]["enabled"], "glm53_setup.runtime.lpa.LPAWorkerExtension"),
        (
            profile["validation"]["component_worker"],
            "glm53_setup.runtime.component_worker.ComponentWorker",
        ),
        (
            profile["validation"]["expert_worker"],
            "glm53_setup.validation.expert_worker.ExpertFixtureWorker",
        ),
        (
            profile["validation"].get("memory_probe"),
            "glm53_setup.runtime.memory_probe.MemoryProbeWorker",
        ),
    ):
        if enabled:
            args += ["--worker-extension-cls", extension]


def apply_profiling(args, profile):
    """On-demand Torch tracing into the record directory."""
    if not profile["profiling"]["enabled"]:
        return
    args += [
        "--profiler-config",
        json.dumps(
            {
                "profiler": "torch",
                "torch_profiler_dir": "/profiles",
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": False,
                "torch_profiler_with_memory": False,
                "torch_profiler_use_gzip": True,
                "ignore_frontend": True,
                "torch_profiler_dump_cuda_time_total": False,
            }
        ),
    ]


def apply_parallelism(args, profile):
    """Expert parallelism, and the PP2 shape that re-splits the two ranks."""
    if profile["runtime"]["expert_parallel"]:
        args.append("--enable-expert-parallel")
    if profile["runtime"]["pipeline_parallel_size"] == 2:
        args[args.index("--tensor-parallel-size") + 1] = "1"
        args += ["--pipeline-parallel-size", "2"]


# Order is part of the contract: these steps write into one argument list and
# several of them index into what the earlier steps left.
SERVE_STEPS = (
    apply_scalar_settings,
    apply_boolean_flags,
    apply_vision,
    apply_cache,
    apply_determinism,
    apply_speculation,
    apply_decode_graphs,
    apply_worker_extension,
    apply_profiling,
    apply_parallelism,
)


def serve_args(profile, rank, model_path):
    """Assemble a profile already validated by load() or server.command()."""
    args = host.serve_args(site(profile, rank), model_path)
    for step in SERVE_STEPS:
        step(args, profile)
    return args


def retention_interval(profile):
    value = profile["cache"].get("prefix_cache_retention_interval", 0)
    return None if value == "dense" else value


def request_body(profile, request):
    body = copy.deepcopy(request)
    if body.get("stream") or not body.get("messages"):
        raise ValueError("server ask requires messages and a non-streaming request")
    if (
        body.get("model", profile["api"]["served_model_name"])
        != profile["api"]["served_model_name"]
    ):
        raise ValueError("Request model does not match server profile")
    body["model"] = profile["api"]["served_model_name"]
    for key in ("temperature", "max_tokens", "reasoning_effort"):
        body.setdefault(key, profile["generation"][key])
    body.setdefault("seed", profile["runtime"]["seed"])
    template = body.setdefault("chat_template_kwargs", {})
    template.setdefault("reasoning_effort", body["reasoning_effort"])
    template.setdefault("clear_thinking", profile["generation"]["clear_thinking"])
    return body


def lpa_request(profile, length):
    lpa = profile["lpa"]
    return {
        "mode": "predict"
        if lpa["enabled"] and length - lpa["tail"] > lpa["break_even_tokens"]
        else "off",
        "cut": lpa["cut"],
        "prompt_length": length,
        "tail": min(lpa["tail"], length),
        "predictor_path": "/lpa/projector.pt",
        "skip_mla_queries": lpa["skip_mla_queries"],
        "allow_mtp": profile["mtp"]["enabled"],
    }
