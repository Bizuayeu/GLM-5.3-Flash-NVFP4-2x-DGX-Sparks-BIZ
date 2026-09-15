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


def validate(profile):
    # The shipped, commented file also defines the complete schema. No silent
    # defaults: a typo or missing category must not silently change a launch.
    with (ROOT / "examples/server.example.toml").open("rb") as stream:
        schema = tomllib.load(stream)

    def check(value, expected, path):
        if isinstance(expected, dict):
            optional = {
                "server.runtime": {"cuda_allocator_conf", "vision"},
                "server.cache": {
                    "prefix_cache_retention_interval",
                    "mm_processor_cache_gb",
                },
                "server.api": {"prompt_tokens_details"},
            }.get(path, set())
            if path.startswith("server.nodes["):
                optional = {"additional_rails"}
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
    if type(profile["runtime"].get("vision", False)) is not bool:
        raise ValueError("runtime.vision must be true or false")
    if "mm_processor_cache_gb" in profile["cache"]:
        size = profile["cache"]["mm_processor_cache_gb"]
        if type(size) not in (int, float) or not math.isfinite(size) or size < 0:
            raise ValueError(
                "cache.mm_processor_cache_gb must be a finite nonnegative number"
            )
    if profile["schema_version"] != 1:
        raise ValueError("Unsupported profile schema_version")
    for key in ("reference_image", "lpa_image"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", profile["runtime"][key]):
            raise ValueError(f"runtime.{key} must be an immutable image ID")
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
    if profile["validation"]["expert_worker"] and (
        profile["validation"]["component_worker"]
        or profile["runtime"]["pipeline_parallel_size"] != 1
        or profile["lpa"]["enabled"]
        or profile["mtp"]["enabled"]
        or profile["cache"]["prefix_caching"]
        or not profile["runtime"]["enforce_eager"]
        or profile["context"]["max_num_seqs"] > 2
    ):
        raise ValueError(
            "EP observer requires independent eager TP2, no other worker/MTP/APC"
        )
    runtime = profile["runtime"]
    if runtime["index_checks"] not in ("auto", "sync", "async"):
        raise ValueError(
            "index_checks must be auto, sync or async; checks cannot be disabled"
        )
    if not runtime["enforce_eager"] and runtime["index_checks"] == "sync":
        raise ValueError("Graph execution requires asynchronous index checks")
    if runtime["pipeline_parallel_size"] not in (1, 2):
        raise ValueError("Only PP sizes 1 and 2 are supported")
    split = runtime["pipeline_split_layer"]
    if not 4 <= split <= MODEL_LAYERS - 2:
        raise ValueError("PP stage boundaries must retain an MLA layer in both stages")
    if runtime["pipeline_parallel_size"] == 2 and (
        runtime["expert_parallel"]
        or not runtime["enforce_eager"]
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
        or not profile["runtime"]["enforce_eager"]
        or profile["context"]["max_num_seqs"] > 2
    ):
        # cc-defer: independent EP with up to two sequences; extend combinations
        # only after their resource and quality gates pass.
        raise ValueError(
            "EP requires eager, at most two sequences, no LPA/MTP/fusion/APC"
        )
    if profile["validation"]["component_worker"] and (
        profile["lpa"]["enabled"]
        or profile["mtp"]["enabled"]
        or profile["context"]["max_num_seqs"] != 1
        or profile["cache"]["prefix_caching"]
        or not profile["runtime"]["enforce_eager"]
    ):
        raise ValueError(
            "Component validation requires eager, one sequence, no LPA/MTP/prefix cache"
        )
    if profile["lpa"]["enabled"] and profile["context"]["max_num_seqs"] != 1:
        raise ValueError(
            "LPA requires max_num_seqs=1; use a separate no-LPA throughput profile"
        )
    if not 0 < profile["cache"]["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if profile["generation"]["temperature"] < 0:
        raise ValueError("temperature must be nonnegative")
    if profile["generation"]["max_tokens"] >= profile["context"]["max_model_len"]:
        raise ValueError("Reserve context space for the input prompt")
    if profile["generation"]["reasoning_effort"] not in {"low", "high", "max"}:
        raise ValueError(
            "Use a supported reasoning_effort; thinking-off is unqualified"
        )
    if profile["mtp"]["num_speculative_tokens"] not in (1, 3):
        raise ValueError("Only MTP depths 1 and 3 have experimental coverage")
    view = PurePosixPath(profile["mtp"]["view"])
    if view.is_absolute() or ".." in view.parts or not view.parts or ":" in str(view):
        raise ValueError("mtp.view must be a relative path inside the HF cache")
    lpa = profile["lpa"]
    if (
        not 0 <= lpa["cut"] < MODEL_LAYERS
        or not 1 <= lpa["tail"] <= profile["context"]["max_model_len"]
        or lpa["break_even_tokens"] < 0
    ):
        raise ValueError("Invalid LPA cut or tail")
    if not re.fullmatch(r"[0-9a-f]{64}", lpa["projector_sha256"]):
        raise ValueError("Invalid projector_sha256")
    if lpa["enabled"] and not profile["runtime"]["enforce_eager"]:
        raise ValueError("LPA requires eager execution")
    if not profile["runtime"]["enforce_eager"] and (
        profile["mtp"]["enabled"]
        or profile["cache"]["prefix_caching"]
        or profile["context"]["max_num_seqs"] != 1
    ):
        # cc-defer: independent serial target Graph evaluation only; extend after
        # the corresponding MTP/APC/batching integration is qualified.
        raise ValueError("Graph experiments require one sequence, no MTP/prefix cache")
    for rank in (0, 1):
        host.validate_site(site(profile, rank))
    for key in ("served_model_name", "reasoning_parser", "tool_call_parser"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", profile["api"][key]):
            raise ValueError(f"Invalid api.{key}")


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
    )
    if "cuda_allocator_conf" in profile["runtime"]:
        result["PYTORCH_CUDA_ALLOC_CONF"] = profile["runtime"]["cuda_allocator_conf"]
    if (
        profile["lpa"]["enabled"]
        or profile["validation"]["component_worker"]
        or profile["validation"]["expert_worker"]
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
        runtime["index_checks"] == "auto" and not runtime["enforce_eager"]
    )


def serve_args(profile, rank, model_path):
    """Assemble a profile already validated by load() or server.command()."""
    args = host.serve_args(site(profile, rank), model_path)
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
    for section, key, flag in [
        ("runtime", "enforce_eager", "--enforce-eager"),
        ("context", "chunked_prefill", "--enable-chunked-prefill"),
        ("api", "auto_tool_choice", "--enable-auto-tool-choice"),
    ]:
        if not profile[section][key]:
            args.remove(flag)
            if key == "chunked_prefill":
                args.append("--no-enable-chunked-prefill")
    if profile["runtime"].get("vision", False):
        args.remove("--language-model-only")
        # Images only. Startup profiling encodes the largest item once, and a
        # 30,000-token video would otherwise set that peak.
        args += ["--limit-mm-per-prompt", json.dumps({"video": 0})]
        # vLLM defaults to 4 GiB, duplicated in the head's API and engine
        # processes; this host keeps about 1 GiB above the memory reserve.
        size = profile["cache"].get("mm_processor_cache_gb", 0.1)
        args += ["--mm-processor-cache-gb", str(size)]
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
    if profile["mtp"]["enabled"]:
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
    if not profile["runtime"]["enforce_eager"]:
        args += [
            "--compilation-config",
            json.dumps(
                {
                    "mode": 0,  # CompilationMode.NONE in the pinned runtime.
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [1],
                }
            ),
        ]
    if profile["lpa"]["enabled"]:
        args += [
            "--worker-extension-cls",
            "glm53_setup.runtime.lpa.LPAWorkerExtension",
        ]
    if profile["validation"]["component_worker"]:
        args += [
            "--worker-extension-cls",
            "glm53_setup.runtime.component_worker.ComponentWorker",
        ]
    if profile["validation"]["expert_worker"]:
        args += [
            "--worker-extension-cls",
            "glm53_setup.validation.expert_worker.ExpertFixtureWorker",
        ]
    if profile["profiling"]["enabled"]:
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
    if profile["runtime"]["expert_parallel"]:
        args.append("--enable-expert-parallel")
    if profile["runtime"]["pipeline_parallel_size"] == 2:
        args[args.index("--tensor-parallel-size") + 1] = "1"
        args += ["--pipeline-parallel-size", "2"]
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
