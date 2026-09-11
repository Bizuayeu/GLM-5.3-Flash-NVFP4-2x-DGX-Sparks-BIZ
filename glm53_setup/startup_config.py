"""Typed operator settings shared by experimental launch and chat requests."""

import copy
import hashlib
import json
import math
import re
import tomllib
from pathlib import Path, PurePosixPath

from . import service
from .config import ROOT, load_lock


def load(path):
    with Path(path).open("rb") as stream:
        profile = tomllib.load(stream)
    # Accept the initial spelling while operators migrate the same live profile.
    for table, old, new in [
        (profile, "llkv", "allkv"),
        (profile.get("runtime", {}), "llkv_image", "allkv_image"),
    ]:
        if old in table:
            if new in table:
                raise ValueError(f"Use only {new}, not both {old} and {new}")
            table[new] = table.pop(old)
    validate(profile)
    return profile


def validate(profile):
    # The shipped, commented file also defines the complete schema. No silent
    # defaults: a typo or missing category must not silently change a launch.
    with (ROOT / "examples/startup.example.toml").open("rb") as stream:
        schema = tomllib.load(stream)

    def check(value, expected, path):
        if isinstance(expected, dict):
            if not isinstance(value, dict) or value.keys() != expected.keys():
                raise ValueError(f"Unknown/missing settings in {path}")
            for key, item in expected.items():
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

    check(profile, schema, "startup")
    if profile["schema_version"] != 1:
        raise ValueError("Unsupported startup schema_version")
    for key in ("reference_image", "allkv_image"):
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
            "run_seconds",
        ),
    }.items():
        for key in keys:
            if profile[section][key] < 1:
                raise ValueError(f"{section}.{key} must be positive")
    if profile["context"]["max_num_seqs"] != 1:
        raise ValueError("This experimental launcher supports max_num_seqs=1")
    if not 0 < profile["cache"]["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if profile["generation"]["temperature"] < 0:
        raise ValueError("temperature must be nonnegative")
    if profile["generation"]["max_tokens"] >= profile["context"]["max_model_len"]:
        raise ValueError("Reserve context space for the input prompt")
    if profile["generation"]["reasoning_effort"] not in {"low", "medium", "high"}:
        raise ValueError(
            "Use a supported reasoning_effort; thinking-off is unqualified"
        )
    if profile["mtp"]["num_speculative_tokens"] not in (1, 3):
        raise ValueError("Only MTP depths 1 and 3 have experimental coverage")
    view = PurePosixPath(profile["mtp"]["view"])
    if view.is_absolute() or ".." in view.parts or not view.parts or ":" in str(view):
        raise ValueError("mtp.view must be a relative path inside the HF cache")
    llkv = profile["allkv"]
    if (
        not 0 <= llkv["cut"] < 45
        or not 1 <= llkv["tail"] <= profile["context"]["max_model_len"]
    ):
        raise ValueError("Invalid LLKV cut or tail")
    if not re.fullmatch(r"[0-9a-f]{64}", llkv["projector_sha256"]):
        raise ValueError("Invalid projector_sha256")
    if llkv["enabled"] and (
        profile["cache"]["prefix_caching"] or not profile["runtime"]["enforce_eager"]
    ):
        raise ValueError("LLKV requires eager execution and prefix caching disabled")
    for rank in (0, 1):
        service.validate_site(site(profile, rank))
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
    # Keep schema-v1 fingerprints stable across the user-facing aLLKV rename.
    canonical = copy.deepcopy(profile)
    canonical["llkv"] = canonical.pop("allkv")
    canonical["runtime"]["llkv_image"] = canonical["runtime"].pop("allkv_image")
    value = {"settings": canonical, "lock": load_lock()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def selected_image(profile):
    return profile["runtime"][
        "allkv_image" if profile["allkv"]["enabled"] else "reference_image"
    ]


def environment(profile, rank):
    result = service.fabric_env(site(profile, rank))
    result.update(
        NCCL_SOCKET_FAMILY="AF_INET",
        NVIDIA_TF32_OVERRIDE="0",
        VLLM_BATCH_INVARIANT="0",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
    )
    if profile["allkv"]["enabled"]:
        result["VLLM_SERVER_DEV_MODE"] = "1"
    return result


def serve_args(profile, rank, model_path):
    validate(profile)
    args = service.serve_args(site(profile, rank), model_path)
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
    if profile["cache"]["prefix_caching"]:
        args[args.index("--no-enable-prefix-caching")] = "--enable-prefix-caching"
    for key in ("kv_cache_memory_bytes", "block_size"):
        args += ["--" + key.replace("_", "-"), str(profile["cache"][key])]
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
    if profile["allkv"]["enabled"]:
        args += [
            "--worker-extension-cls",
            "glm53_setup.runtime.llkv.LLKVWorkerExtension",
        ]
    return args


def request_body(profile, request):
    body = copy.deepcopy(request)
    if body.get("stream") or not body.get("messages"):
        raise ValueError("startup ask requires messages and a non-streaming request")
    if (
        body.get("model", profile["api"]["served_model_name"])
        != profile["api"]["served_model_name"]
    ):
        raise ValueError("Request model does not match startup profile")
    body["model"] = profile["api"]["served_model_name"]
    for key in ("temperature", "max_tokens", "reasoning_effort"):
        body.setdefault(key, profile["generation"][key])
    body.setdefault("seed", profile["runtime"]["seed"])
    template = body.setdefault("chat_template_kwargs", {})
    template.setdefault("reasoning_effort", body["reasoning_effort"])
    template.setdefault("clear_thinking", profile["generation"]["clear_thinking"])
    return body


def llkv_request(profile, length):
    llkv = profile["allkv"]
    return {
        "mode": "predict" if llkv["enabled"] and length > llkv["tail"] else "off",
        "cut": llkv["cut"],
        "prompt_length": length,
        "tail": min(llkv["tail"], length),
        "predictor_path": "/llkv/projector.pt",
        "skip_mla_queries": llkv["skip_mla_queries"],
        "allow_mtp": profile["mtp"]["enabled"],
    }
