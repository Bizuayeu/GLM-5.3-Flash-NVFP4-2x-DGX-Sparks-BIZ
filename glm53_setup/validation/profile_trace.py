"""Count CUDA kernel/launch events without confusing their sum with wall time."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def summarize_trace(payload):
    events = payload.get("traceEvents", [])
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    if not kernels:
        raise ValueError(
            "No GPU kernel events; a CPU-only or empty trace is not evidence"
        )
    launches = [
        e
        for e in events
        if e.get("cat") in {"cuda_runtime", "cuda_driver"}
        and "launch" in e.get("name", "").lower()
    ]
    communication = [e for e in kernels if "nccl" in e.get("name", "").lower()]
    return {
        "kernel_events": len(kernels),
        "launch_api_events": len(launches),
        "nccl_kernel_events": len(communication),
        "summed_kernel_duration_us": sum(e.get("dur", 0) for e in kernels),
        "summed_nccl_duration_us": sum(e.get("dur", 0) for e in communication),
        "summed_launch_api_duration_us": sum(e.get("dur", 0) for e in launches),
        "kernel_names": dict(Counter(e["name"] for e in kernels)),
        "duration_semantics": "summed events may overlap; not wall-clock latency",
    }


def decode_delta(measured, control, output_tokens, control_tokens):
    if not 0 < control_tokens < output_tokens:
        raise ValueError(
            "Measured output must exceed the positive prefill-control output"
        )
    additional = output_tokens - control_tokens
    return {
        "additional_output_tokens": additional,
        "kernel_event_delta_per_token": (
            measured["kernel_events"] - control["kernel_events"]
        )
        / additional,
        "nccl_event_delta_per_token": (
            measured["nccl_kernel_events"] - control["nccl_kernel_events"]
        )
        / additional,
        "interpretation": "paired estimate; same prompt/profile/warmup required, MTP acceptance may change step counts",
    }


def read_trace(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--prefill-control", type=Path)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--control-tokens", type=int)
    args = parser.parse_args(argv)
    result = summarize_trace(read_trace(args.trace))
    if args.prefill_control:
        if args.output_tokens is None or args.control_tokens is None:
            parser.error("Supply both measured token counts for the paired estimate")
        control = summarize_trace(read_trace(args.prefill_control))
        result["decode_delta"] = decode_delta(
            result, control, args.output_tokens, args.control_tokens
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
