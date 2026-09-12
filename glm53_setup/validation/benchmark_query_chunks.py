"""Measure reference query batching independently of scheduler chunks/fusion."""

import argparse
import os
import statistics
import time
from pathlib import Path

from glm53_setup.io import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fused-attention", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(GLM53_FUSED_UNPACK="0", GLM53_ASYNC_INDEX_CHECKS="0")
    import torch

    from glm53_setup.runtime.reference_attention import sparse_nope_reference
    from glm53_setup.validation.profile_trace import read_trace, summarize_trace

    torch.manual_seed(42)
    if args.fused_attention:
        from glm53_setup.runtime.fused_nope import fused_nope_attention
    torch.backends.cuda.matmul.allow_tf32 = False
    packed = torch.zeros((4096, 656), dtype=torch.uint8, device="cuda")
    packed[:, :512] = (
        torch.randn((4096, 512), device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    packed[:, 512:528] = torch.rand((4096, 4), device="cuda").view(torch.uint8)
    report = {
        "status": "running",
        "scope": "FP32 reference query chunks; no scheduler/model changes",
        "cases": [],
        "fused_attention_candidate": args.fused_attention,
        "masked_queries_in_timing": not args.fused_attention,
    }
    try:
        for tokens in (1, 8, 65, 512) if args.fused_attention else (9, 65, 128, 512):
            query = torch.randn((tokens, 32, 512), dtype=torch.bfloat16, device="cuda")
            indices = torch.randint(
                0, 4096, (tokens, 2176), dtype=torch.int32, device="cuda"
            )
            if not args.fused_attention:
                indices[0] = -1
                indices[1, :-1] = -1
                indices[2, ::2] = -1
            expected = sparse_nope_reference(query, packed, indices, 512**-0.5)
            case = {"query_tokens": tokens, "candidates": 2176, "paths": {}}
            report["cases"].append(case)
            for chunk in (8, "fused") if args.fused_attention else (8, 32, 64):

                def call():
                    if chunk == "fused":
                        return fused_nope_attention(query, packed, indices, 512**-0.5)
                    return sparse_nope_reference(
                        query, packed, indices, 512**-0.5, query_chunk=chunk
                    )

                actual = call()
                error = (actual.float() - expected.float()).abs().max().item()
                tolerance = (
                    2
                    * torch.finfo(torch.bfloat16).eps
                    * max(1, expected.abs().max().item())
                )
                row = {
                    "max_abs_error": error,
                    "tolerance": tolerance,
                    "exact": torch.equal(actual, expected),
                    "empty_row_zero": None
                    if args.fused_attention
                    else bool((actual[0] == 0).all()),
                    "finite": bool(torch.isfinite(actual).all()),
                }
                case["paths"][str(chunk)] = row
                write_json(args.output / "result.json", report)
                if (
                    not row["finite"]
                    or row["empty_row_zero"] is False
                    or error > tolerance
                ):
                    raise ValueError("Query chunk numerical check failed")
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                call()
                torch.cuda.synchronize()
                row["extra_peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated() - baseline
                )
                samples = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    for _ in range(10):
                        call()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - began) * 1000 / 10)
                row.update(wall_ms=samples, median_wall_ms=statistics.median(samples))
                trace = args.output / f"q{tokens}-chunk{chunk}.json"
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    call()
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(str(trace))
                row["profile"] = summarize_trace(read_trace(trace))
                write_json(args.output / "result.json", report)
                print(
                    tokens,
                    chunk,
                    row["median_wall_ms"],
                    row["extra_peak_allocated_bytes"],
                    row["exact"],
                    flush=True,
                )
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
