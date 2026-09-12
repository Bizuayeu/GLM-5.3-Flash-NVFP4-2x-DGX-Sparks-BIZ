"""Exclusive GPU component comparison; no model weights or LPA/MTP required."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("iterations must be positive")
    import torch

    from glm53_setup.runtime.fused_unpack import unpack_latent_cuda
    from glm53_setup.runtime.reference_attention import unpack_latent
    from glm53_setup.validation.profile_trace import read_trace, summarize_trace

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU is not a benchmark substitute")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["GLM53_FUSED_UNPACK"] = "0"
    report = {
        "scope": "unpack component only; not model speedup",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "iterations": args.iterations,
        "cases": [],
    }
    for rows in (1, 2176, 8 * 2176):
        packed = torch.zeros((rows, 656), device="cuda", dtype=torch.uint8)
        packed[:, :512] = (torch.arange(512, device="cuda") % 127).to(torch.uint8)
        packed[:, 512:528] = (
            torch.tensor([0.5, 1.0, 1.5, 2.0], device="cuda", dtype=torch.float32)
            .repeat(rows, 1)
            .view(torch.uint8)
        )
        torch.testing.assert_close(
            unpack_latent_cuda(packed), unpack_latent(packed), rtol=0, atol=0
        )
        case = {"rows": rows, "exact": True, "paths": {}}
        for name, fn in (("reference", unpack_latent), ("fused", unpack_latent_cuda)):
            for _ in range(10):
                fn(packed)
            samples = []
            for _ in range(5):
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                began = time.perf_counter()
                start.record()
                for _ in range(args.iterations):
                    fn(packed)
                end.record()
                end.synchronize()
                samples.append(
                    {
                        "cuda_ms_per_call": start.elapsed_time(end) / args.iterations,
                        "wall_ms_per_call": (time.perf_counter() - began)
                        * 1000
                        / args.iterations,
                    }
                )
            trace = args.output / f"{rows}-{name}.json"
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                fn(packed)
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(trace))
            case["paths"][name] = {
                "samples": samples,
                "median_cuda_ms": statistics.median(
                    s["cuda_ms_per_call"] for s in samples
                ),
                "median_wall_ms": statistics.median(
                    s["wall_ms_per_call"] for s in samples
                ),
                "profile": summarize_trace(read_trace(trace)),
            }
        report["cases"].append(case)
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            rows,
            {
                k: {"ms": v["median_cuda_ms"], "kernels": v["profile"]["kernel_events"]}
                for k, v in case["paths"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
