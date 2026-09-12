"""Validate and measure graph capture of the unchanged FP32 NoPE arithmetic."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invalid-case", choices=("negative", "overflow"))
    args = parser.parse_args(argv)
    import torch

    from glm53_setup.runtime.reference_attention import sparse_nope_reference
    from glm53_setup.validation.profile_trace import read_trace, summarize_trace

    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["GLM53_FUSED_UNPACK"] = "1"
    report = {
        "scope": "NoPE component only; graph safety with dynamic tensor contents",
        "device": torch.cuda.get_device_name(),
        "cases": [],
    }
    torch.manual_seed(71)
    for tokens in (1, 8):
        packed = torch.zeros((256, 656), dtype=torch.uint8, device="cuda")
        packed[:, :512] = (
            torch.randn((256, 512), device="cuda")
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        packed[:, 512:528] = (
            torch.tensor([0.5, 1.0, 1.5, 2.0], device="cuda")
            .repeat(256, 1)
            .view(torch.uint8)
        )
        query = torch.randn((tokens, 32, 512), dtype=torch.bfloat16, device="cuda")
        indices = torch.randint(
            0, 256, (tokens, 2176), dtype=torch.int32, device="cuda"
        )
        original_indices = indices.clone()

        def compute():
            return sparse_nope_reference(query, packed, indices, 512**-0.5)

        os.environ["GLM53_ASYNC_INDEX_CHECKS"] = "0"
        expected = compute()
        os.environ["GLM53_ASYNC_INDEX_CHECKS"] = "1"
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compute()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = compute()
        if args.invalid_case:
            indices[0, 0] = -2 if args.invalid_case == "negative" else packed.shape[0]
            graph.replay()
            torch.cuda.synchronize()
            raise RuntimeError("Invalid index was not rejected")
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        indices.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, torch.zeros_like(captured), rtol=0, atol=0)
        indices.copy_(original_indices)
        query.mul_(0.75)
        packed[:, 512:528] = (
            torch.tensor([0.25, 0.5, 0.75, 1.0], device="cuda")
            .repeat(256, 1)
            .view(torch.uint8)
        )
        expected = compute()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        case = {
            "query_tokens": tokens,
            "exact": True,
            "changed_inputs_and_empty_rows": True,
            "paths": {},
        }
        for name, mode, fn in (
            ("synchronous_eager", "0", compute),
            ("asynchronous_eager", "1", compute),
            ("graph", "1", graph.replay),
        ):
            os.environ["GLM53_ASYNC_INDEX_CHECKS"] = mode
            for _ in range(5):
                fn()
            samples = []
            for _ in range(5):
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                began = time.perf_counter()
                start.record()
                for _ in range(50):
                    fn()
                end.record()
                end.synchronize()
                samples.append(
                    {
                        "cuda_ms": start.elapsed_time(end) / 50,
                        "wall_ms": (time.perf_counter() - began) * 1000 / 50,
                    }
                )
            trace = args.output / f"{tokens}-{name}.json"
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                fn()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(trace))
            case["paths"][name] = {
                "samples": samples,
                "median_cuda_ms": statistics.median(s["cuda_ms"] for s in samples),
                "median_wall_ms": statistics.median(s["wall_ms"] for s in samples),
                "profile": summarize_trace(read_trace(trace)),
            }
        report["cases"].append(case)
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            tokens,
            {
                name: {
                    "ms": v["median_cuda_ms"],
                    "launches": v["profile"]["launch_api_events"],
                    "sync": v["profile"]["synchronization_api_events"],
                }
                for name, v in case["paths"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
