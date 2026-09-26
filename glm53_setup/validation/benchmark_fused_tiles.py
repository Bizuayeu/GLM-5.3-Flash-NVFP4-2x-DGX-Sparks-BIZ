"""Bounded register/tiling study for the FP32 fused NoPE component."""

import argparse
import os
import statistics
import time
from pathlib import Path

from glm53_setup.io import write_json
from glm53_setup.validation.parity import bf16_bound


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(GLM53_FUSED_UNPACK="0", GLM53_ASYNC_INDEX_CHECKS="0")
    import torch

    from glm53_setup.runtime.reference_attention import sparse_nope_reference
    from glm53_setup.validation.fused_nope import fused_nope_attention

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = torch.zeros((4096, 656), dtype=torch.uint8, device="cuda")
    cache[:, :512] = (
        torch.randn((4096, 512), device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    cache[:, 512:528] = torch.rand((4096, 4), device="cuda").view(torch.uint8)
    report = {"status": "running", "scope": "component tile tuning only", "cases": []}
    try:
        for tokens in (1, 512):
            query = torch.randn((tokens, 32, 512), dtype=torch.bfloat16, device="cuda")
            indices = torch.randint(
                0, 4096, (tokens, 2176), device="cuda", dtype=torch.int32
            )
            expected = sparse_nope_reference(query, cache, indices, 512**-0.5)
            for tile, warps in ((8, 4), (16, 4), (16, 8), (32, 8)):
                diagnostics = {}
                actual = fused_nope_attention(
                    query,
                    cache,
                    indices,
                    512**-0.5,
                    tile=tile,
                    warps=warps,
                    diagnostics=diagnostics,
                )
                error = (actual.float() - expected.float()).abs().max().item()
                tolerance = bf16_bound(expected.abs().max().item())
                if not bool(torch.isfinite(actual).all()) or error > tolerance:
                    raise ValueError("Tuned kernel parity failed")

                def call():
                    return fused_nope_attention(
                        query, cache, indices, 512**-0.5, tile=tile, warps=warps
                    )

                for _ in range(3):
                    call()
                samples = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    for _ in range(10):
                        call()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - began) * 1000 / 10)
                row = {
                    "queries": tokens,
                    "tile": tile,
                    "warps": warps,
                    "max_abs_error": error,
                    "tolerance": tolerance,
                    "kernel": diagnostics,
                    "wall_ms": samples,
                    "median_wall_ms": statistics.median(samples),
                }
                report["cases"].append(row)
                write_json(args.output / "result.json", report)
                print(
                    tokens, tile, warps, row["median_wall_ms"], diagnostics, flush=True
                )
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
