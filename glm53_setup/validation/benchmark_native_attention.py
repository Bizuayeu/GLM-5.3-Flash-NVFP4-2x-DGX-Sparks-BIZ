"""Probe padded native sparse MLA without changing the serving backend."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from glm53_setup.io import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(GLM53_FUSED_UNPACK="0", GLM53_ASYNC_INDEX_CHECKS="0")
    import torch
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    from vllm import _custom_ops as ops

    from glm53_setup.runtime.reference_attention import sparse_nope_reference
    from glm53_setup.validation.profile_trace import read_trace, summarize_trace

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 2112  # Same cache and tail boundary as the established reference check.
    latent = torch.randn(n, 512, dtype=torch.bfloat16, device="cuda")
    rope = torch.zeros(n, 64, dtype=torch.bfloat16, device="cuda")
    packed = torch.zeros(n // 64, 64, 656, dtype=torch.uint8, device="cuda")
    slots = torch.arange(n, dtype=torch.int64, device="cuda")
    scale = 512**-0.5
    report = {
        "status": "running",
        "scope": "padded native attention component; no full model or Graphs",
        "device": torch.cuda.get_device_name(),
        "cases": [],
        "timings": [],
    }

    def save():
        write_json(args.output / "result.json", report)

    def pack():
        ops.concat_and_cache_mla(
            latent, rope, packed, slots, "fp8_ds_mla", torch.ones(1, device="cuda")
        )

    def native_call(query, indices):
        tokens, heads, _ = query.shape
        width = ((indices.shape[-1] + 127) // 128) * 128
        table = torch.nn.functional.pad(
            indices, (0, width - indices.shape[-1]), value=-1
        )
        # Fixed FlashInfer decode scratch: BF16 partial outputs + FP32 LSE,
        # split every 64 candidates. Prefill (>64 queries) does not use it.
        workspace_bytes = (
            tokens * heads * (width // 64) * (512 * 2 + 4) if tokens <= 64 else 1
        )
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")

        def call():
            return trtllm_batch_decode_with_kv_cache_mla(
                query=torch.nn.functional.pad(query, (0, 64)).unsqueeze(1),
                kv_cache=packed.unsqueeze(1),
                workspace_buffer=workspace,
                qk_nope_head_dim=256,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=table.unsqueeze(1),
                seq_lens=None,
                max_seq_len=width,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                sparse_mla_top_k=width,
                kv_scale_format="arbitrary_fp32",
                backend="sparse",
            ).squeeze(1)

        return call

    save()
    try:
        pack()
        for heads in (32, 64):
            for width in (63, 64, 65, 2048, 2051, 2176):
                query = torch.randn(3, heads, 512, dtype=torch.bfloat16, device="cuda")
                indices = torch.full((3, width), -1, dtype=torch.int32, device="cuda")
                count = min(width, n)
                indices[0, :count] = torch.arange(
                    count, dtype=torch.int32, device="cuda"
                )
                indices[1, :17] = torch.arange(17, dtype=torch.int32, device="cuda")
                expected = sparse_nope_reference(query, packed, indices, scale).float()
                actual = native_call(query, indices)().float()
                torch.cuda.synchronize()
                tolerance = (
                    2
                    * torch.finfo(torch.bfloat16).eps
                    * max(1.0, expected.abs().max().item())
                )
                error = (actual - expected).abs().max().item()
                case = {
                    "heads": heads,
                    "width": width,
                    "max_abs_error": error,
                    "tolerance": tolerance,
                    "finite": bool(torch.isfinite(actual).all()),
                    "empty_row_zero": bool((actual[2] == 0).all()),
                }
                case["passed"] = (
                    case["finite"] and error <= tolerance and case["empty_row_zero"]
                )
                report["cases"].append(case)
                save()
                print(json.dumps(case), flush=True)
        latent.zero_()
        latent[2050, 0] = 16
        pack()
        query = torch.zeros(1, 32, 512, dtype=torch.bfloat16, device="cuda")
        query[..., 0] = 16
        indices = torch.arange(2051, dtype=torch.int32, device="cuda").unsqueeze(0)
        expected = sparse_nope_reference(query, packed, indices, scale).float()
        actual = native_call(query, indices)().float()
        dropped = sparse_nope_reference(query, packed, indices[:, :2048], scale).float()
        tolerance = (
            2 * torch.finfo(torch.bfloat16).eps * max(1.0, expected.abs().max().item())
        )
        report["tail"] = {
            "native_error": (actual - expected).abs().max().item(),
            "omission_difference": (expected - dropped).abs().max().item(),
            "tolerance": tolerance,
        }
        save()
        if not all(c["passed"] for c in report["cases"]) or not (
            report["tail"]["native_error"] <= tolerance
            and report["tail"]["omission_difference"] > 1
        ):
            raise ValueError(
                "Native candidate failed numerical or candidate-preservation checks"
            )
        latent.normal_()
        pack()
        for tokens in (1, 8, 65):
            query = torch.randn(tokens, 32, 512, dtype=torch.bfloat16, device="cuda")
            indices = torch.arange(2176, dtype=torch.int32, device="cuda").repeat(
                tokens, 1
            )
            indices[indices >= n] = -1
            native = native_call(query, indices)

            def reference():
                return sparse_nope_reference(query, packed, indices, scale)

            expected = reference().float()
            actual = native().float()
            tolerance = (
                2
                * torch.finfo(torch.bfloat16).eps
                * max(1.0, expected.abs().max().item())
            )
            if (
                not bool(torch.isfinite(actual).all())
                or (actual - expected).abs().max().item() > tolerance
            ):
                raise ValueError(f"Native dispatch parity failed at {tokens} queries")
            row = {"queries": tokens, "heads": 32, "paths": {}}
            for label, call in (("reference", reference), ("native", native)):
                for _ in range(3):
                    call()
                samples = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    for _ in range(20):
                        call()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - began) * 1000 / 20)
                trace = args.output / f"{tokens}-{label}.json"
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    call()
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(str(trace))
                row["paths"][label] = {
                    "wall_ms": samples,
                    "median_wall_ms": statistics.median(samples),
                    "profile": summarize_trace(read_trace(trace)),
                }
            report["timings"].append(row)
            save()
            print(tokens, "timing complete", flush=True)
        report.update(
            status="complete",
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        save()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
