"""Probe padded native sparse MLA without changing the serving backend.

Without options this is the v15 probe. `--by-row-kind` revisits it at the
2048-candidate width the SM120 GLM decode table accepts: errors split by row
kind, empty rows passed as-is or pointed at slot 0 and zeroed afterwards, the
output change from dropping the lowest-ranked pool (2051 to 2047 candidates) and
timing. The revisit gathers evidence only; it never qualifies 2047 candidates.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

from glm53_setup.io import write_json
from glm53_setup.validation.parity import bf16_bound, judge, judge_tail

REVISIT_WIDTH = 2048  # The widest SM120 v32/GLM decode shape in FlashInfer 0.6.18.
KPOOL = 4  # Tokens per indexer pool; 512 pools fill the 2048-token top-k.
TAIL = (2048, 2049, 2050)  # The incomplete trailing pool: 2051 candidates in all.


def row_kinds(rows):
    """v15 layout: row 0 holds every candidate, row 1 seventeen, the rest none."""
    return ["full", "partial"][:rows] + ["empty"] * max(0, rows - 2)


def row_kind_errors(errors, kinds):
    if len(errors) != len(kinds):
        raise ValueError("One row kind is required per row error")
    result = {}
    for error, kind in zip(errors, kinds):
        current = result.get(kind, error)
        # max() would drop a NaN; a non-finite row must stay visible.
        result[kind] = error if math.isnan(error) or error > current else current
    return result


def reduce_candidates(ranked_pools, tail, kpool=KPOOL, max_pools=512):
    """Tokens after dropping the lowest-ranked pool of a full row; tail kept.

    This is the 2047-candidate layout reported by Mia and drowzeys: 511 pools
    plus the incomplete tail fit the 2048-wide kernel. Rows short of 512 pools
    already fit and lose nothing.
    """
    kept = (
        ranked_pools[: max_pools - 1]
        if len(ranked_pools) >= max_pools
        else ranked_pools
    )
    return [pool * kpool + offset for pool in kept for offset in range(kpool)] + list(
        tail
    )


def native_attention(query, indices, packed, scale, *, zero_output):
    """FlashInfer's SM120 sparse MLA on a zero-RoPE-padded query; returns a call."""
    import torch
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    tokens, heads, _ = query.shape
    width = ((indices.shape[-1] + 127) // 128) * 128
    table = torch.nn.functional.pad(indices, (0, width - indices.shape[-1]), value=-1)
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
            out=torch.zeros(tokens, 1, heads, 512, dtype=torch.bfloat16, device="cuda")
            if zero_output
            else None,
        ).squeeze(1)

    return call


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--by-row-kind", action="store_true")
    parser.add_argument("--image", help="Image ID, recorded as given (revisit only)")
    parser.add_argument("--source-commit", help="Recorded as given (revisit only)")
    args = parser.parse_args(argv)
    if args.by_row_kind and args.prefill_only:
        parser.error("--by-row-kind runs its own decode and prefill rows")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(GLM53_FUSED_UNPACK="0", GLM53_ASYNC_INDEX_CHECKS="0")
    if args.by_row_kind:
        revisit(args)
        return
    import torch
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
        "prefill_only": args.prefill_only,
        "zero_initialized_output": args.prefill_only,
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
        return native_attention(
            query, indices, packed, scale, zero_output=args.prefill_only
        )

    save()
    try:
        pack()
        check_tokens = 65 if args.prefill_only else 3
        for heads in (32, 64):
            for width in (63, 64, 65, 2048, 2051, 2176):
                query = torch.randn(
                    check_tokens, heads, 512, dtype=torch.bfloat16, device="cuda"
                )
                indices = torch.full(
                    (check_tokens, width), -1, dtype=torch.int32, device="cuda"
                )
                count = min(width, n)
                indices[0, :count] = torch.arange(
                    count, dtype=torch.int32, device="cuda"
                )
                partial = slice(-17, None) if args.prefill_only else slice(0, 17)
                indices[1, partial] = torch.arange(17, dtype=torch.int32, device="cuda")
                expected = sparse_nope_reference(query, packed, indices, scale).float()
                actual = native_call(query, indices)().float()
                torch.cuda.synchronize()
                tolerance = bf16_bound(expected.abs().max().item())
                error = (actual - expected).abs().max().item()
                case = {
                    "heads": heads,
                    "queries": check_tokens,
                    "partial_candidates": "suffix" if args.prefill_only else "prefix",
                    "width": width,
                    "max_abs_error": error,
                    "tolerance": tolerance,
                    "finite": bool(torch.isfinite(actual).all()),
                    "empty_row_zero": bool((actual[2:] == 0).all()),
                }
                case["passed"] = judge(case)["passed"]
                report["cases"].append(case)
                save()
                print(json.dumps(case), flush=True)
        latent.zero_()
        latent[2050, 0] = 16
        pack()
        tail_tokens = 65 if args.prefill_only else 1
        query = torch.zeros(tail_tokens, 32, 512, dtype=torch.bfloat16, device="cuda")
        query[..., 0] = 16
        indices = torch.arange(2051, dtype=torch.int32, device="cuda").repeat(
            tail_tokens, 1
        )
        expected = sparse_nope_reference(query, packed, indices, scale).float()
        actual = native_call(query, indices)().float()
        dropped = sparse_nope_reference(query, packed, indices[:, :2048], scale).float()
        tolerance = bf16_bound(expected.abs().max().item())
        report["tail"] = {
            "native_error": (actual - expected).abs().max().item(),
            "omission_difference": (expected - dropped).abs().max().item(),
            "tolerance": tolerance,
        }
        save()
        if (
            not all(c["passed"] for c in report["cases"])
            or not judge_tail(report["tail"])["passed"]
        ):
            raise ValueError(
                "Native candidate failed numerical or candidate-preservation checks"
            )
        latent.normal_()
        pack()
        for tokens in (65, 512) if args.prefill_only else (1, 8, 65):
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
            tolerance = bf16_bound(expected.abs().max().item())
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


def revisit(args):
    import torch
    from vllm import _custom_ops as ops

    from glm53_setup.runtime.reference_attention import (
        sparse_nope_reference,
        unpack_latent,
    )
    from glm53_setup.validation.profile_trace import (
        package_versions,
        read_trace,
        summarize_trace,
    )

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 2112  # The v15 cache: 33 blocks of 64 slots, tail slot 2050 inside.
    scale = 512**-0.5
    latent = torch.randn(n, 512, dtype=torch.bfloat16, device="cuda")
    packed = torch.zeros(n // 64, 64, 656, dtype=torch.uint8, device="cuda")
    ops.concat_and_cache_mla(
        latent,
        torch.zeros(n, 64, dtype=torch.bfloat16, device="cuda"),
        packed,
        torch.arange(n, dtype=torch.int64, device="cuda"),
        "fp8_ds_mla",
        torch.ones(1, device="cuda"),
    )
    report = {
        "status": "running",
        "scope": "SM120 zero-padded native attention revisit; no model or serving change",
        "device": torch.cuda.get_device_name(),
        "image": args.image,
        "source_commit": args.source_commit,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": package_versions(
            ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache", "torch")
        ),
        "width": REVISIT_WIDTH,
        "row_kind_cases": [],
    }

    def save():
        write_json(args.output / "result.json", report)

    save()
    try:
        for heads in (32, 64):
            # Three rows reach the decode kernel, 65 the prefill orchestrator.
            for rows in (3, 65):
                query = torch.randn(
                    rows, heads, 512, dtype=torch.bfloat16, device="cuda"
                )
                indices = torch.full(
                    (rows, REVISIT_WIDTH), -1, dtype=torch.int32, device="cuda"
                )
                indices[0] = torch.arange(
                    REVISIT_WIDTH, dtype=torch.int32, device="cuda"
                )
                partial = slice(-17, None) if rows > 64 else slice(0, 17)
                indices[1, partial] = torch.arange(17, dtype=torch.int32, device="cuda")
                expected = sparse_nope_reference(query, packed, indices, scale).float()
                for treatment in ("kernel", "slot0"):
                    case = {"heads": heads, "rows": rows, "empty_rows": treatment}
                    try:
                        table = indices.clone()
                        if treatment == "slot0":
                            # Mia's handling: one valid slot, output zeroed after.
                            table[2:, 0] = 0
                        actual = native_attention(
                            query, table, packed, scale, zero_output=False
                        )().float()
                        torch.cuda.synchronize()
                        if treatment == "slot0":
                            actual[2:] = 0
                        errors = (actual - expected).abs().amax(dim=(1, 2)).tolist()
                        case.update(
                            finite=bool(torch.isfinite(actual).all()),
                            tolerance=bf16_bound(expected.abs().max().item()),
                            max_abs_error_by_kind=row_kind_errors(
                                errors, row_kinds(rows)
                            ),
                            empty_row_zero=bool((actual[2:] == 0).all()),
                        )
                        # A NaN or inf error fails the comparison; empty rows do not count.
                        case["filled_rows_within_bound"] = all(
                            case["max_abs_error_by_kind"][kind] <= case["tolerance"]
                            for kind in ("full", "partial")
                        )
                    except Exception as error:  # noqa: BLE001 - a rejected shape is evidence
                        case["error"] = repr(error)
                    report["row_kind_cases"].append(case)
                    save()
                    print(json.dumps(case), flush=True)

        # Reduction on synthetic data: real indexer ranks and attention were not
        # captured, so pools are ranked by their reference attention mass
        # (lowest dropped), at random, and adversarially (highest dropped).
        rows, heads = 64, 32
        query = torch.randn(rows, heads, 512, dtype=torch.bfloat16, device="cuda")
        full = list(range(512 * KPOOL)) + list(TAIL)
        indices = torch.tensor([full] * rows, dtype=torch.int32, device="cuda")
        expected = sparse_nope_reference(query, packed, indices, scale).float()
        kv = unpack_latent(packed.reshape(-1, 656)[: len(full)])
        probability = torch.softmax(query.float() @ kv.T * scale, dim=-1)
        mass = probability[..., : 512 * KPOOL].reshape(rows, heads, 512, KPOOL)
        mass = mass.sum(-1).mean(1)
        rankings = {
            "lowest-mass-dropped": mass.argsort(dim=-1, descending=True),
            "random": torch.stack([torch.randperm(512) for _ in range(rows)]),
            "highest-mass-dropped": mass.argsort(dim=-1),
        }
        report["reduction"] = {
            "data": "synthetic random-normal fp8_ds_mla cache and queries",
            "rows": rows,
            "heads": heads,
            "candidates": [len(full), len(full) - KPOOL],
            "rankings": {},
        }
        for name, ranking in rankings.items():
            reduced = [
                reduce_candidates(ranking[row].tolist(), TAIL) for row in range(rows)
            ]
            table = torch.full((rows, len(full)), -1, dtype=torch.int32, device="cuda")
            for row, tokens in enumerate(reduced):
                table[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
            dropped_mass = [mass[row, ranking[row, -1]].item() for row in range(rows)]
            actual = sparse_nope_reference(query, packed, table, scale).float()
            difference = (actual - expected).abs().amax(dim=(1, 2)).tolist()
            report["reduction"]["rankings"][name] = {
                "max_abs_difference": max(difference),
                "median_abs_difference": statistics.median(difference),
                "median_dropped_pool_mass": statistics.median(dropped_mass),
                "reference_max_abs": expected.abs().max().item(),
            }
            save()

        report["timings"] = []
        for tokens in (1, 8, 65, 512):
            query = torch.randn(tokens, 32, 512, dtype=torch.bfloat16, device="cuda")
            indices = torch.arange(REVISIT_WIDTH, dtype=torch.int32, device="cuda")
            indices = indices.repeat(tokens, 1)
            row = {"queries": tokens, "heads": 32, "candidates": REVISIT_WIDTH}
            try:
                native = native_attention(
                    query, indices, packed, scale, zero_output=False
                )

                def reference():
                    return sparse_nope_reference(query, packed, indices, scale)

                expected = reference().float()
                actual = native().float()
                row.update(
                    finite=bool(torch.isfinite(actual).all()),
                    max_abs_error=(actual - expected).abs().max().item(),
                    tolerance=bf16_bound(expected.abs().max().item()),
                    paths={},
                )
                paths = (("reference", reference), ("native", native))
                if not judge(row)["passed"]:
                    row["timing_skipped"] = "parity failed"
                    paths = ()
                for label, call in paths:
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
            except Exception as error:  # noqa: BLE001 - a rejected shape is evidence
                row["error"] = repr(error)
            report["timings"].append(row)
            save()
            print(tokens, "timing recorded", flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
