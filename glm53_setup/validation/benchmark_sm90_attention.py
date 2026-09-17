"""Probe FlashInfer's SM90 FA2 MLA wrapper on GB10 without changing serving.

The pinned vLLM (385dce36) selects FLASHINFER_MLA_SPARSE_SM90 only on compute
capability 9. This component calls flashinfer.mla.BatchMLAPagedAttentionWrapper
the way that backend does: NoPE (head_dim_kpe=0), page size 1 with each query
row's candidates as its KV pages, exact per-row lengths and causal=False, with
backend "fa2" in place of the backend's "fa3". Outputs are compared with the
candidate-preserving reference on the same packed cache. Passing shows only
that the kernel is numerically usable here; a serving switch would requalify
fused unpack, APC, LPA and candidate order, which assume the packed SM120 cache.
"""

import argparse
import hashlib
import importlib.metadata
import os
import statistics
import time
from pathlib import Path

from glm53_setup.io import write_json

WIDTHS = (0, 17, 63, 64, 65, 2048, 2051, 2176)
TAIL_SLOT = 2050
SLOTS = 2240  # 35 blocks of 64: room for 2,176 distinct candidates
BATCH_ROWS = (64, 128, 256)  # FlashInfer 0.6.17 was reported to return NaN here
TIMING_ROWS = (1, 8, 65, 512)
HEADS = 32
SCALE = 512**-0.5
WORKSPACE_BYTES = 128 * 1024**2  # The SM90 backend's float workspace


def parity_cases():
    """Every width with contiguous and sliced queries; empty-row cases run last."""
    filled = [
        {"width": width, "contiguous": contiguous, "empty_row": False}
        for width in WIDTHS
        if width
        for contiguous in (True, False)
    ]
    empty = [
        {"width": width, "contiguous": contiguous, "empty_row": True}
        for width in WIDTHS
        for contiguous in (True, False)
    ]
    return filled + empty


def case_rows(case):
    """Row 0 holds every candidate, row 1 a sole tail slot, row 2 (if any) none."""
    width = case["width"]
    rows = [list(range(width)), [TAIL_SLOT] if width else []]
    return rows + [[]] if case["empty_row"] else rows


def padded_rows(rows, width):
    """Reference layout: -1 padding first, so a sole candidate sits at the tail."""
    return [[-1] * (width - len(row)) + row for row in rows]


def compact(rows):
    """Page-size-one KV ranges: row i attends indices[indptr[i]:indptr[i + 1]]."""
    indptr, indices = [0], []
    for row in rows:
        indices.extend(row)
        indptr.append(len(indices))
    return indptr, indices, [len(row) for row in rows]


def judge(case):
    reasons = []
    if not case["finite"]:
        reasons.append("non-finite")
    if case["max_abs_error"] > case["tolerance"]:
        reasons.append("error-above-bound")
    if case.get("empty_row_zero") is False:
        reasons.append("empty-row-nonzero")
    return {"passed": not reasons, "reasons": reasons}


def judge_tail(tail):
    reasons = []
    if tail["native_error"] > tail["tolerance"]:
        reasons.append("error-above-bound")
    if not tail["omission_difference"] > 1:
        reasons.append("tail-insensitive")
    return {"passed": not reasons, "reasons": reasons}


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", help="Image ID, recorded as given by the runner")
    parser.add_argument("--source-commit", help="Source commit, recorded as given")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(GLM53_FUSED_UNPACK="0", GLM53_ASYNC_INDEX_CHECKS="0")
    import torch
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    from glm53_setup.runtime.reference_attention import (
        sparse_nope_reference,
        unpack_latent,
    )
    from glm53_setup.validation.profile_trace import read_trace, summarize_trace

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    report = {
        "status": "running",
        "scope": "SM90 FA2 MLA wrapper component on GB10; no model or serving change",
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "image": args.image,
        "source_commit": args.source_commit,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": package_versions(
            (
                "flashinfer-python",
                "flashinfer-cubin",
                "flashinfer-jit-cache",
                "torch",
                "vllm",
            )
        ),
        "backend": "fa2",
        "heads": HEADS,
        "cases": [],
    }

    def save():
        write_json(args.output / "result.json", report)

    def random_cache():
        packed = torch.zeros((SLOTS, 656), dtype=torch.uint8, device=device)
        packed[:, :512] = (
            torch.randn((SLOTS, 512), device=device)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        packed[:, 512:528] = torch.rand((SLOTS, 4), device=device).view(torch.uint8)
        return packed

    def bf16_kv(packed):
        ckv = unpack_latent(packed).to(torch.bfloat16).reshape(SLOTS, 1, 512)
        return ckv.contiguous(), ckv.new_empty((SLOTS, 1, 0))

    def queries(count, contiguous=True):
        if contiguous:
            return torch.randn((count, HEADS, 512), dtype=torch.bfloat16, device=device)
        # A slice of a wider buffer, as when q_nope is cut from q.
        wide = torch.randn((count, HEADS, 576), dtype=torch.bfloat16, device=device)
        return wide[..., :512]

    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")

    def plan_inputs(rows):
        indptr, indices, lengths = compact(rows)
        return (
            torch.arange(len(rows) + 1, dtype=torch.int32),
            torch.tensor(indptr, dtype=torch.int32),
            torch.tensor(indices, dtype=torch.int32, device=device),
            torch.tensor(lengths, dtype=torch.int32),
        )

    def plan(inputs, kv_dtype=torch.bfloat16):
        wrapper.plan(
            *inputs,
            HEADS,
            512,  # head_dim_ckv
            0,  # head_dim_kpe: NoPE
            1,  # page size: candidates are the page table
            False,  # causal: encoded by the indexer's selection
            SCALE,
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dtype,
        )

    def run(query, ckv, kpe, **scales):
        q_pe = query.new_zeros((query.shape[0], HEADS, 0))
        return wrapper.run(query, q_pe, ckv, kpe, **scales)

    def reference(query, packed, rows, width):
        indices = torch.tensor(
            padded_rows(rows, width), dtype=torch.int32, device=device
        ).reshape(len(rows), width)
        return sparse_nope_reference(query, packed, indices, SCALE)

    def compare(expected, actual, rows):
        expected, actual = expected.float(), actual.float()
        filled = [i for i, row in enumerate(rows) if row]
        empty = [i for i, row in enumerate(rows) if not row]
        result = {
            "finite": bool(torch.isfinite(actual).all()),
            "tolerance": 2
            * torch.finfo(torch.bfloat16).eps
            * max(1.0, expected.abs().max().item()),
            "max_abs_error": (actual[filled] - expected[filled]).abs().max().item()
            if filled
            else 0.0,
        }
        if empty:
            result["empty_row_zero"] = bool((actual[empty] == 0).all())
            result["empty_row_max_abs"] = actual[empty].abs().max().item()
        return result

    def timed(call):
        # P04 conditions: three warmups, then five synchronized batches of ten.
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
        return {"wall_ms": samples, "median_wall_ms": statistics.median(samples)}

    def profiled(call, name):
        trace = args.output / f"{name}.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            call()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace))
        return summarize_trace(read_trace(trace))

    def parity(case, packed, ckv, kpe):
        rows = case_rows(case)
        row = dict(case)
        try:
            query = queries(len(rows), case["contiguous"])
            expected = reference(query, packed, rows, case["width"])
            plan(plan_inputs(rows))
            actual = run(query, ckv, kpe)
            torch.cuda.synchronize()
            row.update(compare(expected, actual, rows))
            row.update(judge(row))
        except Exception as error:  # noqa: BLE001 - a rejected shape is evidence
            row.update(passed=False, reasons=["error"], error=repr(error))
        report["cases"].append(row)
        save()
        return row["passed"]

    save()
    try:
        packed = random_cache()
        ckv, kpe = bf16_kv(packed)
        began = time.perf_counter()
        plan(plan_inputs([[0]]))  # Loads or JIT-builds the NoPE FA2 module.
        report["first_plan_seconds"] = round(time.perf_counter() - began, 3)
        save()

        cases = parity_cases()
        filled = [parity(c, packed, ckv, kpe) for c in cases if not c["empty_row"]]

        tail_packed = torch.zeros((SLOTS, 656), dtype=torch.uint8, device=device)
        latent = torch.zeros((SLOTS, 512), device=device)
        latent[TAIL_SLOT, 0] = 16
        tail_packed[:, :512] = latent.to(torch.float8_e4m3fn).view(torch.uint8)
        tail_packed[:, 512:528] = torch.ones((SLOTS, 4), device=device).view(
            torch.uint8
        )
        tail_ckv, tail_kpe = bf16_kv(tail_packed)
        query = torch.zeros((1, HEADS, 512), dtype=torch.bfloat16, device=device)
        query[..., 0] = 16
        rows = [list(range(2051))]
        expected = reference(query, tail_packed, rows, 2051).float()
        plan(plan_inputs(rows))
        actual = run(query, tail_ckv, tail_kpe).float()
        dropped = reference(query, tail_packed, [list(range(2048))], 2048).float()
        report["tail"] = {
            "native_error": (actual - expected).abs().max().item(),
            "omission_difference": (expected - dropped).abs().max().item(),
            "tolerance": 2
            * torch.finfo(torch.bfloat16).eps
            * max(1.0, expected.abs().max().item()),
        }
        report["tail"].update(judge_tail(report["tail"]))
        save()

        report["batches"] = []
        for count in BATCH_ROWS:
            rows = [list(range(2176))] * count
            row = {"rows": count, "candidates": 2176}
            try:
                query = queries(count)
                expected = reference(query, packed, rows, 2176)
                plan(plan_inputs(rows))
                actual = run(query, ckv, kpe)
                torch.cuda.synchronize()
                row.update(compare(expected, actual, rows))
                row.update(judge(row))
            except Exception as error:  # noqa: BLE001 - recorded as a failed batch
                row.update(passed=False, reasons=["error"], error=repr(error))
            report["batches"].append(row)
            save()

        numerics = (
            all(filled)
            and report["tail"]["passed"]
            and all(b["passed"] for b in report["batches"])
        )
        report["timings"] = []
        if not numerics:
            report["timing_skipped"] = "numerical checks failed; no speed is claimed"
        for count in TIMING_ROWS if numerics else ():
            rows = [list(range(2176))] * count
            query = queries(count)
            indices = torch.arange(2176, dtype=torch.int32, device=device).repeat(
                count, 1
            )
            inputs = plan_inputs(rows)
            plan(inputs)
            row = {"rows": count, "heads": HEADS, "candidates": 2176, "paths": {}}
            row.update(
                compare(
                    reference(query, packed, rows, 2176), run(query, ckv, kpe), rows
                )
            )
            paths = {
                "reference": lambda: sparse_nope_reference(
                    query, packed, indices, SCALE
                ),
                "sm90_fa2_run": lambda: run(query, ckv, kpe),
            }
            for label, call in paths.items():
                row["paths"][label] = {
                    **timed(call),
                    "profile": profiled(call, f"{count}-{label}"),
                }
            # plan() copies to the host; the backend calls it once per step.
            row["paths"]["sm90_fa2_plan"] = timed(lambda: plan(inputs))
            plan(inputs)
            report["timings"].append(row)
            save()

        if numerics:
            # FlashInfer 0.6.18 accepts FP8 MLA KV only on SM90 devices.
            fp8 = {"per_tensor_scale": 1.0, "cases": []}
            report["fp8"] = fp8
            uniform = packed.clone()
            uniform[:, 512:528] = torch.ones((SLOTS, 4), device=device).view(
                torch.uint8
            )
            ckv8 = uniform[:, :512].contiguous().view(torch.float8_e4m3fn)
            ckv8 = ckv8.reshape(SLOTS, 1, 512)
            kpe8 = ckv8.new_empty((SLOTS, 1, 0))
            try:
                for width in (17, 2048, 2176):
                    rows = [list(range(width))]
                    query = queries(1)
                    expected = reference(query, uniform, rows, width)
                    plan(plan_inputs(rows), kv_dtype=torch.float8_e4m3fn)
                    actual = run(query, ckv8, kpe8, ckv_scale=1.0, kpe_scale=1.0)
                    torch.cuda.synchronize()
                    case = {"width": width, **compare(expected, actual, rows)}
                    case.update(judge(case))
                    fp8["cases"].append(case)
                fp8["status"] = "ran"
            except ValueError as error:
                fp8.update(status="rejected-by-library", error=str(error))
            except RuntimeError as error:
                fp8.update(status="kernel-error", error=str(error))
            save()

        # Last: an empty row may fault the kernel and poison the CUDA context.
        empty = [parity(c, packed, ckv, kpe) for c in cases if c["empty_row"]]
        report["numerically_usable"] = numerics and all(empty)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    if not report["numerically_usable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
