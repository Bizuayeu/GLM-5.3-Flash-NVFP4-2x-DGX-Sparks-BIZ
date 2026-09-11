"""Real GPU cache-pack and attention parity; no model download or serving."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from ..runtime.reference_attention import sparse_nope_reference, unpack_latent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    from vllm import _custom_ops as ops

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    n = 2112  # 33 pages: includes candidates beyond the old 2048 width.
    latent = torch.randn(n, 512, dtype=torch.bfloat16, device=device)
    rope = torch.zeros(n, 64, dtype=torch.bfloat16, device=device)
    packed = torch.zeros(n // 64, 64, 656, dtype=torch.uint8, device=device)
    slots = torch.arange(n, dtype=torch.int64, device=device)
    ops.concat_and_cache_mla(
        latent, rope, packed, slots, "fp8_ds_mla", torch.ones(1, device=device)
    )
    torch.cuda.synchronize()
    decoded = unpack_latent(packed.reshape(n, 656))
    # Independent scalar-group decode on CPU, using bytes written by the real op.
    cpu = packed.cpu().reshape(n, 656)
    expected_kv = torch.empty(n, 512, dtype=torch.float64)
    for group in range(4):
        values = (
            cpu[:, group * 128 : (group + 1) * 128]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .double()
        )
        scales = (
            cpu[:, 512 + group * 4 : 516 + group * 4]
            .contiguous()
            .view(torch.float32)
            .double()
        )
        expected_kv[:, group * 128 : (group + 1) * 128] = values * scales
    cache_rounding_error = float((decoded.cpu().double() - expected_kv).abs().max())
    cache_error = float((decoded.cpu() - expected_kv.float()).abs().max())
    cases = []
    for width in [63, 64, 65, 2048, 2051, 2176]:
        query = torch.randn(3, 32, 512, dtype=torch.bfloat16, device=device)
        indices = torch.full((3, width), -1, dtype=torch.int32, device=device)
        count = min(width, n)
        indices[0, :count] = torch.arange(count, dtype=torch.int32, device=device)
        indices[1, : min(count, 17)] = torch.arange(
            min(count, 17), dtype=torch.int32, device=device
        )
        scale = 512**-0.5
        actual = sparse_nope_reference(query, packed, indices, scale).float().cpu()
        expected = torch.zeros_like(actual)
        for row in [0, 1]:
            ids = indices[row].cpu().long()
            k = expected_kv[ids[ids >= 0]]
            logits = query[row].cpu().double() @ k.T * scale
            expected[row] = (logits.softmax(-1) @ k).bfloat16().float()
        error = float((actual - expected).abs().max())
        # Two BF16 ulps at the largest reference magnitude, fixed before run.
        tolerance = (
            2 * torch.finfo(torch.bfloat16).eps * max(1.0, float(expected.abs().max()))
        )
        cases.append(
            {
                "width": width,
                "max_abs_error": error,
                "tolerance": tolerance,
                "empty_row_zero": bool((actual[2] == 0).all()),
                "passed": error <= tolerance and bool((actual[2] == 0).all()),
            }
        )
    # Sensitivity check: an omitted tail must be detectable, not washed out by tolerance.
    latent.zero_()
    latent[2050, 0] = 16
    ops.concat_and_cache_mla(
        latent, rope, packed, slots, "fp8_ds_mla", torch.ones(1, device=device)
    )
    query = torch.zeros(1, 32, 512, dtype=torch.bfloat16, device=device)
    query[..., 0] = 16
    indices = torch.arange(2051, dtype=torch.int32, device=device).unsqueeze(0)
    full = sparse_nope_reference(query, packed, indices, 512**-0.5)
    dropped = sparse_nope_reference(query, packed, indices[:, :2048], 512**-0.5)
    sensitivity = float((full.float() - dropped.float()).abs().max().item())
    # Exercise the installed backend and its real Triton physical-slot mapper.
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        FlashInferMLASparseSM120Impl,
    )

    backend = object.__new__(FlashInferMLASparseSM120Impl)
    backend._glm53_reference_nope = True
    # The production index table is padded to a multiple of 128. Retain all
    # 2051 candidates; only append -1 padding, never truncate the tail.
    backend.topk_indices_buffer = torch.nn.functional.pad(indices, (0, 125), value=-1)
    backend.scale = 512**-0.5
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(1, dtype=torch.int32, device=device),
        block_table=torch.arange(n // 64, dtype=torch.int32, device=device).unsqueeze(
            0
        ),
        block_size=64,
    )
    integrated, _ = backend.forward_mqa(
        torch.nn.functional.pad(query, (0, 64)), packed, metadata, None
    )
    integration_error = float((integrated.float() - full.float()).abs().max().item())
    report = {
        "cache_decode_max_abs_error": cache_error,
        "fp32_vs_fp64_rounding_error": cache_rounding_error,
        "cases": cases,
        "tail_omission_difference": sensitivity,
        "installed_backend_max_abs_error": integration_error,
        "passed": cache_error == 0
        and all(c["passed"] for c in cases)
        and sensitivity > 1.0
        and integration_error == 0,
        "full_model_inference_validated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
