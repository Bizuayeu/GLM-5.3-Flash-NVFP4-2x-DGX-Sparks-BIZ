"""Probe the exact NoPE dispatch contract on GB10 without loading model weights."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    # TP=2 halves the model's 64 attention heads. The old packed FP8 backend
    # allocates a 656-byte record even for NoPE; these are initialized fixtures.
    query = torch.zeros((1, 1, 32, 512), dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros((1, 1, 64, 656), dtype=torch.uint8, device="cuda")
    workspace = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    tables = torch.zeros((1, 1, 2176), dtype=torch.int32, device="cuda")
    results = []
    for include_lengths in (False, True):
        extra = {}
        if include_lengths:
            extra["sparse_mla_top_k_lens"] = torch.ones(
                1, dtype=torch.int32, device="cuda"
            )
        try:
            result = trtllm_batch_decode_with_kv_cache_mla(
                query=query,
                kv_cache=cache,
                workspace_buffer=workspace,
                qk_nope_head_dim=256,
                kv_lora_rank=512,
                qk_rope_head_dim=0,
                block_tables=tables,
                seq_lens=None,
                max_seq_len=2048,
                bmm1_scale=512**-0.5,
                bmm2_scale=1.0,
                sparse_mla_top_k=2048,
                kv_scale_format="arbitrary_fp32",
                **extra,
            )
            torch.cuda.synchronize()
            results.append(
                {
                    "include_lengths": include_lengths,
                    "dispatch": "returned",
                    "finite": bool(torch.isfinite(result).all().item()),
                }
            )
        except (RuntimeError, ValueError, NotImplementedError, AssertionError) as error:
            results.append(
                {
                    "include_lengths": include_lengths,
                    "dispatch": "rejected",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    report = {
        "capability": list(torch.cuda.get_device_capability()),
        "cases": results,
        "numerical_parity_validated": False,
        "native_dispatch_available": all(r["dispatch"] == "returned" for r in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if not report["native_dispatch_available"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
