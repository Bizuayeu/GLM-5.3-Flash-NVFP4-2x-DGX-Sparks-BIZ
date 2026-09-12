"""Experimental FP32 online NoPE attention without materialized gathered KV."""

import triton
import triton.language as tl


@triton.jit
def _attention(
    Query,
    Cache,
    Indices,
    Output,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64)
    columns = tl.arange(0, 512)
    q = tl.load(Query + token * Q0 + head * Q1 + columns).to(tl.float32)
    acc = tl.full((512,), 0, tl.float32)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    cache_scales = Cache.to(tl.pointer_type(tl.float32))
    for start in range(0, tl.cdiv(WIDTH, BLOCK_K)):
        offsets = start * BLOCK_K + tl.arange(0, BLOCK_K)
        indices = tl.load(
            Indices + token * I0 + offsets * I1, offsets < WIDTH, other=-1
        )
        valid = (offsets < WIDTH) & (indices >= 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            safe = tl.maximum(indices, 0).to(tl.int64)
            bits = tl.load(
                Cache + safe[:, None] * 656 + columns[None, :], valid[:, None], other=0
            )
            values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scales = tl.load(
                cache_scales + safe[:, None] * 164 + 128 + columns[None, :] // 128,
                valid[:, None],
                other=0,
            )
            kv = values * scales
            scores = tl.sum(kv * q[None, :], 1) * SCALE
            scores = tl.where(valid, scores, float("-inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, 0))
            alpha = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum)
            denominator = denominator * alpha + tl.sum(probabilities, 0)
            acc = acc * alpha + tl.sum(probabilities[:, None] * kv, 0)
            maximum = next_maximum
    result = acc / tl.maximum(denominator, 1.0)
    tl.store(Output + (token * HEADS + head) * 512 + columns, result)


def fused_nope_attention(query, packed_cache, physical_indices, scale):
    import torch

    if query.ndim != 3 or query.shape[-1] != 512 or query.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 NoPE query [tokens, heads, 512]")
    if (
        packed_cache.ndim < 1
        or packed_cache.dtype != torch.uint8
        or packed_cache.shape[-1] != 656
    ):
        raise ValueError("Expected 656-byte packed cache")
    if (
        physical_indices.ndim != 2
        or physical_indices.shape[0] != query.shape[0]
        or physical_indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected per-query integer candidate rows")
    if (
        not query.is_cuda
        or packed_cache.device != query.device
        or physical_indices.device != query.device
    ):
        raise ValueError("All attention tensors must share a CUDA device")
    if (
        not packed_cache.is_contiguous()
        or packed_cache.storage_offset() % 4
        or query.stride(-1) != 1
    ):
        raise ValueError(
            "Aligned contiguous cache and unit-stride query columns required"
        )
    cache = packed_cache.reshape(-1, 656)
    if bool((physical_indices >= cache.shape[0]).any().item()):
        raise ValueError("Candidate points outside cache")
    if bool((physical_indices < -1).any().item()):
        raise ValueError("Only -1 is a padding index")
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if query.shape[0] and query.shape[1]:
        # Sixteen keys × 512 columns bounds register pressure. Head is the
        # fast grid axis so CTAs for one query can reuse KV through L2.
        _attention[(query.shape[1], query.shape[0])](
            query,
            cache,
            physical_indices,
            output,
            query.stride(0),
            query.stride(1),
            physical_indices.stride(0),
            physical_indices.stride(1),
            query.shape[1],
            physical_indices.shape[1],
            scale,
            BLOCK_K=16,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
