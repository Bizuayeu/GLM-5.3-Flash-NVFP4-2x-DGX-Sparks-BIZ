"""Candidate-only GLM FP8 indexer score kernel; cache ownership stays external."""

import triton
import triton.language as tl


@triton.jit
def _scores(
    q,
    k,
    scales,
    weights,
    ids,
    limits,
    out,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
):
    query = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    pool = tl.load(ids + query * C + c, c < C, other=-1)
    limit = tl.load(limits + query)
    valid = (c < C) & (pool >= 0) & (pool < limit)
    d = tl.arange(0, D)
    key = (
        tl.load(k + pool[:, None] * D + d[None, :], valid[:, None], other=0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    scale = tl.load(scales + pool, valid, other=0)
    key = key * scale[:, None]
    score = tl.full((BC,), 0, tl.float32)
    for block in range(tl.cdiv(H, BH)):
        h = block * BH + tl.arange(0, BH)
        query_values = (
            tl.load(
                q + (query * H + h[:, None]) * D + d[None, :], h[:, None] < H, other=0
            )
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
        dots = tl.sum(query_values[None, :, :] * key[:, None, :], 2)
        w = tl.load(weights + query * H + h, h < H, other=0)
        score += tl.sum(tl.maximum(dots, 0) * w[None, :], 1)
    tl.store(out + query * C + c, tl.where(valid, score, float("-inf")), c < C)


def candidate_scores_cuda(q_quant, k_quant, k_scale, weights, candidates, pool_limits):
    import torch

    tensors = (q_quant, k_quant, k_scale, weights, candidates, pool_limits)
    if any(
        not t.is_cuda or not t.is_contiguous() or t.device != q_quant.device
        for t in tensors
    ):
        raise ValueError("Contiguous tensors on one CUDA device required")
    if q_quant.ndim != 3 or k_quant.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Invalid candidate scorer layout")
    queries, heads, dim = q_quant.shape
    if (
        dim != 128
        or heads != 32
        or k_quant.shape[1] != dim
        or k_scale.shape != (k_quant.shape[0],)
        or weights.shape != (queries, heads)
        or candidates.shape[0] != queries
        or pool_limits.shape != (queries,)
    ):
        raise ValueError("Expected GLM indexer dimensions: 32 heads, 128 features")
    if (
        q_quant.dtype != torch.float8_e4m3fn
        or k_quant.dtype != torch.float8_e4m3fn
        or k_scale.dtype != torch.float32
        or weights.dtype != torch.float32
        or candidates.dtype not in (torch.int32, torch.int64)
        or pool_limits.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected FP8 Q/K, FP32 scales/weights and integer pool IDs")
    # Kept at the standalone boundary until a validated engine adapter supplies
    # these invariants without host synchronizations.
    if bool(((pool_limits < 0) | (pool_limits > k_quant.shape[0])).any()) or bool(
        ((candidates < -1) | (candidates >= k_quant.shape[0])).any()
    ):
        raise ValueError("Invalid logical pool IDs or causal limits")
    out = torch.empty(candidates.shape, dtype=torch.float32, device=q_quant.device)
    if out.numel():
        _scores[(queries, triton.cdiv(candidates.shape[1], 8))](
            q_quant.view(torch.uint8),
            k_quant.view(torch.uint8),
            k_scale,
            weights,
            candidates,
            pool_limits,
            out,
            heads,
            dim,
            candidates.shape[1],
            8,
            4,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
