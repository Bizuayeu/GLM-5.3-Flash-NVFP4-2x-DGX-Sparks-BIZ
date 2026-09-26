"""Hierarchical shared-pool scoring through the pinned native Tensor Core path."""


def shared_pool_scores(q_quant, k_quant, k_scale, weights, candidate_pool, pool_limits):
    """Score a sorted, unique shared pool, using pool_limits for causality.

    Outputs use candidate_pool coordinates. Pool selection policy is external.
    """
    import torch
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    if q_quant.ndim != 3 or k_quant.ndim != 2 or candidate_pool.ndim != 1:
        raise ValueError("Q/K matrices and a shared one-dimensional pool required")
    queries, heads, dim = q_quant.shape
    if (
        (heads, dim) != (32, 128)
        or k_quant.shape[1] != dim
        or weights.shape != (queries, heads)
        or k_scale.shape != (k_quant.shape[0],)
        or pool_limits.shape != (queries,)
    ):
        raise ValueError("Expected pinned GLM indexer dimensions")
    tensors = (q_quant, k_quant, k_scale, weights, candidate_pool, pool_limits)
    if any(
        not t.is_cuda or not t.is_contiguous() or t.device != q_quant.device
        for t in tensors
    ):
        raise ValueError("Contiguous tensors on one CUDA device required")
    if (
        q_quant.dtype != torch.float8_e4m3fn
        or k_quant.dtype != torch.float8_e4m3fn
        or k_scale.dtype != torch.float32
        or weights.dtype != torch.float32
        or candidate_pool.dtype not in (torch.int32, torch.int64)
        or pool_limits.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected FP8 Q/K, FP32 scales/weights and integer pool IDs")
    if candidate_pool.numel() == 0:
        raise ValueError("Empty shared pool is not supported by this native adapter")
    if bool(
        ((candidate_pool < 0) | (candidate_pool >= k_quant.shape[0])).any()
    ) or bool((candidate_pool[1:] <= candidate_pool[:-1]).any()):
        raise ValueError("Shared pool must contain sorted, unique, valid pool IDs")
    if bool(((pool_limits < 0) | (pool_limits > k_quant.shape[0])).any()):
        raise ValueError("Invalid causal pool limits")
    ids = candidate_pool.long()
    keys = k_quant.view(torch.uint8).index_select(0, ids).view(torch.float8_e4m3fn)
    scales = k_scale.index_select(0, ids)
    ends = torch.searchsorted(candidate_pool, pool_limits).to(torch.int32)
    starts = torch.zeros_like(ends)
    return fp8_fp4_mqa_logits(
        (q_quant, None), (keys, scales), weights, starts, ends, clean_logits=True
    )
