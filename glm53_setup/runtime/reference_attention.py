"""Conservative eager NoPE MLA reference; preserves every selected candidate.

The packed cache layout follows vLLM's Apache-2.0 concat_and_cache_ds_mla_kernel:
512 E4M3 bytes, four FP32 scales, 64 BF16 RoPE values. Used only for validation
and a future correctness-first fallback, never advertised as a fast kernel.
"""

import os


def unpack_latent(packed):
    import torch

    if packed.dtype != torch.uint8 or packed.shape[-1] != 656:
        raise ValueError("Expected the 656-byte fp8_ds_mla record")
    if os.environ.get("GLM53_FUSED_UNPACK") == "1":
        from glm53_setup.runtime.fused_unpack import unpack_latent_cuda

        return unpack_latent_cuda(packed)
    values = packed[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = packed[..., 512:528].contiguous().view(torch.float32)
    return (values.reshape(*values.shape[:-1], 4, 128) * scales.unsqueeze(-1)).flatten(
        -2
    )


def sparse_nope_reference(
    query, packed_cache, physical_indices, scale, *, query_chunk=8
):
    """Compute all supplied candidates with FP32 softmax, in eager query chunks."""
    if type(query_chunk) is not int or query_chunk not in (8, 32, 64):
        raise ValueError("query_chunk must be one of the experimental sizes 8, 32, 64")
    import torch

    if query.ndim != 3 or query.shape[-1] != 512:
        raise ValueError("Expected absorbed NoPE query [tokens, heads, 512]")
    if physical_indices.ndim != 2 or physical_indices.shape[0] != query.shape[0]:
        raise ValueError("One candidate-index row is required per query")
    if packed_cache.dtype != torch.uint8 or packed_cache.shape[-1] != 656:
        raise ValueError("Expected packed fp8_ds_mla cache")
    flat_cache = packed_cache.reshape(-1, 656)
    if os.environ.get("GLM53_ASYNC_INDEX_CHECKS") == "1":
        # Backend-generated indices are an internal invariant. A violation in
        # this opt-in path terminates the CUDA context; it is never ignored.
        torch._assert_async(
            ((physical_indices >= -1) & (physical_indices < flat_cache.shape[0])).all(),
            "Invalid sparse MLA physical index",
        )
    else:
        if bool((physical_indices >= flat_cache.shape[0]).any().item()):
            raise ValueError("Candidate points outside cache")
        if bool((physical_indices < -1).any().item()):
            raise ValueError("Only -1 is a padding index")
    if os.environ.get("GLM53_FA2_ATTENTION") == "1":
        # Opt-in prefill kernel; decode-sized calls continue below.
        from glm53_setup.runtime.fa2_attention import sparse_nope_fa2, use_fa2

        if use_fa2(query.shape[0]):
            return sparse_nope_fa2(query, packed_cache, physical_indices, scale)
    output = torch.empty_like(query)
    # Default eight rows cap FP32 KV scratch at ~34 MiB for 2176 candidates.
    # The larger experimental sizes trade memory for fewer launches; they are
    # independent of the scheduler's prefill chunk budget.
    for start in range(0, query.shape[0], query_chunk):
        indices = physical_indices[start : start + query_chunk]
        valid = indices >= 0
        kv = unpack_latent(flat_cache[indices.clamp_min(0).long()])
        logits = (
            torch.matmul(
                query[start : start + query_chunk].float(), kv.transpose(-1, -2)
            )
            * scale
        )
        logits.masked_fill_(~valid[:, None, :], float("-inf"))
        empty = ~valid.any(dim=-1)
        logits[empty] = 0
        probability = torch.softmax(logits, dim=-1)
        probability.masked_fill_(~valid[:, None, :], 0)
        output[start : start + query_chunk] = torch.matmul(probability, kv).to(
            query.dtype
        )
    return output
