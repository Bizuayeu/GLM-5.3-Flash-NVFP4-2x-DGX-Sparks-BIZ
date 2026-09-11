"""Conservative eager NoPE MLA reference; preserves every selected candidate.

The packed cache layout follows vLLM's Apache-2.0 concat_and_cache_ds_mla_kernel:
512 E4M3 bytes, four FP32 scales, 64 BF16 RoPE values. Used only for validation
and a future correctness-first fallback, never advertised as a fast kernel.
"""


def unpack_latent(packed):
    import torch

    if packed.dtype != torch.uint8 or packed.shape[-1] != 656:
        raise ValueError("Expected the 656-byte fp8_ds_mla record")
    values = packed[..., :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = packed[..., 512:528].contiguous().view(torch.float32)
    return (values.reshape(*values.shape[:-1], 4, 128) * scales.unsqueeze(-1)).flatten(
        -2
    )


def sparse_nope_reference(query, packed_cache, physical_indices, scale):
    """Compute all supplied candidates with FP32 softmax, in eager query chunks."""
    import torch

    if query.ndim != 3 or query.shape[-1] != 512:
        raise ValueError("Expected absorbed NoPE query [tokens, heads, 512]")
    if physical_indices.ndim != 2 or physical_indices.shape[0] != query.shape[0]:
        raise ValueError("One candidate-index row is required per query")
    if packed_cache.dtype != torch.uint8 or packed_cache.shape[-1] != 656:
        raise ValueError("Expected packed fp8_ds_mla cache")
    flat_cache = packed_cache.reshape(-1, 656)
    if bool((physical_indices >= flat_cache.shape[0]).any().item()):
        raise ValueError("Candidate points outside cache")
    if bool((physical_indices < -1).any().item()):
        raise ValueError("Only -1 is a padding index")
    output = torch.empty_like(query)
    # Eight query rows cap gather scratch at ~34 MiB for a 2176-entry table.
    # This is an explicit memory/throughput tradeoff, not a scheduler knob.
    for start in range(0, query.shape[0], 8):
        indices = physical_indices[start : start + 8]
        valid = indices >= 0
        kv = unpack_latent(flat_cache[indices.clamp_min(0).long()])
        logits = (
            torch.matmul(query[start : start + 8].float(), kv.transpose(-1, -2)) * scale
        )
        logits.masked_fill_(~valid[:, None, :], float("-inf"))
        empty = ~valid.any(dim=-1)
        logits[empty] = 0
        probability = torch.softmax(logits, dim=-1)
        probability.masked_fill_(~valid[:, None, :], 0)
        output[start : start + 8] = torch.matmul(probability, kv).to(query.dtype)
    return output
