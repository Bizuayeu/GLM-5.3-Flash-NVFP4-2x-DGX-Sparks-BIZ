"""FlashInfer FA2 MLA for the candidate-preserving NoPE attention, prefill only.

Same contract as ``reference_attention.sparse_nope_reference``: every selected
candidate is kept, ``-1`` is padding, a row without candidates yields zeros.
The packed ``fp8_ds_mla`` cache stays as it is; the rows a call touches are
unpacked to BF16 and handed to ``BatchMLAPagedAttentionWrapper`` with page size
one, each query row's candidates being its KV pages (the way the pinned vLLM's
SM90 backend drives the wrapper, with ``fa2`` because FlashInfer 0.6.18 offers
no ``fa3`` and no FP8 MLA KV off SM90). Prefix-cache blocks, fused unpack and
candidate order are upstream of this call and unchanged.

``plan`` needs the per-row lengths on the host, one synchronisation per MLA
layer. That is cheap beside a prefill chunk and costly in a decode step, so
decode-sized calls stay on the reference path.
"""

import os

# With one sequence a decode step carries at most num_speculative_tokens + 1 = 6
# query rows (server_config accepts depths 1 to 5); anything larger is prefill.
DECODE_MAX_ROWS = 6
# FlashInfer's float workspace, the size the pinned SM90 backend reserves.
WORKSPACE_BYTES = 128 * 1024**2
# Unpacking goes through FP32; slices bound that scratch to 128 MiB.
UNPACK_ROWS = 65536

_wrappers = {}


def fa2_enabled():
    value = os.environ.get("GLM53_FA2_ATTENTION", "0")
    if value not in ("0", "1"):
        raise ValueError("GLM53_FA2_ATTENTION must be 0 or 1")
    return value == "1"


def use_fa2(query_rows):
    return fa2_enabled() and query_rows > DECODE_MAX_ROWS


def compact_candidates(physical_indices):
    """Page-size-one KV ranges over the distinct cache rows the call touches.

    Returns the distinct physical rows, each query row's candidates as positions
    in that list (row-major, padding dropped) and the per-row candidate counts.
    Padding maps to cache row 0, which then rides along unused.
    """
    import torch

    valid = physical_indices >= 0
    rows, inverse = torch.unique(physical_indices.clamp_min(0), return_inverse=True)
    return rows, inverse[valid].to(torch.int32), valid.sum(dim=1).to(torch.int32)


def indptr(lengths):
    import torch

    result = torch.zeros(lengths.numel() + 1, dtype=torch.int32)
    result[1:] = torch.cumsum(lengths, dim=0)
    return result


def _wrapper(device):
    import torch
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    key = (device.type, device.index)
    if key not in _wrappers:
        workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _wrappers[key] = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    return _wrappers[key]


def sparse_nope_fa2(query, packed_cache, physical_indices, scale):
    import torch

    from glm53_setup.runtime.reference_attention import unpack_latent

    if query.dtype != torch.bfloat16:
        raise ValueError("The FA2 path expects BF16 queries")
    flat_cache = packed_cache.reshape(-1, 656)
    rows, kv_indices, lengths = compact_candidates(physical_indices)
    ckv = torch.empty((rows.numel(), 1, 512), dtype=torch.bfloat16, device=query.device)
    for start in range(0, rows.numel(), UNPACK_ROWS):
        part = rows[start : start + UNPACK_ROWS].long()
        ckv[start : start + part.numel(), 0] = unpack_latent(flat_cache[part])
    host_lengths = lengths.to("cpu")  # the one synchronisation of this call
    wrapper = _wrapper(query.device)
    wrapper.plan(
        torch.arange(query.shape[0] + 1, dtype=torch.int32),
        indptr(host_lengths),
        kv_indices,
        host_lengths,
        query.shape[1],
        512,  # head_dim_ckv
        0,  # head_dim_kpe: NoPE
        1,  # page size: the candidates are the page table
        False,  # causal: encoded by the indexer's selection
        scale,
        q_data_type=query.dtype,
        kv_data_type=torch.bfloat16,
    )
    q_pe = query.new_zeros((query.shape[0], query.shape[1], 0))
    return wrapper.run(query, q_pe, ckv, ckv.new_empty((rows.numel(), 1, 0)))
