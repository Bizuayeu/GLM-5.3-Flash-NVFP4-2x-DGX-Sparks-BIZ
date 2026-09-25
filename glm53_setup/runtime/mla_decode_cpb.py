"""One chunks_per_block per decode step shape in the sparse-MLA decode, whatever the partner.

FlashInfer 0.6.18's SM120 sparse-MLA decode (``sparse_mla_sm120_decode_dsv3_2``) splits the 2,048
candidates into 32 chunks of 64 and lets each CTA take ``chunks_per_block`` of them. Left unset, a
C++ heuristic picks it from the token count of the whole call and the SM count, so a request's rows
are summed in another order as soon as a second sequence shares the step (on GB10, 2 for one draft
token and 3 for two; 6 for one verification step of four tokens and 15 for two). The stock route
also rebuilds FlashInfer's runner, normalizes the index segments, goes through a custom op and asks
FlashInfer's AutoTuner on every call.

With ``GLM53_MLA_DECODE_CPB=1`` (``runtime.mla_decode_cpb``) the patched backend asks
``decode_chunks_per_block`` first and, for a step it names, calls the same FlashInfer decode entry
the stock route ends in, with the value taken from the tokens of one sequence. The values are the
heuristic's own at one sequence, so a request alone computes bit for bit as before, and with a
partner its rows no longer change. On one GB10 through the patched backend the call skipped the
wrapper cost: a verification step of one sequence took 221 us against 633 and of two 535 against
917 (records/20260925-moe-batch, run 4); on the serving pair decode tokens/s did not move.
Every other step keeps the stock call.
"""

import os

# Tokens of one sequence -> chunks_per_block, for 32 heads per rank and top-k 2,048 on GB10.
CHUNKS_PER_BLOCK = {1: 2, 4: 6}
HEADS = 32
TOP_K = 2048
# cc-defer: measured for one and two sequences only; a profile with max_num_seqs > 2 needs the
# per-sequence values timed at its token counts (at 16 sequences of four tokens cpb 6 is slower).
MAX_SEQUENCES = 2


def cpb_enabled():
    value = os.environ.get("GLM53_MLA_DECODE_CPB", "0")
    if value not in ("0", "1"):
        raise ValueError("GLM53_MLA_DECODE_CPB must be 0 or 1")
    return value == "1"


def decode_chunks_per_block(
    num_tokens, num_reqs, max_query_len, num_prefills, num_heads, top_k
):
    """The pinned value for a decode step of equal-length sequences, or None for the stock choice."""
    if not cpb_enabled():
        return None
    if (
        num_prefills
        or num_heads != HEADS
        or top_k != TOP_K
        or not 0 < num_reqs <= MAX_SEQUENCES
    ):
        return None
    if num_tokens != num_reqs * max_query_len:
        return None
    return CHUNKS_PER_BLOCK.get(max_query_len)


def pinned_decode(
    q, kv_c_and_k_pe_cache, indices, output, workspace, scale, kv_scale_format, cpb
):
    """Run the stock SM120 decode entry with ``chunks_per_block`` set; False if it would not dispatch.

    Mirrors FlashInfer's own route for ``seq_lens=None``: flattened indices, no top-k lengths, the
    split scratch cut from the same workspace buffer, the model type from the KV scale format.
    """
    import torch
    from flashinfer.mla import _core
    from flashinfer.mla import _sparse_mla_sm120 as sm120

    num_tokens, num_heads, d_qk = q.shape
    kv_cache = kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1)
    model_type = sm120._resolve_model_type(d_qk, kv_scale_format)
    page = sm120._packed_kv_page_block_size(
        kv_cache, model_type=model_type, name="kv_cache"
    )
    top_k = indices.shape[-1]
    if not sm120._decode_dsv3_2_dispatchable(num_tokens, num_heads, top_k, d_qk, page):
        return False
    mid_out, mid_lse = _core._sparse_mla_decode_workspace(
        workspace,
        num_tokens=num_tokens,
        num_heads=num_heads,
        d_v=output.shape[-1],
        topk=top_k,
        extra_topk=0,
    )
    if mid_out is None:
        return False
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    sm120.sparse_mla_sm120_decode_dsv3_2(
        q,
        kv_cache,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        scale,
        model_type=model_type,
        chunks_per_block=cpb,
    )
    return True
