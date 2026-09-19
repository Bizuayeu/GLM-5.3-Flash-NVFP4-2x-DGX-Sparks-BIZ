"""One set of candidates for one input: the kpool indexer's top-k with ties settled.

The pinned vLLM's ``persistent_topk`` (decode) and ``top_k_per_row_prefill``
return a different *set* of pools for the same input when pools tie across the
k-th rank (measured on GB10: three sets in 1,200 decode calls, four in 18,000
prefill rows). The order of a set is canonicalised before attention; the set
is not, so one tie at one step forks a completion. Here a tie always goes to
the lower pool index.

Decode replaces the kernel with a stable descending sort: at most six rows, a
fraction of a millisecond, no synchronisation. Prefill keeps the kernel, which
is 25 times faster than the sort on a 2,048-row chunk, and repairs only the rows
whose k-th value is shared by more pools than fit; finding them costs one
synchronisation per call, which a prefill chunk already pays elsewhere.
"""

import os

# Rows of a prefill chunk examined at once. The comparison keeps a boolean and an
# int32 per cell: 256 rows x 65,536 pools (256K tokens / kpool 4) is about 80 MiB.
TIE_CHECK_ROWS = 256
# Rows sorted at once when a tie is repaired: values, indices and a masked copy,
# 16 bytes per cell, so 64 rows x 65,536 pools stay near 70 MiB.
SORT_ROWS = 64


def stable_topk_enabled():
    value = os.environ.get("GLM53_STABLE_INDEXER_TOPK", "1")
    if value not in ("0", "1"):
        raise ValueError("GLM53_STABLE_INDEXER_TOPK must be 0 or 1")
    return value == "1"


def stable_topk(logits, lengths, output, k, max_len):
    """Fill ``output`` [rows, k] with each row's top-k column indices, -1 padded.

    Columns past a row's length hold whatever the logits buffer held
    (``clean_logits=False``) and are masked out.
    """
    import torch

    rows = output.shape[0]
    width = min(logits.shape[1], max_len)
    limit = lengths.reshape(-1)[:rows].to(torch.int64).unsqueeze(1)
    columns = torch.arange(width, device=logits.device)
    masked = logits[:rows, :width].float().masked_fill(columns >= limit, float("-inf"))
    order = torch.sort(masked, dim=-1, descending=True, stable=True).indices[:, :k]
    ranks = torch.arange(order.shape[1], device=logits.device)
    output.fill_(-1)
    output[:, : order.shape[1]] = torch.where(ranks < limit, order, -1).to(output.dtype)


def stable_topk_ranges(logits, starts, ends, output, k):
    """The prefill form: row ``r`` selects among columns ``[starts[r], ends[r])``.

    Indices are relative to the row's start, as ``top_k_per_row_prefill`` returns them.
    """
    import torch

    columns = torch.arange(logits.shape[1], device=logits.device)
    output.fill_(-1)
    for first in range(0, output.shape[0], SORT_ROWS):
        part = slice(first, first + SORT_ROWS)
        low = starts[part].to(torch.int64).unsqueeze(1)
        high = ends[part].to(torch.int64).unsqueeze(1)
        masked = (
            logits[part]
            .float()
            .masked_fill((columns < low) | (columns >= high), float("-inf"))
        )
        order = torch.sort(masked, dim=-1, descending=True, stable=True).indices[:, :k]
        ranks = torch.arange(order.shape[1], device=logits.device)
        output[part, : order.shape[1]] = torch.where(
            ranks < high - low, order - low, -1
        ).to(output.dtype)


def repair_boundary_ties(logits, starts, ends, output, k):
    """Re-select the rows where more pools reach the k-th value than the kernel took.

    Without such a tie a row's top-k set is unique and the kernel's choice stands.
    """
    import torch

    columns = torch.arange(logits.shape[1], device=logits.device)
    tied = []
    for first in range(0, output.shape[0], TIE_CHECK_ROWS):
        part = slice(first, first + TIE_CHECK_ROWS)
        low = starts[part].to(torch.int64).unsqueeze(1)
        high = ends[part].to(torch.int64).unsqueeze(1)
        chosen = output[part].to(torch.int64)
        values = logits[part].gather(1, (chosen + low).clamp_min(0))
        threshold = values.masked_fill(chosen < 0, float("inf")).amin(
            dim=1, keepdim=True
        )
        reach = (logits[part] >= threshold) & (columns >= low) & (columns < high)
        tied.append(reach.sum(dim=1, dtype=torch.int32) > k)
    rows = torch.cat(tied).nonzero().squeeze(1)
    # Gathering the tied rows copies them, so they go a slice at a time too.
    for first in range(0, rows.numel(), SORT_ROWS):
        part = rows[first : first + SORT_ROWS]
        repaired = torch.empty(
            (part.numel(), k), dtype=output.dtype, device=output.device
        )
        stable_topk_ranges(logits[part], starts[part], ends[part], repaired, k)
        output[part] = repaired


def decode_topk(logits, lengths, output, workspace, k, max_seq_len):
    """The signature of ``persistent_topk``; the patched indexer calls this instead."""
    import torch

    if stable_topk_enabled():
        stable_topk(logits, lengths, output, k, max_seq_len)
    else:
        torch.ops._C.persistent_topk(logits, lengths, output, workspace, k, max_seq_len)


def prefill_topk(logits, starts, ends, output, rows, stride0, stride1, k):
    """The signature of ``top_k_per_row_prefill``; the patched indexer calls this instead."""
    import torch

    torch.ops._C.top_k_per_row_prefill(
        logits, starts, ends, output, rows, stride0, stride1, k
    )
    if stable_topk_enabled():
        repair_boundary_ties(
            logits[:rows], starts[:rows], ends[:rows], output[:rows], k
        )
