"""Top-k with ties broken by the lower index, for the kpool indexer's decode step.

The pinned vLLM's ``persistent_topk`` returns the right set in a different
order on every call, and when pools tie across the k-th rank it returns a
different *set* for the same input (measured on GB10: three sets in 1,200
calls). The order is canonicalised before attention; the set is not, so one
tie at one step forks a completion. A stable descending sort over each row's
valid prefix picks the same pools every time. Columns past a row's length hold
whatever the logits buffer held and are masked out.
"""


def stable_topk(logits, lengths, output, k, max_len):
    """Fill ``output`` [rows, k] with each row's top-k column indices, -1 padded."""
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
