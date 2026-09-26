"""Request-local selection reuse and candidate-only FP8 indexer reference scoring."""


class CandidateReuse:
    """Own logical IDs only; caller must still update each target's K/tail caches."""

    def __init__(self, request_id, pairs):
        pairs = tuple(pairs)
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Request identity required")
        if any(
            type(s) is not int or type(t) is not int or not 0 <= s < t for s, t in pairs
        ):
            raise ValueError("Reuse must go from an earlier to a later layer")
        self.sources = {target: source for source, target in pairs}
        if len(self.sources) != len(pairs):
            raise ValueError("A target may have only one source")
        self.request_id = request_id
        self.cache = {}
        self.closed = False

    def remember(self, row):
        self._check(row["request_id"])
        position = row["query_position"]
        if (
            row["coordinate_space"] != "logical_tokens"
            or type(position) is not int
            or position < 0
            or type(row["layer"]) is not int
            or row["layer"] < 0
        ):
            raise ValueError("Causal logical token coordinates required")
        if any(type(v) is not int or v < -1 or v > position for v in row["indices"]):
            raise ValueError("Invalid causal candidates")
        key = (row["layer"], row["query_position"])
        if key in self.cache:
            raise ValueError(
                "Duplicate query; speculative rollback needs a new session"
            )
        self.cache[key] = tuple(row["indices"])

    def selection(self, request_id, target_layer, query_position):
        self._check(request_id)
        source = self.sources[target_layer]
        return list(self.cache[(source, query_position)])

    def _check(self, request_id):
        if self.closed or request_id != self.request_id:
            raise ValueError("Closed session or different request")

    def close(self):
        self.cache.clear()
        self.closed = True

    def __enter__(self):
        self._check(self.request_id)
        return self

    def __exit__(self, *_):
        self.close()


def candidate_scores_reference(
    q_quant, k_quant, k_scale, weights, candidates, pool_limits
):
    """Score only supplied logical pools; Q scale is already folded in weights.

    Shapes: Q [queries, heads, dim], K [pools, dim], scales [pools],
    weights [queries, heads], candidates [queries, candidates], limits [queries].
    This is an FP32 reference for the FP8 path, not a production fast kernel.
    """
    import torch

    if q_quant.ndim != 3 or k_quant.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Invalid candidate scorer layout")
    queries, heads, dim = q_quant.shape
    if (
        k_quant.shape[1] != dim
        or weights.shape != (queries, heads)
        or k_scale.shape != (k_quant.shape[0],)
        or candidates.shape[0] != queries
        or pool_limits.shape != (queries,)
    ):
        raise ValueError("Candidate scorer dimensions differ")
    if candidates.dtype not in (torch.int32, torch.int64) or pool_limits.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Logical pool IDs and limits must be integer tensors")
    if k_quant.shape[0] == 0:
        raise ValueError("At least one pooled key required")
    if bool(((pool_limits < 0) | (pool_limits > k_quant.shape[0])).any()):
        raise ValueError("Invalid causal pool limits")
    if bool(((candidates < -1) | (candidates >= k_quant.shape[0])).any()):
        raise ValueError("Invalid logical pool IDs")
    valid = (candidates >= 0) & (candidates < pool_limits[:, None])
    ids = candidates.clamp_min(0).long()
    keys = k_quant[ids].float() * k_scale[ids, None]
    # No [queries, full_context] score matrix is constructed.
    logits = torch.einsum("qhd,qcd->qch", q_quant.float(), keys).relu()
    return (logits * weights[:, None, :]).sum(-1).masked_fill(~valid, float("-inf"))


def select_candidate_pools(scores, candidates, count):
    """Checked reference selection: deduplicate, rank, pad; ties use pool ID."""
    import torch

    if (
        scores.ndim != 2
        or scores.shape != candidates.shape
        or scores.device != candidates.device
        or candidates.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Matching score/ID matrices required")
    if type(count) is not int or count < 1:
        raise ValueError("Positive pool selection count required")
    if bool((candidates < -1).any()) or bool(
        (torch.isnan(scores) | torch.isposinf(scores)).any()
    ):
        raise ValueError("Invalid candidate ID or score")
    output = torch.full(
        (scores.shape[0], count), -1, dtype=candidates.dtype, device=candidates.device
    )
    if candidates.shape[1] == 0:
        return output
    ids, order = torch.sort(candidates, dim=-1, stable=True)
    ordered_scores = scores.gather(1, order)
    duplicate = torch.zeros_like(ids, dtype=torch.bool)
    duplicate[:, 1:] = ids[:, 1:] == ids[:, :-1]
    inconsistent = (
        duplicate[:, 1:]
        & (ids[:, 1:] >= 0)
        & (ordered_scores[:, 1:] != ordered_scores[:, :-1])
    )
    if bool(inconsistent.any()):
        raise ValueError("Duplicate pool IDs have different scores")
    ordered_scores = ordered_scores.masked_fill(duplicate | (ids < 0), float("-inf"))
    ranking = torch.argsort(ordered_scores, dim=-1, descending=True, stable=True)[
        :, :count
    ]
    selected = ids.gather(1, ranking)
    selected = selected.masked_fill(
        ~torch.isfinite(ordered_scores.gather(1, ranking)), -1
    )
    output[:, : selected.shape[1]] = selected
    return output


def expand_pools_reference(pool_ids, seq_lens, kpool=4):
    """Expand complete pools and append this query's unfinished tail once.

    Output is logical tokens with -1 padding; backend alignment is external.
    """
    import torch

    if (
        type(kpool) is not int
        or kpool < 1
        or pool_ids.ndim != 2
        or seq_lens.shape != (pool_ids.shape[0],)
        or pool_ids.device != seq_lens.device
        or pool_ids.dtype not in (torch.int32, torch.int64)
        or seq_lens.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Integer pool IDs and per-query lengths required")
    if bool((pool_ids < -1).any()) or bool((seq_lens < 0).any()):
        raise ValueError("Invalid pool IDs or lengths")
    complete = seq_lens // kpool
    valid = (pool_ids >= 0) & (pool_ids < complete[:, None])
    tokens = pool_ids[:, :, None] * kpool + torch.arange(kpool, device=pool_ids.device)
    tokens = tokens.masked_fill(~valid[:, :, None], -1).flatten(1)
    tail = complete[:, None] * kpool + torch.arange(kpool - 1, device=pool_ids.device)
    tail = tail.masked_fill(tail >= seq_lens[:, None], -1)
    return torch.cat((tokens, tail), dim=1)
