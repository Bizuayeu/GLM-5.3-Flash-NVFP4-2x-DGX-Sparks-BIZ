"""Statistics over a ``GLM53_DRAFT_OBSERVE`` record (``runtime/draft_observe.py`` has the fields).

A draft at depth i is only meaningful when depths 1..i-1 were accepted (otherwise it continues a
wrong prefix), so every per-depth figure is over the steps that *reached* that depth, and "held"
means the target's argmax equals the draft there.

The speed projections treat recorded steps as independent samples: a rule that drafts ``d'`` on a
recorded step gets ``1 + min(accepted, d')`` tokens for ``cost[d']``. A shallower rule would start
its next step elsewhere than the record did; that shift is ignored here, for the rule and for the
fixed depths alike, so the comparison between them is like for like but not a run.
"""

import json

BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0000001)
# Target top-1 minus top-2 logit at a rejected position. BF16 logits of magnitude 8 to 32 step
# by 1/64 to 1/8 (records REPORT, 2026-09-21), so up to 0.125 is within a step or two of a tie.
MARGINS = (("tie", 0.0), ("near-tie", 0.125), ("uncertain", 1.0))


def read(lines):
    return [json.loads(line) for line in lines if line.strip()]


def samples(rows, depth):
    """``(draft probability, held)`` for the steps that reached ``depth`` (1-based)."""
    return [
        (row["dp"][depth - 1], row["a"] >= depth)
        for row in rows
        if len(row["d"]) >= depth and row["a"] >= depth - 1
    ]


def auc(pairs):
    """Probability that a held draft outscores a rejected one (ties count half); None if one-sided."""
    ordered = sorted(pairs)
    held = sum(1 for _, label in ordered if label)
    rejected = len(ordered) - held
    if not held or not rejected:
        return None
    wins, below, index = 0.0, 0, 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        tied = ordered[index:end]
        tied_rejected = sum(1 for _, label in tied if not label)
        tied_held = len(tied) - tied_rejected
        wins += tied_held * (below + tied_rejected / 2)
        below += tied_rejected
        index = end
    return wins / (held * rejected)


def calibration(pairs, bins=BINS):
    table = []
    for low, high in zip(bins, bins[1:]):
        inside = [(score, label) for score, label in pairs if low <= score < high]
        if inside:
            table.append(
                {
                    "from": low,
                    "to": min(high, 1.0),
                    "count": len(inside),
                    "mean_probability": sum(s for s, _ in inside) / len(inside),
                    "held": sum(1 for _, label in inside if label) / len(inside),
                }
            )
    return table


def margin_class(margin):
    for name, limit in MARGINS:
        if margin <= limit:
            return name
    return "confident"


def rejections(rows, depth):
    """Where the first rejection fell at ``depth``: the target's margin and the draft's top-5."""
    classes, ranks, stale = {}, {}, 0
    for row in rows:
        if len(row["d"]) < depth or row["a"] != depth - 1:
            continue
        if row["d5"][depth - 1][0] != row["d"][depth - 1]:
            stale += 1
            continue
        name = margin_class(row["tm"][depth - 1])
        classes[name] = classes.get(name, 0) + 1
        top = row["d5"][depth - 1]
        wanted = row["t"][depth - 1]
        rank = str(top.index(wanted) + 1) if wanted in top else ">5"
        ranks[rank] = ranks.get(rank, 0) + 1
    return {"target_margin": classes, "target_rank_in_draft": ranks, "stale": stale}


def cost_table(ceiling, base_ms=47.0, step_ms=13.5, overrides=None):
    table = {depth: base_ms + step_ms * depth for depth in range(1, ceiling + 1)}
    # Depth 1 measured about 66 ms on the reference pair; the line says 60.5.
    table.update({1: 66.0} if overrides is None else overrides)
    return table


def gated_depth(row, tau):
    """Draft until a depth's own top-1 probability falls under ``tau``; that draft is kept.

    Its forward is already paid when its confidence is known, so the rule stops *after* it.
    """
    for index, probability in enumerate(row["dp"]):
        if probability < tau:
            return index + 1
    return len(row["dp"])


def project(rows, costs, depth_of):
    tokens = ms = depth_sum = 0
    for row in rows:
        depth = max(1, min(depth_of(row), len(row["d"])))
        tokens += 1 + min(row["a"], depth)
        ms += costs[depth]
        depth_sum += depth
    return {
        "tokens_per_second": 1000.0 * tokens / ms,
        "mean_depth": depth_sum / len(rows),
    }


def speed(rows, costs, taus=(0.3, 0.5, 0.7, 0.8, 0.9, 0.95)):
    ceiling = max(len(row["d"]) for row in rows)
    result = {
        f"fixed-{depth}": project(rows, costs, lambda row, depth=depth: depth)
        for depth in range(1, ceiling + 1)
    }
    for tau in taus:
        result[f"gate-{tau}"] = project(
            rows, costs, lambda row, t=tau: gated_depth(row, t)
        )
    # The bound no rule can beat on these steps: draft exactly one past the accepted run.
    result["oracle"] = project(rows, costs, lambda row: row["a"] + 1)
    return result


def summary(rows, costs):
    ceiling = max(len(row["d"]) for row in rows)
    depths = {}
    for depth in range(1, ceiling + 1):
        pairs = samples(rows, depth)
        if pairs:
            depths[depth] = {
                "reached": len(pairs),
                "held": sum(1 for _, label in pairs if label) / len(pairs),
                "auc": auc(pairs),
                "calibration": calibration(pairs),
                "rejections": rejections(rows, depth),
            }
    return {"steps": len(rows), "depths": depths, "speed": speed(rows, costs)}
