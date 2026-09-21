"""Replay a draft-depth policy over recorded verification steps (``GLM53_DEPTH_TRACE``).

A trace line says: at output position ``p`` the step checked ``d`` drafts and the first ``a`` held.
That fixes, per output token, whether the draft for it held: tokens ``p+1..p+a`` did, token
``p+a+1`` did not when ``a < d``, and when ``a == d`` it was the target's own token and no draft
for it was ever checked (unknown). A replayed step at position ``q`` and depth ``d'`` accepts the
run of held tokens after ``q``, up to ``d'``, and moves to the token after it.

Exact: a replayed step that starts where a recorded step started, at a depth the record reached
(prefix acceptance). Approximate: a shallower policy lands between recorded starts, and there the
replay treats "the draft for this token held" as a property of the token, while the engine's draft
depends on where the step started and at which draft depth the token came up. Unknown tokens come
after a fully accepted step, so where acceptance is high they are common and calling them all
misses makes every depth below the recorded one look far worse than it is. They are therefore
filled in once per request (``unknown="impute"``): a seeded draw at the request's own rate of "held,
given the token before it held". ``"miss"`` and ``"hit"`` bracket that. Timing is the cost table, not
the recorded clock, and a different depth can change the text itself on this model (BF16 ties),
which no replay sees.
"""

import json
import random

from glm53_setup.runtime.adaptive_depth import BASE_MS, STEP_MS, DepthPolicy


def read_trace(lines):
    """Request id -> steps in order, each ``(position, drafted, accepted)``."""
    requests = {}
    for line in lines:
        if line.strip():
            request, _step, position, drafted, accepted, *_rest = json.loads(line)
            requests.setdefault(request, []).append((position, drafted, accepted))
    return requests


def token_hits(steps, unknown="impute", seed=0):
    """Output token index -> whether the draft for it held; see the module text for ``unknown``."""
    if unknown not in ("impute", "miss", "hit"):
        raise ValueError("unknown must be impute, miss or hit")
    hits, open_tokens, held, failed = {}, [], 0, 0
    for position, drafted, accepted in steps:
        for offset in range(1, accepted + 1):
            hits[position + offset] = True
        if accepted < drafted:
            hits[position + accepted + 1] = False
            failed += accepted > 0
        else:
            open_tokens.append(position + accepted + 1)
        held += max(accepted - 1, 0)
    rate = {"miss": 0.0, "hit": 1.0}.get(unknown, held / max(held + failed, 1))
    rng = random.Random(seed)
    for token in open_tokens:
        hits[token] = rng.random() < rate
    return hits


def span(steps):
    position, _drafted, accepted = steps[-1]
    return steps[0][0], position + accepted + 1


def cost_table(ceiling, base_ms=BASE_MS, step_ms=STEP_MS, overrides=None):
    table = {depth: base_ms + step_ms * depth for depth in range(1, ceiling + 1)}
    table.update(overrides or {})
    return table


class Fixed:
    def __init__(self, depth):
        self.depth = depth

    def next_depth(self):
        return self.depth

    def observe(self, drafted, accepted):
        pass


def replay(steps, policy, costs, unknown="impute"):
    """Walk the recorded text with ``policy``; returns tokens, steps, ms and the depths used."""
    hits = token_hits(steps, unknown)
    position, end = span(steps)
    start, count, ms, depths = position, 0, 0.0, {}
    while position < end:
        # Like the engine, never draft past the last token of the request.
        depth = max(1, min(policy.next_depth(), end - position - 1))
        accepted = 0
        while accepted < depth and hits.get(position + accepted + 1, False):
            accepted += 1
        policy.observe(depth, accepted)
        position += accepted + 1
        count += 1
        ms += costs[depth]
        depths[depth] = depths.get(depth, 0) + 1
    return {
        "tokens": position - start,
        "steps": count,
        "ms": ms,
        "tokens_per_second": 1000.0 * (position - start) / ms,
        "mean_depth": sum(d * n for d, n in depths.items()) / count,
        "depths": dict(sorted(depths.items())),
    }


# Candidates to compare on recorded sequences. cc-defer: the numbers below are placeholders that
# make each variant distinct, not choices; pick them on the tuning inputs' traces and confirm on
# the evaluation inputs before any of them replaces the policy that ran on the pair.
CANDIDATES = {
    "current": {},
    "slow-ema": {"alpha": 1 / 32},
    "band": {"band": 0.2},
    "probe-next": {"probe": "next", "probe_every": 4},
    "band+probe-next": {"band": 0.2, "probe": "next", "probe_every": 4},
}


def candidate(name, ceiling, base_ms=BASE_MS, step_ms=STEP_MS):
    return DepthPolicy(ceiling, base_ms=base_ms, step_ms=step_ms, **CANDIDATES[name])


def compare(steps, ceiling, costs, base_ms=BASE_MS, step_ms=STEP_MS, unknown="impute"):
    """Every fixed depth and every candidate over one request's steps."""
    rows = {
        f"fixed-{depth}": replay(steps, Fixed(depth), costs, unknown)
        for depth in range(1, ceiling + 1)
    }
    for name in CANDIDATES:
        policy = candidate(name, ceiling, base_ms, step_ms)
        rows[name] = replay(steps, policy, costs, unknown)
    return rows
