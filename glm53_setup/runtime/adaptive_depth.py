"""Per-request MTP draft depth from recent acceptance; the configured depth is the ceiling.

``S_i`` is the share of a request's recent verification steps whose first ``i`` drafts were all
accepted, kept as an exponential moving average and updated only for the positions a step really
drafted. A step at depth ``d`` costs ``base_ms + step_ms * d`` and yields ``1 + S_1 + ... + S_d``
tokens, so depth ``i`` raises tokens per second exactly while

    S_i > step_ms * (1 + S_1 + ... + S_(i-1)) / (base_ms + step_ms * (i - 1)).

The policy climbs from the floor (1 unless configured) while that holds. Every estimate starts at
1.0, so a request starts at the ceiling (the fixed-depth behaviour) and comes down only on
evidence. Positions past the current depth get no evidence, so every ``probe_every``-th step
drafts the full depth.

Off unless ``GLM53_ADAPTIVE_DEPTH=1``. This module imports nothing from vLLM: the pinned scheduler
calls ``batch_depth`` while it builds a step and ``observe`` where it counts accepted drafts
(``patch_adaptive_depth.py``).
"""

import itertools
import math
import os

# Step-time fit on the reference pair (2026-09-21, records Track B stage 0): a decode step is
# about 47 ms plus 13.5 ms per draft depth, k = 2..5.
BASE_MS = 47.0
STEP_MS = 13.5
# cc-defer: alpha and the probe cadence come from replaying the measured S_i vectors through this
# policy (alpha 1/16..1/4 and cadence 8..32 all stay within 5% of the best fixed depth per input;
# 1/8 and 16 are the middle), not from a run on the model. Settle them on the tuning inputs before
# the policy becomes a default. The linear cost is about 7% optimistic at depth 1 (measured 66 ms).
ALPHA = 0.125
PROBE_EVERY = 16

_ATTRIBUTE = "glm53_adaptive_depth"


def enabled():
    value = os.environ.get("GLM53_ADAPTIVE_DEPTH", "0")
    if value not in ("0", "1"):
        raise ValueError("GLM53_ADAPTIVE_DEPTH must be 0 or 1")
    return value == "1"


def _number(name, default, minimum, integer=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw) if integer else float(raw)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a number of at least {minimum}")
    return value


def settings():
    """Costs and cadence from the environment; ``probe_every`` 0 means never probe."""
    alpha = _number("GLM53_ADAPTIVE_DEPTH_ALPHA", ALPHA, 0.0)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("GLM53_ADAPTIVE_DEPTH_ALPHA must be in (0, 1]")
    return {
        "base_ms": _number("GLM53_ADAPTIVE_DEPTH_BASE_MS", BASE_MS, 0.0),
        "step_ms": _number("GLM53_ADAPTIVE_DEPTH_STEP_MS", STEP_MS, 0.0),
        "alpha": alpha,
        "probe_every": _number(
            "GLM53_ADAPTIVE_DEPTH_PROBE_EVERY", PROBE_EVERY, 0, integer=True
        ),
        "floor": _number("GLM53_ADAPTIVE_DEPTH_MIN", 1, 1, integer=True),
    }


def worthwhile_depth(shares, base_ms, step_ms, floor=1):
    """Deepest depth reached by climbing from the floor while the next draft pays for itself."""
    # S_i cannot exceed S_(i-1). A position past the current depth keeps its old estimate while
    # the shallower ones fall, so each share is read through the running minimum; otherwise an
    # untouched 1.0 behind a fallen S_(i-1) sends the depth back up without any evidence.
    shares = list(itertools.accumulate(shares, min))
    depth = min(floor, len(shares))
    expected_tokens = 1.0 + sum(shares[:depth])
    for index in range(depth, len(shares)):
        step_time = base_ms + step_ms * index
        if shares[index] * step_time <= step_ms * expected_tokens:
            break
        depth = index + 1
        expected_tokens += shares[index]
    return depth


class DepthPolicy:
    def __init__(
        self,
        ceiling,
        base_ms=BASE_MS,
        step_ms=STEP_MS,
        alpha=ALPHA,
        probe_every=PROBE_EVERY,
        floor=1,
    ):
        if ceiling < 1:
            raise ValueError("The depth ceiling must be at least 1")
        self.ceiling = ceiling
        self.base_ms = base_ms
        self.step_ms = step_ms
        self.alpha = alpha
        self.probe_every = probe_every
        self.floor = floor
        self.shares = [1.0] * ceiling
        self.steps = 0

    def observe(self, drafted, accepted):
        """One verification step: ``drafted`` positions were checked, the first ``accepted`` held."""
        for index in range(min(drafted, self.ceiling)):
            hit = 1.0 if accepted > index else 0.0
            self.shares[index] += self.alpha * (hit - self.shares[index])

    def next_depth(self):
        self.steps += 1
        if self.probe_every and self.steps % self.probe_every == 0:
            return self.ceiling
        return worthwhile_depth(self.shares, self.base_ms, self.step_ms, self.floor)


def _policy(request, ceiling):
    policy = getattr(request, _ATTRIBUTE, None)
    if policy is None:
        policy = DepthPolicy(ceiling, **settings())
        setattr(request, _ATTRIBUTE, policy)
    return policy


def batch_depth(requests, ceiling):
    """Depth to draft in this step: the deepest any scheduled request asks for.

    The pinned scheduler carries one depth per step (``num_spec_tokens_to_schedule``); with one
    sequence that is the request's own depth.
    """
    depths = [_policy(request, ceiling).next_depth() for request in requests]
    return max(depths, default=ceiling)


def observe(request, ceiling, drafted, accepted):
    _policy(request, ceiling).observe(drafted, accepted)
