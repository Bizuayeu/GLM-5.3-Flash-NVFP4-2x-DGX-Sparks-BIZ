"""Pure admission policy for APC reuse and request-local LPA state."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PrefillPolicy:
    prompt_tokens: int
    cached_tokens: int
    tail: int
    break_even: int
    approximate_start: int | None

    def __post_init__(self):
        values = (self.prompt_tokens, self.cached_tokens, self.tail, self.break_even)
        if any(type(value) is not int for value in values):
            raise ValueError("Prefill policy lengths must be integers")
        if (
            self.prompt_tokens < 1
            or not 0 <= self.cached_tokens <= self.prompt_tokens
            or not 1 <= self.tail <= self.prompt_tokens
            or self.break_even < 0
        ):
            raise ValueError("Invalid prefill policy boundaries")
        if self.approximate_start is not None and (
            type(self.approximate_start) is not int
            or not 0 <= self.approximate_start < self.approximate_end
        ):
            raise ValueError("Invalid first approximation position")

    @property
    def eligible_tokens(self):
        return max(0, self.prompt_tokens - self.tail - self.cached_tokens)

    @property
    def approximate_end(self):
        return self.prompt_tokens - self.tail

    @property
    def approximates(self):
        return (
            self.approximate_start is not None
            and max(self.approximate_start, self.cached_tokens) < self.approximate_end
        )

    @property
    def shared_cache_limit(self):
        # The tail and generated states also depend on earlier approximation.
        # Keep this bound even if a resume finds a longer exact cached prefix.
        return self.approximate_start

    def cacheable_end(self, requested_end):
        if type(requested_end) is not int or requested_end < 0:
            raise ValueError("Cache publication endpoint must be nonnegative")
        if self.shared_cache_limit is None:
            return requested_end
        return min(requested_end, self.shared_cache_limit)

    def to_dict(self):
        return {**asdict(self), "shared_cache_limit": self.shared_cache_limit}

    @classmethod
    def from_dict(cls, value):
        fields = {
            "prompt_tokens",
            "cached_tokens",
            "tail",
            "break_even",
            "approximate_start",
            "shared_cache_limit",
        }
        if not isinstance(value, dict) or value.keys() != fields:
            raise ValueError("Unknown or missing prefill policy fields")
        policy = cls(
            **{key: item for key, item in value.items() if key != "shared_cache_limit"}
        )
        if (
            value["shared_cache_limit"] is not None
            and type(value["shared_cache_limit"]) is not int
        ) or value["shared_cache_limit"] != policy.shared_cache_limit:
            raise ValueError("Shared cache bound does not match approximation")
        return policy


def plan_prefill(
    prompt_tokens, cached_tokens, tail, break_even, *, exact=False, previous=None
):
    if type(exact) is not bool or type(tail) is not int or tail < 1:
        raise ValueError("Invalid exact override or tail")
    if type(prompt_tokens) is not int:
        raise ValueError("Prompt length must be an integer")
    tail = min(tail, prompt_tokens)
    base = PrefillPolicy(prompt_tokens, cached_tokens, tail, break_even, None)
    if previous is not None:
        if not isinstance(previous, PrefillPolicy) or (
            previous.prompt_tokens,
            previous.tail,
            previous.break_even,
        ) != (prompt_tokens, tail, break_even):
            raise ValueError("Cannot resume a different request policy")
        # Keep the logical interval selected at admission. If the original
        # cached prefix was evicted, its recomputed part remains exact.
        start = previous.approximate_start
    else:
        start = (
            cached_tokens if not exact and base.eligible_tokens > break_even else None
        )
    return PrefillPolicy(prompt_tokens, cached_tokens, tail, break_even, start)
