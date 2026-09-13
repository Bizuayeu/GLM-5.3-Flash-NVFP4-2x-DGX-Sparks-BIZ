"""Scheduler-side admission and exact-only prefix publication boundaries."""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path

from .apc_policy import PrefillPolicy, plan_prefill

POLICY_KEY = "_glm53_apc_lpa_policy"
STATE_ATTR = "_glm53_apc_lpa_state"
MODE_KEY = "glm53_lpa_mode"
_ALLOCATION_OPTIONS = (
    "num_new_computed_tokens",
    "new_computed_blocks",
    "num_lookahead_tokens",
    "num_external_computed_tokens",
    "delay_cache_blocks",
    "num_encoder_tokens",
    "full_sequence_must_fit",
    "reserved_blocks",
    "has_scheduled_reqs",
)


@dataclass(frozen=True)
class RuntimeSettings:
    cut: int
    tail: int
    break_even: int
    projector_path: str
    projector_sha256: str
    skip_mla_queries: bool

    def __post_init__(self):
        if any(type(v) is not int for v in (self.cut, self.tail, self.break_even)):
            raise ValueError("LPA policy settings must use integer token boundaries")
        if self.cut < 0 or self.tail < 1 or self.break_even < 0:
            raise ValueError("Invalid LPA policy settings")
        if not isinstance(
            self.projector_path, str
        ) or not self.projector_path.startswith("/"):
            raise ValueError("Projector must have an absolute container path")
        if not isinstance(self.projector_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.projector_sha256
        ):
            raise ValueError("Invalid projector digest")
        if type(self.skip_mla_queries) is not bool:
            raise ValueError("skip_mla_queries must be boolean")

    @property
    def signature(self):
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True).encode()
        ).hexdigest()


@lru_cache(maxsize=4)
def _parse_settings(raw):
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return RuntimeSettings(**value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid APC/LPA environment configuration") from error


def settings():
    return _parse_settings(os.environ.get("GLM53_APC_LPA_CONFIG", ""))


@dataclass(frozen=True)
class RequestState:
    request_id: str
    policy: PrefillPolicy
    signature: str
    override: str

    def wire(self):
        return {
            "version": 1,
            "request_id": self.request_id,
            "policy": self.policy.to_dict(),
            "signature": self.signature,
            "override": self.override,
        }


def audit(event, **data):
    path = os.environ.get("GLM53_APC_LPA_AUDIT")
    if path:
        # Opt-in correctness diagnostics only; leave unset for latency trials.
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **data}, sort_keys=True) + "\n")


def request_state(request, *, required=False):
    state = getattr(request, STATE_ATTR, None)
    if state is None:
        if required:
            raise ValueError("Cache publication lacks an admitted LPA policy")
        return None
    if not isinstance(state, RequestState) or state.request_id != request.request_id:
        raise ValueError("Request policy owner mismatch")
    return state


def allocation_guard(original):
    """Keep allocation intact; install a provisional cap before its cache calls."""

    @wraps(original)
    def allocate(manager, request, num_new_tokens, *args, **kwargs):
        config = settings()
        if config is None:
            return original(manager, request, num_new_tokens, *args, **kwargs)
        if len(args) > len(_ALLOCATION_OPTIONS):
            raise TypeError("Unexpected pinned allocation signature")
        options = dict(zip(_ALLOCATION_OPTIONS, args))
        if options.keys() & kwargs.keys():
            raise TypeError("Duplicate allocation arguments")
        options.update(kwargs)
        restored_new = options.get("num_new_computed_tokens", 0)
        if (
            type(restored_new) is not int
            or restored_new < 0
            or type(request.num_computed_tokens) is not int
            or request.num_computed_tokens < 0
        ):
            raise ValueError("Invalid computed-token boundary")
        if (
            not manager.enable_caching
            or options.get("num_external_computed_tokens", 0)
            or options.get("delay_cache_blocks", False)
            or options.get("num_encoder_tokens", 0)
        ):
            raise ValueError("APC/LPA requires locally restored text cache")
        if (
            request.sampling_params is None
            or request.prompt_embeds is not None
            or request.mm_features
            or request.lora_request is not None
        ):
            raise ValueError("APC/LPA currently supports plain text without LoRA")
        extra = request.sampling_params.extra_args or {}
        override = extra.get(MODE_KEY, "auto")
        if override not in ("auto", "off"):
            raise ValueError("glm53_lpa_mode must be auto or off")
        previous = request_state(request)
        if previous is None and POLICY_KEY in extra:
            raise ValueError("Internal LPA policy cannot be supplied by the client")
        if previous is not None and (
            previous.signature != config.signature or previous.override != override
        ):
            raise ValueError("Cannot change a live request's computation policy")
        if (
            previous is not None
            and request.num_prompt_tokens != previous.policy.prompt_tokens
        ):
            raise ValueError("Streaming input changes are unsupported for APC/LPA")
        if (
            previous is not None
            and POLICY_KEY in extra
            and extra[POLICY_KEY] != previous.wire()
        ):
            raise ValueError("Internal request policy was modified")
        if previous is None or request.num_computed_tokens == 0:
            if previous is None and request.num_computed_tokens:
                raise ValueError("A resumed request lacks its admission policy")
            restored = request.num_computed_tokens + restored_new
            if previous is None and restored > request.num_prompt_tokens:
                raise ValueError("Restored prefix exceeds a new request's prompt")
            policy = plan_prefill(
                request.num_prompt_tokens,
                min(restored, request.num_prompt_tokens),
                config.tail,
                config.break_even,
                exact=override == "off",
                previous=previous.policy if previous else None,
            )
            candidate = RequestState(
                request.request_id, policy, config.signature, override
            )
        else:
            candidate = previous
        setattr(request, STATE_ATTR, candidate)
        # An unexpected exception keeps the conservative cap. A normal failed
        # admission has not computed/published new state and may retry with H'.
        result = original(manager, request, num_new_tokens, *args, **kwargs)
        if result is None:
            if previous is None:
                delattr(request, STATE_ATTR)
            else:
                setattr(request, STATE_ATTR, previous)
            return None
        if candidate is not previous or POLICY_KEY not in extra:
            request.sampling_params.extra_args = {**extra, POLICY_KEY: candidate.wire()}
            audit("admitted", **candidate.wire())
        return result

    return allocate


def publication_end(request, requested_end):
    state = request_state(request, required=settings() is not None)
    return state.policy.cacheable_end(requested_end) if state else requested_end


def validate_publication(request, new_blocks, first_block, block_size, block_mask):
    """Fail before a full-block hash is inserted if any caller bypasses the cap."""
    state = request_state(request, required=settings() is not None)
    if state is None:
        return
    checked = []
    for index, block in enumerate(new_blocks):
        if block.is_null or (block_mask is not None and not block_mask[index]):
            continue
        end = (first_block + index + 1) * block_size
        if state.policy.cacheable_end(end) != end:
            raise ValueError("Attempt to share state after the first approximation")
        checked.append(end)
    audit(
        "full_publication_checked",
        request_id=request.request_id,
        block_ends=checked,
        limit=state.policy.shared_cache_limit,
    )


def validate_partial_publication(request, num_tokens):
    if publication_end(request, num_tokens) != num_tokens:
        raise ValueError("Attempt to share an approximate partial checkpoint")
    audit("partial_publication_checked", request_id=request.request_id, end=num_tokens)
