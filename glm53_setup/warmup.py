"""Post-readiness request ladder that compiles kernels before interactive use.

The pinned launch disables vLLM's own JIT warmup, so a prompt shape seen for the
first time compiles while serving. On this kit that compile burst once pushed the
head below its memory reserve (docs/vision.md). The rungs come from the shapes
that were observed compiling during serving: a short text turn (also at the
checkpoint's sampling), a tool call, one image and the longest prompt the
operator intends to serve. A counting canary closes the ladder and judges
whether the launch decodes correctly (docs/operations.md).
"""

import base64
import math
import re
import struct
import zlib

from .server_config import optional

COMPILED = re.compile(r"JIT compilation during inference: (.+?)\. This causes")
ANSWER_TOKENS = 32  # Enough decode steps to reach the sampling kernels.
LONG_LINE = "warmup line {index}.\n"
# The last rung doubles as a correctness canary (upstream MiaAI-Lab recipe #268):
# a launch that answers /health but decodes garbage fails its switch instead of
# serving. Counting to 80 is checkable exactly and long enough that the MTP
# check judges from this rung alone, whatever other rungs a profile runs: on the
# 1.19.0 pair it took 168 tokens and drafted 126 at effort low, 207 and 162 at
# max (records/20260928-upstream-review/obs268), so 256 answers either way.
CANARY_COUNT = 80
CANARY_TOKENS = 256
# Zero accepted drafts only counts over at least this many drafted tokens, as
# upstream's GLM53_WARMUP_CANARY_MIN_DRAFTS.
MIN_DRAFTS = 64
SPEC_METRICS = {
    "vllm:spec_decode_num_draft_tokens_total": "num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "num_accepted_tokens",
}


def parse_metrics(text, names):
    """Sum each named Prometheus sample over its label sets; absent names are omitted."""
    totals = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name = head.split("{", 1)[0]
        if name in names:
            try:
                totals[name] = totals.get(name, 0.0) + float(value)
            except ValueError:
                continue
    return totals


def spec_counters(metrics):
    """The draft and accepted totals from a Prometheus /metrics text."""
    return {SPEC_METRICS[k]: v for k, v in parse_metrics(metrics, SPEC_METRICS).items()}


def compiled_kernels(logs):
    """Kernel names the jit_monitor reported as compiled while serving."""
    return set(COMPILED.findall(logs))


def png_data_url(width=672, height=336, rgb=(255, 140, 0)):
    """A solid-colour PNG built with the standard library; no image dependency.

    The default is the size of the synthetic image the 200K vision checks sent
    (records/20260915-vision-200k), which the processor accepted as 288 tokens.
    """

    def chunk(kind, payload):
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes(rgb) * width
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height, 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def long_prompt(target_tokens, limit_tokens, count_tokens, sample_lines=256):
    """Build a prompt of about target_tokens, never above limit_tokens.

    count_tokens(text) asks the running server's tokenizer, so the estimate
    uses the served chat template rather than a guessed tokens-per-word ratio.
    """
    if target_tokens < 1 or target_tokens > limit_tokens:
        raise ValueError("Long rung must be positive and fit the context")

    def build(lines):
        text = "".join(LONG_LINE.format(index=i) for i in range(lines))
        return text, count_tokens(text)

    sample = "".join(LONG_LINE.format(index=i) for i in range(sample_lines))
    lines = max(1, math.ceil(target_tokens * sample_lines / count_tokens(sample)))
    text, tokens = build(lines)
    # A line costs more once its index grows a digit, so a short sample
    # under-counts: measured 82,006 tokens for a 65,536 target on the reference
    # tokenizer. Rescale against the real count instead of trusting the sample.
    for _ in range(4):
        if not tokens or abs(tokens - target_tokens) <= target_tokens // 50:
            break
        scaled = max(1, int(lines * target_tokens / tokens))
        if scaled == lines:
            break
        lines = scaled
        text, tokens = build(lines)
    while tokens > limit_tokens and lines > 1:
        lines = max(1, math.floor(lines * limit_tokens / tokens) - 1)
        text, tokens = build(lines)
    return text, tokens


def rungs(profile):
    """Ordered (name, request) pairs; the long rung is built at run time."""
    ladder = [
        (
            "text",
            {
                "messages": [{"role": "user", "content": "Reply with the word ready."}],
                "max_tokens": ANSWER_TOKENS,
            },
        ),
        (
            # A client that sends no temperature (ZCode) gets the checkpoint's
            # generation_config, temperature 1.0 and top_p 0.95; its first such
            # request compiled three _topp_sb_* kernels, which the other rungs,
            # sent at the profile's temperature 0, never reach (2026-09-28).
            "sampled",
            {
                "messages": [{"role": "user", "content": "Reply with the word ready."}],
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": ANSWER_TOKENS,
            },
        ),
        (
            "tool",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What time is it in Tokyo? Use the tool.",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "current_time",
                            "description": "Current time in a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "max_tokens": ANSWER_TOKENS,
            },
        ),
    ]
    if optional(profile, "runtime", "vision"):
        ladder.append(
            (
                "image",
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Name the colour."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": png_data_url()},
                                },
                            ],
                        }
                    ],
                    "max_tokens": ANSWER_TOKENS,
                },
            )
        )
    if optional(profile, "generation", "warmup_long_tokens"):
        ladder.append(("long", None))
    ladder.append(
        (
            "canary",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Count from 1 to {CANARY_COUNT}, separated by "
                        "single spaces. Reply with the numbers only.",
                    }
                ],
                "temperature": 0,
                "reasoning_effort": "low",
                "max_tokens": CANARY_TOKENS,
            },
        )
    )
    return ladder


def read_counters(spec_counters):
    try:
        return spec_counters() if spec_counters is not None else None
    except Exception:  # noqa: BLE001 - unreadable metrics leave the verdict open
        return None


def canary_verdict(profile, row, before, after):
    """None where the ladder cannot judge; only a False verdict is degenerate."""
    verdict = {"content": row.get("content"), "finish_reason": row.get("finish_reason")}
    if row["status"] != "ok":
        verdict["answer_ok"] = None
    else:
        numbers = re.split(r"[\s,]+", (row.get("content") or "").strip().rstrip("."))
        verdict["answer_ok"] = (
            numbers == [str(i) for i in range(1, CANARY_COUNT + 1)]
            and row["finish_reason"] == "stop"
        )
    verdict["acceptance_ok"] = None
    if profile["mtp"]["enabled"] and before is not None and after is not None:
        drafted = after.get("num_draft_tokens", 0) - before.get("num_draft_tokens", 0)
        accepted = after.get("num_accepted_tokens", 0) - before.get(
            "num_accepted_tokens", 0
        )
        verdict.update(draft_tokens=drafted, accepted_tokens=accepted)
        if drafted >= MIN_DRAFTS:
            verdict["acceptance_ok"] = accepted > 0
    verdict["degenerate"] = False in (verdict["answer_ok"], verdict["acceptance_ok"])
    return verdict


def run(profile, *, ask, count_tokens, logs, clock, reset=None, spec_counters=None):
    """Send every rung through ask(request); return the record, never raise.

    ask, count_tokens and logs are injected so the CPU tests exercise the ladder
    without a server. reset, when given, drops the warmup prefixes afterwards.
    spec_counters, when given, returns the MTP draft/accepted totals.
    """
    before = compiled_kernels(logs())
    counters = {}
    limit = profile["context"]["max_model_len"] - profile["generation"]["max_tokens"]
    rows = []
    for name, request in rungs(profile):
        row = {"rung": name}
        started = clock()
        try:
            if request is None:
                text, tokens = long_prompt(
                    profile["generation"]["warmup_long_tokens"], limit, count_tokens
                )
                row["built_prompt_tokens"] = tokens
                request = {
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": ANSWER_TOKENS,
                }
            if name == "canary":
                counters["before"] = read_counters(spec_counters)
            result = ask(request)
            if name == "canary":
                counters["after"] = read_counters(spec_counters)
            row["prompt_tokens"] = result["usage"]["prompt_tokens"]
            row["finish_reason"] = result["choices"][0]["finish_reason"]
            if name == "canary":
                row["content"] = result["choices"][0]["message"].get("content")
            row["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - every rung is recorded, none aborts the ladder
            row["status"] = "failed"
            row["error"] = type(error).__name__
        row["seconds"] = round(clock() - started, 3)
        rows.append(row)
    after = compiled_kernels(logs())
    canary = canary_verdict(
        profile, rows[-1], counters.get("before"), counters.get("after")
    )
    record = {
        "rungs": rows,
        "compiled_during_warmup": sorted(after - before),
        "compiled_before_warmup": sorted(before),
        "prefix_cache_reset": False,
        "canary": canary,
        "degenerate": canary.pop("degenerate"),
    }
    if reset is not None:
        try:
            reset()
            record["prefix_cache_reset"] = True
        except Exception as error:  # noqa: BLE001 - reset failure is evidence, not a ladder failure
            record["prefix_cache_reset_error"] = type(error).__name__
    record["passed"] = (
        all(row["status"] == "ok" for row in rows) and not record["degenerate"]
    )
    return record
