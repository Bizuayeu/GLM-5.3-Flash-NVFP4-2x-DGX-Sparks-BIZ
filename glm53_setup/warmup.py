"""Post-readiness request ladder that compiles kernels before interactive use.

The pinned launch disables vLLM's own JIT warmup, so a prompt shape seen for the
first time compiles while serving. On this kit that compile burst once pushed the
head below its memory reserve (docs/vision.md). The rungs come from the shapes
that were observed compiling during serving: a short text turn, a tool call, one
image and the longest prompt the operator intends to serve. A profile that
admits more than one sequence also decodes with several at once, a shape one
rung at a time never reaches, so bursts of 2..max_num_seqs concurrent requests
follow the rungs.
"""

import base64
import math
import re
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

from .server_config import optional

COMPILED = re.compile(r"JIT compilation during inference: (.+?)\. This causes")
ANSWER_TOKENS = 32  # Enough decode steps to reach the sampling kernels.
LONG_LINE = "warmup line {index}.\n"


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
    return ladder


def burst_widths(profile):
    """One burst per concurrent width the profile admits; none at one sequence."""
    return list(range(2, profile["context"]["max_num_seqs"] + 1))


def burst_request(width, member):
    """A short turn per member; the texts differ so each sequence decodes its own."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Reply with the word ready. ({member + 1} of {width})",
            }
        ],
        "max_tokens": ANSWER_TOKENS,
    }


def threads(calls):
    """Run the calls at once; each result, or the exception it raised."""

    def guarded(call):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - a member's failure is its result
            return error

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(guarded, calls))


def burst(width, ask, gather, clock):
    """Send width requests together through gather; the row records every member."""
    row = {"burst": width}
    started = clock()
    try:
        results = gather(
            [lambda m=member: ask(burst_request(width, m)) for member in range(width)]
        )
        members = []
        for result in results:
            if isinstance(result, Exception):
                members.append({"status": "failed", "error": type(result).__name__})
                continue
            members.append(
                {
                    "status": "ok",
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "finish_reason": result["choices"][0]["finish_reason"],
                }
            )
        if len(members) != width:
            raise ValueError("A burst returned a different number of results")
        row["members"] = members
        row["status"] = "ok" if all(m["status"] == "ok" for m in members) else "failed"
    except Exception as error:  # noqa: BLE001 - every burst is recorded, none aborts the ladder
        row["status"] = "failed"
        row["error"] = type(error).__name__
    row["seconds"] = round(clock() - started, 3)
    return row


def run(profile, *, ask, count_tokens, logs, clock, reset=None, gather=threads):
    """Send every rung through ask(request); return the record, never raise.

    ask, count_tokens, logs and gather (which runs a burst's requests at once)
    are injected so the CPU tests exercise the ladder without a server. reset,
    when given, drops the warmup prefixes afterwards.
    """
    before = compiled_kernels(logs())
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
            result = ask(request)
            row["prompt_tokens"] = result["usage"]["prompt_tokens"]
            row["finish_reason"] = result["choices"][0]["finish_reason"]
            row["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - every rung is recorded, none aborts the ladder
            row["status"] = "failed"
            row["error"] = type(error).__name__
        row["seconds"] = round(clock() - started, 3)
        rows.append(row)
    bursts = [burst(width, ask, gather, clock) for width in burst_widths(profile)]
    after = compiled_kernels(logs())
    record = {
        "rungs": rows,
        "bursts": bursts,
        "compiled_during_warmup": sorted(after - before),
        "compiled_before_warmup": sorted(before),
        "prefix_cache_reset": False,
    }
    if reset is not None:
        try:
            reset()
            record["prefix_cache_reset"] = True
        except Exception as error:  # noqa: BLE001 - reset failure is evidence, not a ladder failure
            record["prefix_cache_reset_error"] = type(error).__name__
    record["passed"] = all(row["status"] == "ok" for row in rows + bursts)
    return record
