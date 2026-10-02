"""Cold/warm prefix-cache correctness gate on one long shared prefix.

``tools/check_prefix_cache.py`` shows whether a repeat hits the cache; this
checks whether what the hit restores is right. N short tasks share one long
synthetic log whose answers are fixed by its seed. They are asked in parallel
under a fresh ``cache_salt`` (cold: the first request computes the prefix, the
others may read it while or after it is written), then again under the same
salt (warm: every request should hit), and every answer is checked against the
log. A task the cold pass answered and the warm pass got wrong is read as a
broken prefix cache; a warm pass that reports no cached tokens did not test
the cache and never passes.

Adapted from knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4, ``bench/prefix_scan.py``
at 770d1153062aa916b06591411c61d5c593ea0f03 (MIT, Copyright (c) 2026 knapcio;
notice in LICENSES/knapcio-MIT.txt): the record format, the lookup and quote
tasks and their parsing. Local changes: exact answers at temperature 0
instead of sampled rounds, the server's own chat tokenization for the length,
a paired cold/warm verdict, and no logprob drift, degenerate-run or block-phase
checks.

Nothing here talks to a server; the caller passes ``ask`` and ``count_tokens``.
"""

import random
import re
from concurrent.futures import ThreadPoolExecutor

# Upper bounds on the prompt, in the server's chat tokens. Long: the ~99K-token
# log that reproduced knapcio's issue #2. Short: a length measured to restore
# 9,216 tokens on the 4,608-token block (docs/server-configuration.md); below
# two blocks a repeat restores nothing.
LENGTHS = {"long": 99_000, "short": 14_025}
DEFAULT_LENGTH = "long"
SEED = 7  # knapcio's default; the log is a function of it.
# The lookups near the end reach 60 records back.
MIN_RECORDS = 64
# knapcio's estimate of one record's tokens; only the first sizing guess.
TOKENS_PER_RECORD = 32
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ"
STATES = ("idle", "dead", "drained", "leased", "stale")
RECORD = (
    "Record {:05d} host=node-{:02d} state={} retries={} at t={}s ref={};"
    " lease held then released."
)
QUOTED = re.compile(r"Record \d{5} host=[^\n]*?released\.")
LOOKED_UP = re.compile(r"Record\s*(\d{1,5})\s*[:=\-]?\s*(?:ref=)?\s*([A-Z]{5})\b")


class Prefix:
    """``records`` log lines; each ref code is random, so a right one was read."""

    def __init__(self, records, seed):
        draw = random.Random(seed)
        self.seed = seed
        self.codes = [
            "".join(draw.choice(ALPHABET) for _ in range(5)) for _ in range(records)
        ]
        self.lines = [
            RECORD.format(
                i,
                i % 29,
                STATES[draw.randrange(len(STATES))],
                draw.randrange(6),
                7 * i,
                self.codes[i],
            )
            for i in range(records)
        ]
        self.text = "\n".join(self.lines)


def build_prefix(records, seed):
    return Prefix(records, seed)


def tasks(prefix):
    """Four lookups spread over the log, including its end, and two quotes."""
    n = len(prefix.lines)

    def at(share):
        return int(share * n)

    rows = [
        {
            "kind": "lookup",
            "targets": (a, b),
            "question": f"Give the ref code of Record {a:05d} and of Record {b:05d}, "
            "one per line exactly as 'Record N: CODE'. Output only those two lines.",
        }
        for a, b in (
            (at(0.10), at(0.35)),
            (at(0.60), at(0.85)),
            (n - 60, n - 25),
            (n - 40, n - 8),
        )
    ]
    middle = at(0.5)
    rows.append(
        {
            "kind": "quote",
            "targets": (middle, middle + 1),
            "question": f"Quote verbatim, exactly as they appear, the lines for "
            f"Record {middle:05d} and Record {middle + 1:05d}, one per line.",
        }
    )
    rows.append(
        {
            "kind": "quote",
            "targets": (n - 3, n - 2, n - 1),
            "question": "Quote verbatim the last three lines of the log, one per line.",
        }
    )
    return rows


def request(prefix, task, salt):
    return {
        "messages": [
            {"role": "user", "content": prefix.text + "\n\n" + task["question"]}
        ],
        "temperature": 0,
        "cache_salt": salt,
    }


def answer(prefix, task):
    """The one correct reply."""
    if task["kind"] == "lookup":
        return "\n".join(f"Record {i:05d}: {prefix.codes[i]}" for i in task["targets"])
    return "\n".join(prefix.lines[i] for i in task["targets"])


def verify(prefix, task, content):
    """Correct means every asked fact is present and nothing stated is false."""
    known = set(prefix.lines)
    mismatches = [line for line in QUOTED.findall(content) if line not in known]
    found = {}
    for number, code in LOOKED_UP.findall(content):
        index = int(number)
        found[index] = code
        if index >= len(prefix.codes):
            mismatches.append(f"Record {index:05d}: {code} (no such record)")
        elif code != prefix.codes[index]:
            mismatches.append(
                f"Record {index:05d}: {code} (truth {prefix.codes[index]})"
            )
    if task["kind"] == "lookup":
        present = all(found.get(i) == prefix.codes[i] for i in task["targets"])
    else:
        present = all(prefix.lines[i] in content for i in task["targets"])
    return {"correct": present and not mismatches, "mismatches": mismatches}


def fit_prefix(count_tokens, limit, seed):
    """The longest log whose every task prompt stays within ``limit`` tokens."""

    def longest(records):
        prefix = build_prefix(records, seed)
        return prefix, max(count_tokens(request(prefix, t, "")) for t in tasks(prefix))

    records = max(MIN_RECORDS, limit // TOKENS_PER_RECORD)
    prefix, used = longest(records)
    records = max(1, records * limit // used)
    prefix, used = longest(records)
    while used > limit and records > 1:
        records = max(1, min(records - 1, records * limit // used))
        prefix, used = longest(records)
    if used > limit or records < MIN_RECORDS:
        raise ValueError(f"{limit} tokens is too short for {MIN_RECORDS} records")
    return prefix, used


def ask_all(ask, prefix, salt, parallel):
    """One phase: ``parallel`` requests at once, task ``i`` of the list cycled."""
    listed = tasks(prefix)
    chosen = [listed[i % len(listed)] for i in range(parallel)]

    def one(task):
        row = {"kind": task["kind"], "request": task["question"]}
        try:
            reply = ask(request(prefix, task, salt))
            content = reply["choices"][0]["message"].get("content") or ""
            usage = reply["usage"]
        except Exception as error:  # noqa: BLE001 - a failed request is recorded, not a pass
            return {**row, "error": type(error).__name__, "cached_tokens": None}
        details = usage.get("prompt_tokens_details") or {}
        return {
            **row,
            **verify(prefix, task, content),
            "reply": content,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
        }

    with ThreadPoolExecutor(parallel) as pool:
        return list(pool.map(one, chosen))


def phase_summary(rows):
    correct = sum(bool(row.get("correct")) for row in rows)
    return {
        "correct": correct,
        "accuracy": correct / len(rows),
        "cached_tokens": [row["cached_tokens"] for row in rows],
        "prompt_tokens": [row.get("prompt_tokens") for row in rows],
        "failures": [
            {"index": i, **{k: v for k, v in row.items() if k != "correct"}}
            for i, row in enumerate(rows)
            if not row.get("correct")
        ],
    }


def judge(cold, warm):
    """Reasons, in a fixed order; only a warm-only error is a cache failure."""
    reasons = []
    if any("error" in row for row in cold + warm):
        reasons.append("request_error")
    if any("error" not in row and not row["correct"] for row in cold):
        reasons.append("cold_incorrect")
    if any(
        c.get("correct") and "error" not in w and not w["correct"]
        for c, w in zip(cold, warm)
    ):
        reasons.append("prefix_cache_corruption")
    if any("error" not in row and not row["cached_tokens"] for row in warm):
        reasons.append("no_cache_hit")
    verdict = (
        "fail"
        if "prefix_cache_corruption" in reasons
        else "inconclusive"
        if reasons
        else "pass"
    )
    return reasons, verdict


def run(ask, count_tokens, *, max_prompt_tokens, salt, seed=SEED, parallel=None):
    """Size the log, ask every task cold then warm under one salt; never raise on answers."""
    prefix, used = fit_prefix(count_tokens, max_prompt_tokens, seed)
    parallel = parallel or len(tasks(prefix))
    cold = ask_all(ask, prefix, salt, parallel)
    warm = ask_all(ask, prefix, salt, parallel)
    reasons, verdict = judge(cold, warm)
    return {
        "prefix": {
            "seed": seed,
            "records": len(prefix.lines),
            "max_prompt_tokens": max_prompt_tokens,
            "prompt_tokens": used,
        },
        "cache_salt": salt,
        "parallel": parallel,
        "cold": phase_summary(cold),
        "warm": phase_summary(warm),
        "reasons": reasons,
        "verdict": verdict,
        "passed": verdict == "pass",
    }
