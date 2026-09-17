"""Teacher-forced agreement with a reference run, per token.

Quality gates so far (FreedomBench, tool-eval, MTP acceptance) score answers.
None says how often the model's next-token choice still equals the unmodified
server's choice. This module feeds fixed, self-authored texts through
``/v1/completions`` with ``prompt_logprobs`` and records, per position, the
rank and log-probability of the actual next token. A later run on a changed
server (FP8 projections, a different attention kernel, a reduced candidate
set) is compared with the saved reference position by position.

Two readings, kept apart:

* teacher-forced: how often the actual token is rank 1 / within the top five,
  and its mean negative log-probability. This is the number SAGE-style model
  cards report against the native checkpoint.
* agreement: between two runs on the same text, how often the argmax token is
  the same, how much of the top-k overlaps, and how far the actual token's
  log-probability moved.

Nothing here talks to a server or a file; the callers pass senders in.
"""

import math

TOP_K = 5
TEXTS = {
    "ja-prose": (
        "港町の朝は霧から始まる。倉庫の屋根が濡れて光り、荷揚げの合図が鳴るころには、"
        "霧は海のほうへ引いていく。祖父はこの町で四十年、秤の目盛りを読んで暮らした。"
        "魚の重さは一日のうちで変わらないが、値段は変わる。だから帳面には重さだけを書き、"
        "値は別の紙に書いた。混ぜると、あとで自分が信じられなくなるからだという。"
        "私は最初、その言葉を年寄りの癖だと思っていた。帳面を引き継いで三年目に、"
        "台風で市場が二週間閉まった。再開の日、値は前の半分になっていたが、重さの列は"
        "そのまま続いていた。そこで初めて、祖父が何を守っていたのかが分かった。"
        "測ったものと決めたものを、同じ列に並べてはいけない。並べた瞬間に、"
        "どちらが先だったかを誰も覚えていられなくなる。"
    ),
    "en-prose": (
        "The lighthouse keeper kept two logbooks. The first recorded what the "
        "instruments said: barometer, wind, the hour the lamp was lit. The "
        "second recorded what she decided: when to sound the horn, when to send "
        "the boat, when to leave a ship to its own judgment. She never let a "
        "line from one book cross into the other. Visitors found this fussy "
        "until the winter the barometer failed for nine days. The first book "
        "shows a flat line and a note in pencil: instrument suspect. The second "
        "book shows every decision she made in those nine days, each with the "
        "reason she gave herself at the time. Years later an inquiry into a "
        "wreck on the far shoal read both books side by side and cleared her in "
        "an afternoon. A single book would have looked like a woman guessing."
    ),
    "code": (
        "def merge_ledgers(measured, decided):\n"
        '    """Join two ledgers by day without letting either column overwrite the other.\n'
        "\n"
        "    measured: dict day -> weight in kilograms\n"
        "    decided: dict day -> price per kilogram\n"
        "    Returns rows sorted by day; a missing side is recorded as None, never 0.\n"
        '    """\n'
        "    days = sorted(set(measured) | set(decided))\n"
        "    rows = []\n"
        "    for day in days:\n"
        "        weight = measured.get(day)\n"
        "        price = decided.get(day)\n"
        "        total = None if weight is None or price is None else weight * price\n"
        '        rows.append({"day": day, "weight": weight, "price": price, "total": total})\n'
        "    return rows\n"
        "\n"
        "\n"
        "def test_missing_side_stays_none():\n"
        '    rows = merge_ledgers({"d1": 12.0, "d2": 8.5}, {"d1": 300})\n'
        '    assert rows[0]["total"] == 3600.0\n'
        '    assert rows[1]["price"] is None and rows[1]["total"] is None\n'
    ),
    "math": (
        "Let a_1 = 3 and a_{n+1} = 2 a_n - 1 for n >= 1. Claim: a_n = 2^n + 1.\n"
        "Base case: a_1 = 2^1 + 1 = 3, which matches the definition.\n"
        "Inductive step: assume a_n = 2^n + 1. Then\n"
        "a_{n+1} = 2(2^n + 1) - 1 = 2^{n+1} + 2 - 1 = 2^{n+1} + 1,\n"
        "so the formula holds for n + 1. By induction it holds for all n >= 1.\n"
        "Consequence: a_n - 1 = 2^n is a power of two, so a_n is odd for every n,\n"
        "and a_{n+1} - a_n = 2^n, which doubles at each step. The sum of the first\n"
        "n terms is (2^{n+1} - 2) + n, since the geometric part sums to 2^{n+1} - 2\n"
        "and the constant part contributes n.\n"
    ),
}


def parse_prompt_logprobs(prompt_logprobs, prompt_token_ids):
    """vLLM's completions ``prompt_logprobs`` into rows of (token_id, logprob).

    The first entry is None (no prediction for the first token). Each later
    entry maps token-id strings to {"logprob", "rank", "decoded_token"} and
    always contains the actual prompt token, even when it is outside the
    top-k. Rows are sorted by log-probability, highest first. The shape is
    checked, not assumed; a different server would raise here, not mis-score.
    """
    if not isinstance(prompt_logprobs, list) or not prompt_logprobs:
        raise ValueError("prompt_logprobs must be a non-empty list")
    if len(prompt_logprobs) != len(prompt_token_ids):
        raise ValueError("prompt_logprobs and prompt token ids differ in length")
    if prompt_logprobs[0] is not None:
        raise ValueError("The first prompt_logprobs entry must be None")
    rows = []
    for position, entry in enumerate(prompt_logprobs[1:], start=1):
        if not isinstance(entry, dict) or not entry:
            raise ValueError(f"prompt_logprobs[{position}] is not a token map")
        row = []
        for token, detail in entry.items():
            logprob = detail["logprob"]
            if not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
                raise ValueError(f"prompt_logprobs[{position}] has a non-finite value")
            row.append((int(token), float(logprob)))
        row.sort(key=lambda pair: (-pair[1], pair[0]))
        actual = prompt_token_ids[position]
        if all(token != actual for token, _ in row):
            raise ValueError(f"prompt_logprobs[{position}] omits the prompt token")
        rows.append(row)
    return rows


def teacher_forced(rows, prompt_token_ids, top_k=TOP_K):
    """How the actual next token ranks in each predicted distribution."""
    if len(rows) != len(prompt_token_ids) - 1:
        raise ValueError("rows must cover every prompt position but the first")
    ranks, logprobs = [], []
    for row, actual in zip(rows, prompt_token_ids[1:]):
        rank = next(i for i, (token, _) in enumerate(row, start=1) if token == actual)
        ranks.append(rank)
        logprobs.append(next(lp for token, lp in row if token == actual))
    positions = len(rows)
    return {
        "positions": positions,
        "top1": sum(rank == 1 for rank in ranks) / positions,
        f"top{top_k}": sum(rank <= top_k for rank in ranks) / positions,
        "mean_nll": -sum(logprobs) / positions,
        "ranks": ranks,
        "logprobs": logprobs,
    }


def agreement(reference_rows, candidate_rows, top_k=TOP_K):
    """Argmax agreement, top-k overlap and log-probability drift, per position."""
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(
            "Reference and candidate cover a different number of positions"
        )
    if not reference_rows:
        raise ValueError("Nothing to compare")
    argmax_same, overlaps = [], []
    for reference, candidate in zip(reference_rows, candidate_rows):
        argmax_same.append(reference[0][0] == candidate[0][0])
        top_reference = {token for token, _ in reference[:top_k]}
        top_candidate = {token for token, _ in candidate[:top_k]}
        overlaps.append(
            len(top_reference & top_candidate) / len(top_reference | top_candidate)
        )
    positions = len(reference_rows)
    first_divergence = next(
        (i for i, same in enumerate(argmax_same, start=1) if not same), None
    )
    return {
        "positions": positions,
        "argmax_agreement": sum(argmax_same) / positions,
        f"top{top_k}_overlap_mean": sum(overlaps) / positions,
        f"top{top_k}_overlap_min": min(overlaps),
        "first_argmax_divergence": first_divergence,
    }


def drift(reference_logprobs, candidate_logprobs):
    """Largest and mean absolute move of the actual token's log-probability."""
    if len(reference_logprobs) != len(candidate_logprobs) or not reference_logprobs:
        raise ValueError("Log-probability series differ in length or are empty")
    moves = [abs(a - b) for a, b in zip(reference_logprobs, candidate_logprobs)]
    return {
        "max_abs_logprob_move": max(moves),
        "mean_abs_logprob_move": sum(moves) / len(moves),
    }


def score_text(name, prompt_token_ids, response, top_k=TOP_K):
    """One text's teacher-forced reading plus the rows a later run compares against."""
    choice = response["choices"][0]
    rows = parse_prompt_logprobs(choice["prompt_logprobs"], prompt_token_ids)
    forced = teacher_forced(rows, prompt_token_ids, top_k)
    return {
        "name": name,
        "prompt_tokens": len(prompt_token_ids),
        "top1": forced["top1"],
        f"top{top_k}": forced[f"top{top_k}"],
        "mean_nll": forced["mean_nll"],
        "ranks": forced["ranks"],
        "logprobs": forced["logprobs"],
        "rows": rows,
        "prompt_token_ids": list(prompt_token_ids),
    }


def compare_records(reference, candidate, top_k=TOP_K):
    """Compare two ``run`` outputs text by text; texts missing on either side are listed."""
    reference_texts = {row["name"]: row for row in reference["texts"] if "rows" in row}
    candidate_texts = {row["name"]: row for row in candidate["texts"] if "rows" in row}
    shared = sorted(reference_texts.keys() & candidate_texts.keys())
    results = []
    for name in shared:
        left, right = reference_texts[name], candidate_texts[name]
        if left["prompt_token_ids"] != right["prompt_token_ids"]:
            results.append({"name": name, "error": "tokenization differs"})
            continue
        row = {"name": name, "positions": left["prompt_tokens"] - 1}
        row.update(agreement(left["rows"], right["rows"], top_k))
        row.update(drift(left["logprobs"], right["logprobs"]))
        row["top1_reference"], row["top1_candidate"] = left["top1"], right["top1"]
        results.append(row)
    scored = [row for row in results if "error" not in row]
    summary = {
        "texts": results,
        "missing_in_reference": sorted(candidate_texts.keys() - reference_texts.keys()),
        "missing_in_candidate": sorted(reference_texts.keys() - candidate_texts.keys()),
    }
    if scored:
        weights = sum(row["positions"] for row in scored)
        summary["argmax_agreement"] = (
            sum(row["argmax_agreement"] * row["positions"] for row in scored) / weights
        )
        summary["max_abs_logprob_move"] = max(
            row["max_abs_logprob_move"] for row in scored
        )
    return summary


def run(tokenize, complete, texts=None, top_k=TOP_K, repeats=2):
    """Score each text ``repeats`` times through the given senders; never raise.

    ``tokenize(text)`` returns the prompt token ids; ``complete(token_ids, top_k)``
    returns the completions response. The first raw response is kept so the
    record shows the server's actual ``prompt_logprobs`` shape. Repeats of the
    same text inside one run measure the instrument: identical rows are
    expected, and any difference is reported as ``self_agreement``.
    """
    texts = TEXTS if texts is None else texts
    rows, raw_first, self_checks = [], None, []
    for name, text in texts.items():
        scored = []
        for _ in range(repeats):
            try:
                token_ids = tokenize(text)
                response = complete(token_ids, top_k)
                if raw_first is None:
                    raw_first = response
                scored.append(score_text(name, token_ids, response, top_k))
            except Exception as error:  # noqa: BLE001 - a failed request is recorded
                scored.append(
                    {"name": name, "error": f"{type(error).__name__}: {error}"}
                )
        rows.extend(scored)
        good = [row for row in scored if "error" not in row]
        if len(good) >= 2:
            check = {"name": name}
            check.update(agreement(good[0]["rows"], good[1]["rows"], top_k))
            check.update(drift(good[0]["logprobs"], good[1]["logprobs"]))
            self_checks.append(check)
    errors = [row for row in rows if "error" in row]
    return {
        "top_k": top_k,
        "repeats": repeats,
        "texts": rows,
        "self_agreement": self_checks,
        "errors": len(errors),
        "first_raw_response": raw_first,
        "passed": not errors
        and bool(self_checks)
        and all(check["argmax_agreement"] == 1.0 for check in self_checks),
    }
