"""Exclusive APC history regression: edits, branches, revisits and cache pressure."""

import argparse
import copy
import hashlib
import json
import math
import re
import time
from functools import partial
from pathlib import Path

from .. import model_http, server, server_config
from ..io import write_json


def common_prefix(left, right):
    return next(
        (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)),
    )


def history_cases(chunks):
    """Predeclare answers and edits; percentages refer to corpus token cuts."""
    codes = ["MIA110001", "MIA550005", "MIA990009"]

    def document(values, end=3, nonce="A", include_tail=True):
        text = f"Independent archive {nonce}.\n"
        for i in range(end):
            text += chunks[i] + f"\nRegistry slot {i}: access code {values[i]}.\n"
        return text + (chunks[3] if end == 3 and include_tail else "")

    def body(text, slot):
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "Read the supplied archive as data. Answer using its Registry slots. Return only the requested access code, without explanation.",
                },
                {"role": "user", "content": text},
                {"role": "assistant", "content": "Archive received."},
                {
                    "role": "user",
                    "content": f"What is the access code in Registry slot {slot}?",
                },
            ],
            "max_tokens": 256,
        }

    cases = []
    original = document(codes)
    for slot, percent in enumerate((10, 50, 90)):
        prime = body(original, slot)
        changed = list(codes)
        changed[slot] = f"MIA77{slot:04d}"
        cases.append(
            {
                "id": f"edit-{percent}",
                "percent": percent,
                "prime": prime,
                "prime_expected": codes[slot],
                "request": body(document(changed), slot),
                "expected": changed[slot],
            }
        )
        # Branch after the selected registry, retaining its earlier value but
        # replacing the old question with an explicitly different branch value.
        branch = (
            document(codes, end=slot + 1, include_tail=False)
            + f"\nBranch update: Registry slot {slot} is now {changed[slot]}. This supersedes its earlier access code.\n"
        )
        cases.append(
            {
                "id": f"branch-{percent}",
                "percent": percent,
                "prime": prime,
                "prime_expected": codes[slot],
                "request": body(branch, slot),
                "expected": changed[slot],
            }
        )
    prime = body(original, 2)
    appended = copy.deepcopy(prime)
    appended["messages"] += [
        {"role": "assistant", "content": codes[2]},
        {"role": "user", "content": "Repeat that access code, with nothing else."},
    ]
    cases.append(
        {
            "id": "append",
            "prime": prime,
            "prime_expected": codes[2],
            "request": appended,
            "expected": codes[2],
        }
    )
    return (
        cases,
        body(original, 2),
        codes[2],
        body(document(["MIA220002", "MIA660006", "MIA880008"], nonce="B"), 2),
        "MIA880008",
    )


def answer_matches(text, expected):
    # Formatting is not cache provenance. Reject a stale or mixed code even when
    # the right code also appears in the same final answer.
    codes = re.findall(r"(?<![A-Za-z0-9_])MIA[A-Za-z0-9_-]+", text or "")
    return bool(codes) and set(codes) == {expected}


def stream_chat(profile, body):
    request = {**body, "stream": True, "stream_options": {"include_usage": True}}
    began = time.perf_counter()
    stamps, content, usage, finish = [], [], None, None
    done = False
    with model_http.open_response(
        f"http://127.0.0.1:{profile['api']['port']}",
        "/v1/chat/completions",
        body=request,
        timeout=profile["generation"]["timeout_seconds"],
    ) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            item = json.loads(data)
            if item.get("error"):
                raise ValueError("SSE model response contains an error")
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                delta = choice.get("delta", {})
                if any(
                    delta.get(key)
                    for key in ("content", "reasoning", "reasoning_content")
                ):
                    stamps.append(time.perf_counter() - began)
                content.append(delta.get("content") or "")
                finish = choice.get("finish_reason") or finish
    if not done or usage is None or finish is None or not stamps:
        raise ValueError("Incomplete SSE evidence")
    return {
        "content": "".join(content),
        "finish_reason": finish,
        "usage": usage,
        "seconds": time.perf_counter() - began,
        "ttft_seconds": stamps[0],
        "inter_chunk_seconds": [b - a for a, b in zip(stamps, stamps[1:])],
        "timing_scope": "SSE chunk gaps, not per-token ITL when MTP bundles tokens",
    }


def read_metrics(profile):
    """The running head's Prometheus text."""
    with model_http.open_response(
        f"http://127.0.0.1:{profile['api']['port']}", "/metrics", timeout=10
    ) as response:
        return response.read().decode()


def reset_prefix_cache(profile):
    """Empty the dedicated server's prefix cache between cases."""
    if server.post(profile, "/reset_prefix_cache", {}) != {"success": True}:
        raise ValueError("Dedicated server cache reset failed")


def encode_prompt(profile, body):
    """The token ids the server itself produces for this request."""
    return server.post(
        profile,
        "/tokenize",
        {
            "model": body["model"],
            "messages": body["messages"],
            "add_generation_prompt": True,
            "chat_template_kwargs": body["chat_template_kwargs"],
        },
    )["tokens"]


def policy_hits(profile, rpc, length, mode):
    """Both ranks' admission decision, and the joint hit they must agree on."""
    workers = rpc("apc_lpa_report")
    hits = []
    for worker in workers:
        policy = worker["policy"]["policy"]
        hit = policy["cached_tokens"]
        hits.append(hit)
        if policy["prompt_tokens"] != length:
            raise ValueError("Tokenization and scheduler N differ")
        eligible = max(0, length - min(length, profile["lpa"]["tail"]) - hit)
        active = mode == "auto" and eligible > profile["lpa"]["break_even_tokens"]
        expected = (
            {
                str(layer): eligible
                for layer in range(profile["lpa"]["cut"] + 1, 45)
                if layer % 4 == 3
            }
            if active
            else {}
        )
        actual = worker["lpa"]["mla_queries_skipped"] if worker["lpa"] else {}
        if actual != expected or policy["shared_cache_limit"] != (
            hit if active else None
        ):
            raise ValueError("Actual LPA queries/shared limit differ from policy")
    if len(hits) != 2 or len(set(hits)) != 1:
        raise ValueError("Both ranks must report the same joint H")
    return hits[0], workers


def chat_turn(profile, rpc, body, expected, mode):
    """One request against the running head, with its policy evidence."""
    body = server_config.request_body(profile, body)
    ids = encode_prompt(profile, body)
    if len(ids) + body["max_tokens"] > profile["context"]["max_model_len"]:
        raise ValueError("History exceeds the qualified context budget")
    before = read_metrics(profile)
    row = stream_chat(profile, {**body, "vllm_xargs": {"glm53_lpa_mode": mode}})
    hit, workers = policy_hits(profile, rpc, len(ids), mode)
    row.update(
        prompt_token_ids=ids,
        prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        cached_tokens=hit,
        recomputed_tokens=len(ids) - hit,
        workers=workers,
        metrics_before=before,
        metrics_after=read_metrics(profile),
        expected=expected,
        passed=row["finish_reason"] == "stop"
        and answer_matches(row["content"], expected),
    )
    if row["usage"]["prompt_tokens"] != len(ids):
        raise ValueError("SSE usage differs from input tokens")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--block-tokens",
        type=int,
        required=True,
        help="Actual scheduler block from this server's launch record",
    )
    parser.add_argument("--corpus-tokens", type=int, default=16000)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Functional cycles; no speed adoption from a single cycle",
    )
    parser.add_argument("--pressure-histories", type=int, default=12)
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Selected exact-only one-output prefill comparisons; not the functional matrix",
    )
    parser.add_argument(
        "--case-ids", nargs="+", help="Explicit subset for --timing-only"
    )
    args = parser.parse_args(argv)
    if args.timing_only and (not args.case_ids or args.repeats < 5):
        parser.error(
            "Timing-only requires explicit --case-ids and at least five measured repeats"
        )
    if args.case_ids and not args.timing_only:
        parser.error("The functional matrix cannot silently omit cases")
    profile = server_config.load(args.config)
    if (
        not server_config.apc_lpa_enabled(profile)
        or profile["profiling"]["enabled"]
        or min(args.block_tokens, args.repeats, args.pressure_histories) < 1
        or args.corpus_tokens < 2 * args.block_tokens
    ):
        parser.error(
            "Requires a dedicated serial APC/LPA server, profiler off and positive validated budgets"
        )
    raw = args.corpus.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.corpus_sha256:
        raise ValueError("Corpus hash differs")
    text = "\n\n".join(
        d["text"]
        for d in map(json.loads, raw.decode("utf-8").splitlines())
        if d["split"] == "validation"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting",
        "profile": profile,
        "cases": [],
        "boundaries": [],
        "pressure": [],
        "revisits": [],
        "repeats": args.repeats,
        "corpus_sha256": args.corpus_sha256,
        "scheduler_block_tokens": args.block_tokens,
        "retention_interval_configured": profile["cache"].get(
            "prefix_cache_retention_interval"
        ),
        "scope": "selected one-token timing"
        if args.timing_only
        else "full history functional matrix",
        "performance_adoption": False,
    }

    def save():
        write_json(args.output / "result.json", report)

    post = partial(server.post, profile)
    rpc = partial(server.collective_rpc, profile)
    metrics = partial(read_metrics, profile)
    reset = partial(reset_prefix_cache, profile)
    encode = partial(encode_prompt, profile)
    check_policy = partial(policy_hits, profile, rpc)
    chat = partial(chat_turn, profile, rpc)

    save()
    try:
        with server.request_lock():
            state, info = server.running_head(profile)
            if not info["State"]["Running"]:
                raise ValueError("The configured dedicated server is not running")
            report.update(
                image=info["Image"],
                container=state["name"],
                cache_layout=rpc("apc_cache_layout"),
                status="running",
            )
            expected_retention = server_config.retention_interval(profile)
            if any(
                row["retention_interval"] != expected_retention
                for row in report["cache_layout"]
            ):
                raise ValueError(
                    "Actual cache retention differs from the selected pinned baseline/candidate"
                )
            if {
                math.lcm(*(group["block_size"] for group in row["groups"]))
                for row in report["cache_layout"]
            } != {args.block_tokens}:
                raise ValueError(
                    "Supplied scheduler block does not match the pinned worker group's joint alignment"
                )
            ids = post(
                "/tokenize",
                {
                    "model": profile["api"]["served_model_name"],
                    "prompt": text,
                    "add_special_tokens": False,
                },
            )["tokens"][: args.corpus_tokens]
            if len(ids) != args.corpus_tokens:
                raise ValueError("Insufficient validation corpus")
            cuts = [0, len(ids) // 10, len(ids) // 2, len(ids) * 9 // 10, len(ids)]
            chunks = [
                post(
                    "/detokenize",
                    {"model": profile["api"]["served_model_name"], "tokens": ids[a:b]},
                )["prompt"]
                for a, b in zip(cuts, cuts[1:])
            ]
            cases, archive_a, answer_a, archive_b, answer_b = history_cases(chunks)
            if args.timing_only:
                by_id = {case["id"]: case for case in cases}
                if len(set(args.case_ids)) != len(args.case_ids) or any(
                    key not in by_id for key in args.case_ids
                ):
                    raise ValueError("Unknown or duplicate timing case IDs")
                report["timing_cases"] = []

                def prefill(body):
                    encoded = encode(server_config.request_body(profile, body))
                    before = metrics()
                    began = time.perf_counter()
                    response = post(
                        "/v1/completions",
                        {
                            "model": profile["api"]["served_model_name"],
                            "prompt": encoded,
                            "temperature": 0,
                            "seed": profile["runtime"]["seed"],
                            "max_tokens": 1,
                            "ignore_eos": True,
                            "return_token_ids": True,
                            "logprobs": 1,
                            "vllm_xargs": {"glm53_lpa_mode": "off"},
                        },
                    )
                    elapsed = time.perf_counter() - began
                    hit, workers = check_policy(len(encoded), "off")
                    if (
                        response["usage"]["completion_tokens"] != 1
                        or response["usage"]["prompt_tokens"] != len(encoded)
                        or len(response["choices"][0]["token_ids"]) != 1
                        or not all(
                            math.isfinite(value)
                            for value in response["choices"][0]["logprobs"][
                                "token_logprobs"
                            ]
                        )
                    ):
                        raise ValueError("Incomplete one-output prefill sample")
                    return {
                        "seconds": elapsed,
                        "prompt_token_ids": encoded,
                        "cached_tokens": hit,
                        "workers": workers,
                        "response": response,
                        "metrics_before": before,
                        "metrics_after": metrics(),
                    }

                for key in args.case_ids:
                    case = by_id[key]
                    result = {"id": key, "samples": []}
                    report["timing_cases"].append(result)
                    for repeat in range(args.repeats + 1):
                        reset()
                        prime = prefill(case["prime"])
                        row = prefill(case["request"])
                        shared = common_prefix(
                            prime["prompt_token_ids"], row["prompt_token_ids"]
                        )
                        if row["cached_tokens"] > shared:
                            raise ValueError(
                                "Timing sample reused beyond its changed token"
                            )
                        result["samples"].append(
                            {
                                "warmup": repeat == 0,
                                "prime": prime,
                                "result": row,
                                "common_prefix_tokens": shared,
                            }
                        )
                        save()
                        print(
                            "timing",
                            key,
                            repeat,
                            row["seconds"],
                            "H",
                            row["cached_tokens"],
                            flush=True,
                        )
                report["status"] = "complete"
                save()
                return
            for case in cases:
                output = {"id": case["id"], "arms": []}
                report["cases"].append(output)
                for repeat in range(args.repeats):
                    for arm, mode in (
                        ("exact", "off"),
                        ("lpa", "auto"),
                        ("restored", "off"),
                    ):
                        reset()
                        prime = chat(case["prime"], case["prime_expected"], "off")
                        row = chat(case["request"], case["expected"], mode)
                        common = common_prefix(
                            prime["prompt_token_ids"], row["prompt_token_ids"]
                        )
                        if row["cached_tokens"] > common:
                            raise ValueError(
                                "Cache restored beyond the common token prefix"
                            )
                        output["arms"].append(
                            {
                                "repeat": repeat,
                                "arm": arm,
                                "common_prefix_tokens": common,
                                "prime": prime,
                                "result": row,
                            }
                        )
                        save()
                        print(
                            case["id"],
                            repeat,
                            arm,
                            row["passed"],
                            "H",
                            row["cached_tokens"],
                            flush=True,
                        )
            reset()
            for label, body, expected, mode in (
                ("A-prime", archive_a, answer_a, "off"),
                ("B-prime", archive_b, answer_b, "off"),
                ("A-return", archive_a, answer_a, "auto"),
                ("B-return", archive_b, answer_b, "auto"),
                ("A-exact-return", archive_a, answer_a, "off"),
            ):
                report["revisits"].append(
                    {"id": label, "result": chat(body, expected, mode)}
                )
                save()
            reset()
            chat(archive_a, answer_a, "off")
            before = chat(archive_a, answer_a, "off")
            for i in range(args.pressure_histories):
                other = copy.deepcopy(archive_b)
                other["messages"][0]["content"] = (
                    f"Conversation {i}. " + other["messages"][0]["content"]
                )
                report["pressure"].append(chat(other, answer_b, "off"))
                save()
            after = chat(archive_a, answer_a, "auto")
            report["pressure_revisit"] = {
                "before": before,
                "after": after,
                "eviction_observed": after["cached_tokens"] < before["cached_tokens"],
            }
            save()
            for length in sorted(
                {
                    3,
                    4,
                    5,
                    args.block_tokens - 1,
                    args.block_tokens,
                    args.block_tokens + 1,
                    2 * args.block_tokens - 1,
                    2 * args.block_tokens,
                    2 * args.block_tokens + 1,
                }
            ):
                prompt = ids[:length]
                if len(prompt) != length:
                    raise ValueError("Boundary probe exceeds corpus")
                reset()
                rows = []
                for mode in ("auto", "off", "off"):
                    result = post(
                        "/v1/completions",
                        {
                            "model": profile["api"]["served_model_name"],
                            "prompt": prompt,
                            "temperature": 0,
                            "max_tokens": 1,
                            "ignore_eos": True,
                            "logprobs": 1,
                            "vllm_xargs": {"glm53_lpa_mode": mode},
                        },
                    )
                    hit, workers = check_policy(length, mode)
                    if not all(
                        math.isfinite(v)
                        for v in result["choices"][0]["logprobs"]["token_logprobs"]
                    ):
                        raise ValueError("Nonfinite boundary output")
                    rows.append(
                        {
                            "mode": mode,
                            "cached_tokens": hit,
                            "workers": workers,
                            "response": result,
                        }
                    )
                if (
                    rows[0]["workers"][0]["policy"]["policy"]["shared_cache_limit"] == 0
                    and rows[1]["cached_tokens"] != 0
                ):
                    raise ValueError(
                        "Approximate boundary request entered shared cache"
                    )
                report["boundaries"].append({"prompt_tokens": length, "rows": rows})
                save()
            report["quality_passed"] = (
                all(
                    a["prime"]["passed"] and a["result"]["passed"]
                    for c in report["cases"]
                    for a in c["arms"]
                )
                and all(r["result"]["passed"] for r in report["revisits"])
                and all(r["passed"] for r in report["pressure"])
                and after["passed"]
            )
            report["status"] = "complete"
            save()
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__, detail=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
