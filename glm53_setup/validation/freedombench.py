"""Run pinned FreedomBench questions against the explicitly selected local GLM."""

import argparse
import ast
import hashlib
import json
import os
import random
import time
import urllib.error
from collections import Counter
from pathlib import Path

from .. import server, server_config
from ..config import DEFAULT_PROFILE, ROOT, load_lock
from ..io import read_json, write_json
from .freedom_scoring import classify_attempt, summarize

# The upstream reasoning budget per answer; hle.py uses the same.
REASONING_BUDGET = 8192

# Adapted from the pinned Apache-2.0 upstream runner; see NOTICE.
SYSTEM_PROMPT = (
    "You are taking a multiple-choice quiz of basic factual questions. "
    "Choose the single best answer. You may think briefly first, but you must "
    "end your reply with a line in exactly this format and nothing after it:\n"
    "ANSWER: X\nwhere X is one of A, B, C, or D."
)
# Outcomes that leave a type E case without a complete visible reply.
UNANSWERED = ("transport_error", "protocol_error", "empty_final", "truncated")
INSTRUCTION = "Answer with a single letter (A, B, C, or D)."


def parse_questions(directory):
    """The pinned upstream items as literals, without executing upstream Python."""
    lock = read_json(ROOT / "config/freedombench.lock.json")
    data = (directory / "freedombench/questions.py").read_bytes()
    if hashlib.sha256(data).hexdigest() != lock["questions_sha256"]:
        raise ValueError("FreedomBench question revision/hash mismatch")
    tree = ast.parse(data.decode("utf-8"))
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "QUESTIONS"
    )
    items = []
    for call in assignment.value.elts:
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "q"
            or call.keywords
        ):
            raise ValueError("Unexpected question representation")
        qid, topic, question, correct, distractors, source = [
            ast.literal_eval(a) for a in call.args
        ]
        items.append(
            {
                "id": qid,
                "topic": topic,
                "question": question,
                "correct": correct,
                "distractors": list(distractors),
                "source": source,
            }
        )
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Question IDs must be unique")
    return lock, items


def render(qid, question, correct, distractors, instruction):
    """The upstream prompt; the ID seeds the order, so every language shares it."""
    options = [correct, *distractors]
    seed = int.from_bytes(hashlib.sha256(qid.encode()).digest()[:8], "big")
    random.Random(seed).shuffle(options)
    prompt = "\n".join(
        [
            question,
            "",
            *[f"{letter}) {text}" for letter, text in zip("ABCD", options)],
            "",
            instruction,
        ]
    )
    return prompt, "ABCD"[options.index(correct)]


def load_questions(directory):
    lock, items = parse_questions(directory)
    questions = []
    for item in items:
        prompt, answer = render(
            item["id"],
            item["question"],
            item["correct"],
            item["distractors"],
            INSTRUCTION,
        )
        questions.append(
            {
                "id": item["id"],
                "topic": item["topic"],
                "prompt": prompt,
                "answer": answer,
                "source": item["source"],
            }
        )
    return lock, questions


def load_translation(directory, path):
    """The pinned suite in a reviewed translation, checked against its lock.

    Each item keeps the upstream id, English text and distractor order, so the
    A-D order and answer letter equal the English ones. Only the answer line
    format `ANSWER: X` stays English, so the upstream extractor still scores it.
    """
    lock, items = parse_questions(directory)
    ja_lock = read_json(ROOT / "config/freedombench-ja.lock.json")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ja_lock["translation_sha256"]:
        raise ValueError("Translation hash mismatch")
    if ja_lock["questions_sha256"] != lock["questions_sha256"]:
        raise ValueError("The translation lock pins other questions")
    doc = json.loads(data)
    if doc.get("teacher_excluded") is not True:
        raise ValueError("A translation must be marked teacher_excluded")
    if doc.get("translation_version") != ja_lock["translation_version"]:
        raise ValueError("Translation version mismatch")
    if [t["id"] for t in doc["questions"]] != [item["id"] for item in items]:
        raise ValueError("Translation IDs differ from the pinned questions")
    system_prompt, instruction = doc.get("system_prompt"), doc.get("instruction")
    if not system_prompt or not instruction or "ANSWER: X" not in system_prompt:
        raise ValueError("Translation prompts must keep the ANSWER: X line")
    questions = []
    for item, t in zip(items, doc["questions"]):
        en, ja = t["en"], t["ja"]
        upstream = [item["question"], item["correct"], item["distractors"]]
        if [en["question"], en["correct"], en["distractors"]] != upstream:
            raise ValueError(f"{item['id']}: English text differs from upstream")
        if (
            not ja["question"]
            or not ja["correct"]
            or len(ja["distractors"]) != 3
            or not all(ja["distractors"])
        ):
            raise ValueError(f"{item['id']}: incomplete translation")
        prompt, answer = render(
            item["id"], ja["question"], ja["correct"], ja["distractors"], instruction
        )
        if answer != render(item["id"], *upstream, INSTRUCTION)[1]:
            raise ValueError(f"{item['id']}: answer letter differs from English")
        questions.append(
            {
                "id": item["id"],
                "topic": item["topic"],
                "prompt": prompt,
                "answer": answer,
                "source": item["source"],
                "system_prompt": system_prompt,
                "language": doc["translation_version"],
            }
        )
    return {**lock, "translation": ja_lock}, questions


def condition_key(case):
    """One run per condition; a control's counter arm (C) runs with the S arm."""
    arm = "S" if case["arm"] == "C" else case["arm"]
    return f"{case['case_type']}-{arm}-{case['position']}-{case['language']}"


def load_cases(path, expected_sha256, condition):
    """The cases of one condition from a derived case file pinned by its hash.

    Type M keeps the upstream system prompt and answer format; type E carries
    its own prompts and has no answer key (it is scored by hand).
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Case file hash mismatch")
    doc = json.loads(data)
    if doc.get("teacher_excluded") is not True:
        raise ValueError("A derived case file must be marked teacher_excluded")
    cases = [c for c in doc["cases"] if condition_key(c) == condition]
    if not cases or [c["id"] for c in cases] != doc["conditions"].get(condition):
        raise ValueError(f"Condition {condition} does not match the case file")
    questions = []
    for case in cases:
        if case["case_type"] == "M":
            if case["system_prompt"] != SYSTEM_PROMPT or case["answer"] not in (
                "A",
                "B",
                "C",
                "D",
            ):
                raise ValueError(f"{case['id']}: type M must keep the upstream quiz")
        elif case["case_type"] != "E" or case["answer"] is not None:
            raise ValueError(f"{case['id']}: unknown case type or an E answer key")
        questions.append(
            {
                "id": case["id"],
                "topic": case["topic"],
                "case_type": case["case_type"],
                "arm": case["arm"],
                "position": case["position"],
                "language": case["language"],
                "system_prompt": case["system_prompt"],
                "prompt": case["user"],
                "answer": case["answer"],
                "document_sha256": case["document_sha256"],
                "designated_distractor": case["designated_distractor"],
            }
        )
    meta = {
        "path": str(path),
        "sha256": expected_sha256,
        "condition": condition,
        "case": doc.get("case"),
        "benchmark": doc.get("benchmark"),
    }
    return meta, questions


def summarize_open(questions, results):
    """Type E: whether each case got a complete visible reply; scoring is by hand."""
    outcomes = Counter(
        classify_attempt(row["attempts"][-1])["outcome"]
        for row in results
        if row["attempts"]
    )
    failed = sum(outcomes[k] for k in UNANSWERED)
    answered = sum(outcomes.values()) - failed
    return {
        "planned": len(questions),
        "received": len(results),
        "answered": answered,
        "outcomes": dict(outcomes),
        "valid_complete_run": answered == len(questions),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, help="The pinned upstream suite")
    parser.add_argument(
        "--cases", type=Path, help="A derived case file instead of the upstream suite"
    )
    parser.add_argument(
        "--cases-sha256", help="The case file hash fixed before any result was seen"
    )
    parser.add_argument(
        "--condition", help="One condition of the case file, e.g. M-S-middle-en"
    )
    parser.add_argument(
        "--translation",
        type=Path,
        help="A reviewed translation of the pinned suite, e.g. FB-04 Japanese",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, help="Pilot only; never a full-suite score"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=REASONING_BUDGET,
        help="Upstream reasoning budget; must fit context",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Upstream initial plus four no-choice retries",
    )
    args = parser.parse_args(argv)
    if args.translation and (
        args.benchmark_dir is None or args.cases or args.limit is not None
    ):
        parser.error(
            "--translation needs --benchmark-dir, and takes no --cases or --limit"
        )
    if (args.benchmark_dir is None) == (args.cases is None):
        parser.error("Give either --benchmark-dir or --cases")
    if args.cases and (not args.cases_sha256 or not args.condition or args.limit):
        parser.error(
            "--cases needs --cases-sha256 and --condition, and takes no --limit"
        )
    if os.name != "posix":
        parser.error("Run evaluations on the Linux model host; scoring is CPU-portable")
    if (
        args.max_tokens < 1
        or args.max_attempts < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Budgets and limit must be positive")
    profile = server_config.load(args.config)
    if args.cases:
        meta, selected = load_cases(args.cases, args.cases_sha256, args.condition)
        lock = meta["benchmark"]
    elif args.translation:
        lock, selected = load_translation(args.benchmark_dir, args.translation)
    else:
        lock, questions = load_questions(args.benchmark_dir)
        selected = questions[: args.limit] if args.limit else questions
    open_ended = selected[0].get("case_type") == "E"
    current, info = server.running_head(profile)
    if not info["State"]["Running"]:
        parser.error("The configured local server is not running")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "benchmark": lock,
        "model_lock": load_lock(),
        "profile": profile,
        "image_id": info["Image"],
        "pilot": bool(args.limit or args.cases),
        "questions": selected,
        "results": [],
        "max_attempts": args.max_attempts,
        "max_tokens": args.max_tokens,
        "language": selected[0].get("language", "original-English"),
        "system_prompt": None
        if args.cases
        else selected[0].get("system_prompt", SYSTEM_PROMPT),
        "human_refusal_audit": "not performed",
    }
    if args.cases:
        # Derived cases are a separate extension, never mixed with the upstream score.
        report.update(cases=meta, condition=args.condition, teacher_excluded=True)
    if args.translation:
        # A full run, but a separate score from the original English one.
        report["teacher_excluded"] = True

    def save():
        score = summarize_open if open_ended else summarize
        report["summary"] = score(selected, report["results"])
        report["full_suite_complete"] = (
            not report["pilot"] and report["summary"]["valid_complete_run"]
        )
        write_json(args.output / "result.json", report)

    with server.request_lock():
        save()
        try:
            for question in selected:
                row = {"id": question["id"], "attempts": []}
                report["results"].append(row)
                for _ in range(args.max_attempts):
                    began = time.monotonic()
                    try:
                        response = server.ask(
                            profile,
                            {
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": question.get(
                                            "system_prompt", SYSTEM_PROMPT
                                        ),
                                    },
                                    {"role": "user", "content": question["prompt"]},
                                ],
                                "max_tokens": args.max_tokens,
                            },
                        )
                    except (
                        urllib.error.URLError,
                        TimeoutError,
                        ConnectionError,
                    ) as error:
                        response = {"error": type(error).__name__}
                        if getattr(error, "code", None) in (401, 403):
                            response.update(
                                error="authentication failed", http_status=error.code
                            )
                    response["elapsed_seconds"] = time.monotonic() - began
                    row["attempts"].append(response)
                    save()
                    # Type E is not a choice question: no no-choice retries.
                    if (
                        open_ended
                        or response.get("error")
                        or classify_attempt(response)["choice"] is not None
                    ):
                        break
                save()
                print(question["id"], classify_attempt(row["attempts"][-1]), flush=True)
            report["status"] = "complete"
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            save()
    if not report["summary"]["valid_complete_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
