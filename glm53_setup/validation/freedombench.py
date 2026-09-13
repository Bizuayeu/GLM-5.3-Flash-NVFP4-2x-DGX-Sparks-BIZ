"""Run pinned FreedomBench questions against the explicitly selected local GLM."""

import argparse
import ast
import hashlib
import os
import random
import time
import urllib.error
from pathlib import Path

from .. import startup, startup_config
from ..config import ROOT, load_lock
from ..io import write_json
from .freedom_scoring import classify_attempt, summarize

# Adapted from the pinned Apache-2.0 upstream runner; see NOTICE.
SYSTEM_PROMPT = (
    "You are taking a multiple-choice quiz of basic factual questions. "
    "Choose the single best answer. You may think briefly first, but you must "
    "end your reply with a line in exactly this format and nothing after it:\n"
    "ANSWER: X\nwhere X is one of A, B, C, or D."
)


def load_questions(directory):
    lock = startup.read_json(ROOT / "config/freedombench.lock.json")
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
    questions = []
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
        options = [correct, *distractors]
        seed = int.from_bytes(hashlib.sha256(qid.encode()).digest()[:8], "big")
        random.Random(seed).shuffle(options)
        answer = "ABCD"[options.index(correct)]
        prompt = "\n".join(
            [
                question,
                "",
                *[f"{letter}) {text}" for letter, text in zip("ABCD", options)],
                "",
                "Answer with a single letter (A, B, C, or D).",
            ]
        )
        questions.append(
            {
                "id": qid,
                "topic": topic,
                "prompt": prompt,
                "answer": answer,
                "source": source,
            }
        )
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    return lock, questions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "state/startup.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, help="Pilot only; never a full-suite score"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Upstream reasoning budget; must fit context",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Upstream initial plus four no-choice retries",
    )
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("Run evaluations on the Linux model host; scoring is CPU-portable")
    if (
        args.max_tokens < 1
        or args.max_attempts < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Budgets and limit must be positive")
    profile = startup_config.load(args.config)
    lock, questions = load_questions(args.benchmark_dir)
    selected = questions[: args.limit] if args.limit else questions
    current = startup.read_json(ROOT / "state/startup-rank0.json")
    info = startup.inspect_owned(current["name"], startup_config.fingerprint(profile))
    if not info["State"]["Running"]:
        parser.error("The configured local server is not running")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "benchmark": lock,
        "model_lock": load_lock(),
        "profile": profile,
        "image_id": info["Image"],
        "pilot": bool(args.limit),
        "questions": selected,
        "results": [],
        "max_attempts": args.max_attempts,
        "max_tokens": args.max_tokens,
        "language": "original-English",
        "system_prompt": SYSTEM_PROMPT,
        "human_refusal_audit": "not performed",
    }

    def save():
        report["summary"] = summarize(selected, report["results"])
        report["full_suite_complete"] = (
            not report["pilot"] and report["summary"]["valid_complete_run"]
        )
        write_json(args.output / "result.json", report)

    import fcntl

    with (ROOT / "state/startup-request.lock").open("a") as request_lock:
        fcntl.flock(request_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        for question in selected:
            row = {"id": question["id"], "attempts": []}
            report["results"].append(row)
            for _ in range(args.max_attempts):
                began = time.monotonic()
                try:
                    response = startup.ask(
                        profile,
                        {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": question["prompt"]},
                            ],
                            "max_tokens": args.max_tokens,
                        },
                    )
                except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                    response = {"error": type(error).__name__}
                    if getattr(error, "code", None) in (401, 403):
                        response.update(
                            error="authentication failed", http_status=error.code
                        )
                response["elapsed_seconds"] = time.monotonic() - began
                row["attempts"].append(response)
                save()
                if (
                    response.get("error")
                    or classify_attempt(response)["choice"] is not None
                ):
                    break
            save()
            print(question["id"], classify_attempt(row["attempts"][-1]), flush=True)
        report["status"] = "complete"
        save()
    if not report["summary"]["valid_complete_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
