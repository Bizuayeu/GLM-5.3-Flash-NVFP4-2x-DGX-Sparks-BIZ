"""Answer a pinned HLE question file with the explicitly selected local GLM.

The question file is exported off-host and carries no reference answers;
grading happens elsewhere. Each answer is saved as its own file, so a run can
stop between questions (a STOP file or an error) and resume without resending
completed questions. Nothing here is teacher data.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.error
from pathlib import Path

from .. import server, server_config
from ..config import DEFAULT_PROFILE, load_lock
from ..io import read_json, write_json
from .freedombench import REASONING_BUDGET
from .hle_scoring import SYSTEM_PROMPT, extract, final_content

# FreedomBench's upstream reasoning budget; fits the 600 s client timeout.
MAX_TOKENS = REASONING_BUDGET
REQUIRED = {"id", "question", "image", "category"}


def load_questions(path):
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    for row in rows:
        if set(row) != REQUIRED:
            raise ValueError(f"Question rows must have exactly {sorted(REQUIRED)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Question IDs must be unique")
    return hashlib.sha256(data).hexdigest(), rows


def messages(row):
    if row["image"]:
        if not row["image"].startswith("data:image/"):
            raise ValueError(f"{row['id']}: image must be a data URI")
        user = [
            {"type": "image_url", "image_url": {"url": row["image"]}},
            {"type": "text", "text": row["question"]},
        ]
    else:
        user = row["question"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer_path(output, qid):
    return output / "answers" / f"{qid}.json"


def summarize(rows, output):
    done = [
        read_json(answer_path(output, r["id"]))
        for r in rows
        if answer_path(output, r["id"]).exists()
    ]
    parsed = [d for d in done if d.get("answer") is not None]
    return {
        "planned": len(rows),
        "answered": len(done),
        "parsed": len(parsed),
        "truncated": sum(d.get("finish_reason") == "length" for d in done),
        "complete": len(done) == len(rows),
    }


def client_profile(profile, timeout):
    """The profile the client asks with: only its wait changes, never the served one."""
    if timeout is None:
        return profile
    return dict(
        profile,
        generation=dict(profile.get("generation", {}), timeout_seconds=timeout),
    )


def run_manifest(
    profile,
    *,
    digest,
    questions,
    pilot,
    label,
    image_id,
    lock,
    max_tokens,
    sampling,
    timeout,
    fingerprint,
):
    """What identifies a run; a resumed output must carry the same fields."""
    return {
        "questions_sha256": digest,
        "questions": questions,
        "pilot": pilot,
        "label": label,
        "profile": profile,
        "fingerprint": fingerprint,
        "image_id": image_id,
        "model_lock": lock,
        "system_prompt": SYSTEM_PROMPT,
        "max_tokens": max_tokens,
        "sampling": sampling,
        "timeout_seconds": timeout,
        "teacher_excluded": True,
    }


def same_run(previous, manifest):
    return {k: previous.get(k) for k in manifest} == manifest


def next_step(*, done, stop, answered_now, max_new):
    """skip an answered row, then honour STOP, then the max-new pause, else ask."""
    if done:
        return "skip"
    if stop:
        return "stopped"
    if max_new is not None and answered_now == max_new:
        return "paused"
    return "ask"


def answer_record(row, response, label, elapsed):
    content, finish = final_content(response)
    answer, confidence = extract(content)
    message = (response.get("choices") or [{}])[0].get("message") or {}
    return {
        "id": row["id"],
        "category": row["category"],
        "has_image": bool(row["image"]),
        "label": label,
        "elapsed_seconds": elapsed,
        "finish_reason": finish,
        "content": content,
        "reasoning": message.get("reasoning_content") or message.get("reasoning"),
        "answer": answer,
        "confidence": confidence,
        "usage": response.get("usage"),
        "teacher_excluded": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label",
        required=True,
        help="Profile label for the record, e.g. default or axl",
    )
    parser.add_argument("--limit", type=int, help="Pilot only; never a full-set score")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, help="Default: the profile's")
    parser.add_argument("--top-p", type=float, help="Default: the server's")
    parser.add_argument(
        "--timeout", type=int, help="Client seconds per question; default the profile's"
    )
    parser.add_argument(
        "--max-new",
        type=int,
        help="Answer at most this many new questions, then pause (status paused), "
        "so a driver can wait for the hosts to cool between questions",
    )
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("Run on the Linux model host; grading is CPU-portable")
    if any(
        value is not None and value < 1
        for value in (args.max_tokens, args.limit, args.timeout, args.max_new)
    ):
        parser.error("Budgets, limit, timeout and max-new must be positive")
    profile = server_config.load(args.config)
    sampling = {
        key: value
        for key, value in (("temperature", args.temperature), ("top_p", args.top_p))
        if value is not None
    }
    asking = client_profile(profile, args.timeout)
    digest, rows = load_questions(args.questions)
    selected = rows[: args.limit] if args.limit else rows
    _, info = server.running_head(profile)
    if not info["State"]["Running"]:
        parser.error("The configured local server is not running")
    manifest = run_manifest(
        profile,
        digest=digest,
        questions=len(selected),
        pilot=bool(args.limit),
        label=args.label,
        image_id=info["Image"],
        lock=load_lock(),
        max_tokens=args.max_tokens,
        sampling=sampling,
        timeout=args.timeout,
        fingerprint=server_config.fingerprint(profile),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stored = args.output / "manifest.json"
    if stored.exists():
        if not same_run(read_json(stored), manifest):
            parser.error("Output belongs to a different run; use a new directory")
    else:
        write_json(stored, manifest)
    stop = args.output / "STOP"
    status = {"status": "running"}
    answered_now = 0
    with server.request_lock():
        try:
            for row in selected:
                target = answer_path(args.output, row["id"])
                step = next_step(
                    done=target.exists(),
                    stop=stop.exists(),
                    answered_now=answered_now,
                    max_new=args.max_new,
                )
                if step == "skip":
                    continue
                if step != "ask":
                    status = {"status": step}
                    break
                began = time.monotonic()
                try:
                    response = server.ask(
                        asking,
                        {
                            "messages": messages(row),
                            "max_tokens": args.max_tokens,
                            **sampling,
                        },
                    )
                except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                    write_json(
                        args.output / "errors" / f"{row['id']}-{int(time.time())}.json",
                        {
                            "id": row["id"],
                            "error": repr(error),
                            "elapsed_seconds": time.monotonic() - began,
                        },
                    )
                    raise
                record = answer_record(
                    row, response, args.label, time.monotonic() - began
                )
                write_json(target, record)
                answered_now += 1
                print(
                    row["id"],
                    record["finish_reason"],
                    record["answer"] is not None,
                    flush=True,
                )
            else:
                status = {"status": "complete"}
        except BaseException as error:
            status = {"status": "failed", "error": repr(error)}
            raise
        finally:
            write_json(
                args.output / "status.json",
                dict(status, summary=summarize(selected, args.output)),
            )


if __name__ == "__main__":
    main()
