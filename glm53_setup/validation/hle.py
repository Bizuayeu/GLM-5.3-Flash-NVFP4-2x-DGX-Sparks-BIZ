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
from .hle_scoring import SYSTEM_PROMPT, extract, final_content

MAX_TOKENS = (
    8192  # FreedomBench's upstream reasoning budget; fits the 600 s client timeout
)
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
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("Run on the Linux model host; grading is CPU-portable")
    if args.max_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Budgets and limit must be positive")
    profile = server_config.load(args.config)
    digest, rows = load_questions(args.questions)
    selected = rows[: args.limit] if args.limit else rows
    _, info = server.running_head(profile)
    if not info["State"]["Running"]:
        parser.error("The configured local server is not running")
    manifest = {
        "questions_sha256": digest,
        "questions": len(selected),
        "pilot": bool(args.limit),
        "label": args.label,
        "profile": profile,
        "fingerprint": server_config.fingerprint(profile),
        "image_id": info["Image"],
        "model_lock": load_lock(),
        "system_prompt": SYSTEM_PROMPT,
        "max_tokens": args.max_tokens,
        "teacher_excluded": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    stored = args.output / "manifest.json"
    if stored.exists():
        previous = read_json(stored)
        if {k: previous.get(k) for k in manifest} != manifest:
            parser.error("Output belongs to a different run; use a new directory")
    else:
        write_json(stored, manifest)
    stop = args.output / "STOP"
    status = {"status": "running"}
    with server.request_lock():
        try:
            for row in selected:
                target = answer_path(args.output, row["id"])
                if target.exists():
                    continue
                if stop.exists():
                    status = {"status": "stopped"}
                    break
                began = time.monotonic()
                try:
                    response = server.ask(
                        profile,
                        {"messages": messages(row), "max_tokens": args.max_tokens},
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
                content, finish = final_content(response)
                answer, confidence = extract(content)
                message = (response.get("choices") or [{}])[0].get("message") or {}
                write_json(
                    target,
                    {
                        "id": row["id"],
                        "category": row["category"],
                        "has_image": bool(row["image"]),
                        "label": args.label,
                        "elapsed_seconds": time.monotonic() - began,
                        "finish_reason": finish,
                        "content": content,
                        "reasoning": message.get("reasoning_content")
                        or message.get("reasoning"),
                        "answer": answer,
                        "confidence": confidence,
                        "usage": response.get("usage"),
                        "teacher_excluded": True,
                    },
                )
                print(row["id"], finish, answer is not None, flush=True)
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
