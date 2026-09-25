"""Build the source-pinned reference image; this step does not validate TP=2 serving."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import RECORDS, ROOT, load_lock
from .io import write_json


def build_command(lock):
    return [
        "docker",
        "build",
        "--platform",
        lock["platform"],
        "--build-arg",
        "BASE_IMAGE=" + lock["image"],
        "-f",
        str(ROOT / "docker/Dockerfile.reference"),
        "-t",
        lock["reference_candidate"]["tag"],
        str(ROOT),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", action="store_true", help="Print the command without building"
    )
    parser.add_argument("--record-dir", type=Path)
    args = parser.parse_args(argv)
    lock = load_lock()
    command = build_command(lock)
    if args.plan:
        print(json.dumps(command, indent=2))
        return
    record = args.record_dir or RECORDS / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ-build"
    )
    record.mkdir(parents=True, exist_ok=True)
    write_json(record / "command.json", command)
    with (record / "build.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    # A build runs no model: tp2_validated is always False (the record's shape).
    write_json(
        record / "result.json", {"exit_code": result.returncode, "tp2_validated": False}
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    result = subprocess.run(
        ["docker", "image", "inspect", lock["reference_candidate"]["tag"]],
        capture_output=True,
        text=True,
        check=True,
    )
    (record / "image-inspect.json").write_text(result.stdout, encoding="utf-8")
    print("Build record:", record)
