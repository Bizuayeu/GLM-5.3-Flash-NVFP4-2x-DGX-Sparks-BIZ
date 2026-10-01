"""Build the source-pinned reference image; this step does not validate TP=2 serving."""

import argparse
import json
import subprocess
import sys
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


# overlay2 stops loading an image past about 125 layers ("max depth exceeded" on
# `docker load`; MiaAI-Lab recipe #301-#304, moby/moby#46740). The warning mirrors
# that recipe's budget test (#304), which fails above 123.
LAYER_LIMIT = 125
LAYER_WARNING = 123


def layer_budget(inspect):
    """Layers of an image from `docker image inspect` JSON and its headroom to the limit.

    RootFS.Layers is the stack overlay2 mounts; `docker history` also lists steps
    that add no layer.
    """
    layers = len(inspect[0]["RootFS"]["Layers"])
    return {
        "layers": layers,
        "limit": LAYER_LIMIT,
        "headroom": LAYER_LIMIT - layers,
        "warning": layers > LAYER_WARNING,
    }


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
    budget = layer_budget(json.loads(result.stdout))
    write_json(record / "image-layers.json", budget)
    if budget["warning"]:
        print(
            f"Warning: the image has {budget['layers']} layers; overlay2 cannot "
            f"load more than about {LAYER_LIMIT} (docker load: max depth exceeded)",
            file=sys.stderr,
        )
    print("Build record:", record)
