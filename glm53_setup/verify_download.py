"""Wait for the existing downloader, then verify the pinned HF cache checksums."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import MODEL, REVISION, STATE
from .io import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    status_file = args.output / "checksum-status.json"

    def save(status, **extra):
        write_json(
            status_file,
            {
                "status": status,
                "model": MODEL,
                "revision": REVISION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **extra,
            },
        )

    save("waiting_for_download")
    while True:
        status = json.loads((STATE / "download-status.json").read_text())
        if status["model"] != MODEL or status["revision"] != REVISION:
            save("failed", reason="revision mismatch")
            raise SystemExit(1)
        if status["status"] == "complete":
            break
        if status["status"] in ("failed", "paused") or not args.wait:
            paused = status["status"] == "paused"
            save("paused" if paused else "failed", reason="download not complete")
            raise SystemExit(2 if paused else 1)
        # Polling cadence only; does not bound or retry downloads.
        time.sleep(30)
    save("verifying")
    command = [
        str(args.hf),
        "cache",
        "verify",
        MODEL,
        "--revision",
        REVISION,
        "--fail-on-missing-files",
        "--fail-on-extra-files",
        "--json",
    ]
    with (
        (args.output / "checksum.json").open("w", encoding="utf-8") as out,
        (args.output / "checksum.log").open("w", encoding="utf-8") as err,
    ):
        result = subprocess.run(command, stdout=out, stderr=err, check=False)
    save(
        "complete" if result.returncode == 0 else "failed", exit_code=result.returncode
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
