"""Download the pinned NVIDIA GLM checkpoint into the shared Hugging Face cache."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import MODEL, REVISION, ROOT, STATE
from .io import write_json

# Under STATE; the preflight and the checksum run read what this writes there.
STATUS_FILE = "download-status.json"


def download():
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import HfApi, snapshot_download

    STATE.mkdir(parents=True, exist_ok=True)
    status = {
        "model": MODEL,
        "revision": REVISION,
        "status": "downloading",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def save():
        write_json(STATE / STATUS_FILE, status)

    save()
    try:
        info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise RuntimeError("Unexpected model revision")
        files = [{"path": item.rfilename, "bytes": item.size} for item in info.siblings]
        (STATE / "model-manifest.json").write_text(
            json.dumps({"model": MODEL, "revision": REVISION, "files": files}, indent=2)
            + "\n"
        )
        status["total_bytes"] = sum(item["bytes"] or 0 for item in files)
        save()
        print("Downloading:", MODEL, REVISION, status["total_bytes"], flush=True)
        # Bound simultaneous shard downloads on unified-memory hosts.
        snapshot = Path(snapshot_download(MODEL, revision=REVISION, max_workers=2))
        for item in files:
            file = snapshot / item["path"]
            if (
                not file.is_file()
                or item["bytes"] is not None
                and file.stat().st_size != item["bytes"]
            ):
                raise RuntimeError("Incomplete download: " + item["path"])
        weights = json.loads((snapshot / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        if not weights or not all(
            (snapshot / name).is_file() for name in set(weights.values())
        ):
            raise RuntimeError("Indexed model weights are missing")
        status.update(
            status="complete",
            snapshot=str(snapshot),
            file_count=len(files),
            weight_shards=len(set(weights.values())),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        save()
        print("COMPLETE:", snapshot, flush=True)
    except Exception as error:  # noqa: BLE001 -- persist unexpected failures at the job boundary
        status.update(status="failed", error_type=type(error).__name__)
        save()
        print("DOWNLOAD FAILED:", type(error).__name__, flush=True)
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(argv)
    if args.background:
        if os.name != "posix":
            parser.error("Background jobs require Linux; use foreground mode here")
        STATE.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        with (STATE / "download.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "glm53_setup", "download"],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print("Download PID:", process.pid)
    else:
        from filelock import FileLock, Timeout

        STATE.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(STATE / "download.lock", timeout=0):
                (STATE / "download.pid").write_text(str(os.getpid()) + "\n")
                download()
        except Timeout:
            raise SystemExit(
                "A download already owns this workspace; inspect its state"
            ) from None


if __name__ == "__main__":
    main()
