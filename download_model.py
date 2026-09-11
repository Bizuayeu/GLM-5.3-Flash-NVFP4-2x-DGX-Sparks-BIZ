"""Download the pinned NVIDIA GLM checkpoint into the shared Hugging Face cache."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL = "nvidia/GLM-5.3-Flash-NVFP4"
REVISION = "423acf37583782c51c142d145aef733d72943d93"
STATE = Path(__file__).resolve().parent / "state"


def main():
    from huggingface_hub import HfApi, snapshot_download

    STATE.mkdir(parents=True, exist_ok=True)
    status = {
        "model": MODEL,
        "revision": REVISION,
        "status": "downloading",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def save():
        (STATE / "download-status.json").write_text(json.dumps(status, indent=2) + "\n")

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
        # Match the existing Gemma downloader's concurrency and timeout settings.
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


if __name__ == "__main__":
    if "--background" in sys.argv:
        STATE.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(HF_XET_NUM_CONCURRENT_RANGE_GETS="4", HF_HUB_DOWNLOAD_TIMEOUT="120")
        with (STATE / "download.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, __file__],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (STATE / "download.pid").write_text(str(process.pid) + "\n")
        print("Download PID:", process.pid)
    else:
        main()
