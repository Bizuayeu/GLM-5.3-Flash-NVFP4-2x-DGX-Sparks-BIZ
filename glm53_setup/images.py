"""Fetch the pinned candidate and record its actual GPU/runtime capabilities."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, load_lock

PROBE = """
import json, importlib.metadata as m, platform
import torch
result = {'architecture': platform.machine(), 'packages': {}}
for name in ['vllm', 'torch', 'transformers', 'flashinfer-python', 'nvidia-nccl-cu13', 'nvidia-nccl-cu12']:
    try: result['packages'][name] = m.version(name)
    except m.PackageNotFoundError: pass
result['cuda'] = torch.version.cuda
result['gpu'] = torch.cuda.get_device_name(0)
result['capability'] = list(torch.cuda.get_device_capability(0))
x = torch.arange(65539, device='cuda', dtype=torch.int64)
result['gpu_mismatches'] = int(((x + x) != x * 2).sum().item())
from vllm.model_executor.models import ModelRegistry
result['glm_architectures'] = sorted(x for x in ModelRegistry.get_supported_archs() if 'glm5' in x.lower())
print(json.dumps(result))
"""


def run(record: Path):
    lock = load_lock()
    image = lock["image"]
    record.mkdir(parents=True, exist_ok=True)
    status = {"started_at": datetime.now(timezone.utc).isoformat(), "image": image}

    def save():
        (record / "prepare-status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )

    try:
        status["status"] = "pulling"
        save()
        with (record / "pull-image.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                ["docker", "pull", "--platform", lock["platform"], image],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        status["status"] = "probing"
        save()
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=True,
        )
        (record / "image-inspect.json").write_text(inspect.stdout, encoding="utf-8")
        # Keep the diagnostic container as evidence; no automatic deletion.
        name = "glm53-probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        status["probe_container"] = name
        save()
        result = subprocess.run(
            [
                "docker",
                "run",
                "--name",
                name,
                "--gpus",
                "all",
                "--network",
                "none",
                "--entrypoint",
                "python3",
                image,
                "-c",
                PROBE,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (record / "runtime-probe.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        result.check_returncode()
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        (record / "runtime-probe.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        if (
            payload["capability"] != [12, 1]
            or payload["gpu_mismatches"]
            or not payload["glm_architectures"]
        ):
            raise RuntimeError("GPU or model registration validation failed")
        status["status"] = "prepared_not_inference_validated"
    except Exception as error:
        status.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--record-dir", type=Path)
    args = parser.parse_args(argv)
    if args.background and os.name != "posix":
        parser.error("Background preparation requires Linux; use foreground mode here")
    record = (
        args.record_dir
        or ROOT
        / "records"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-prepare")
    ).resolve()
    if not args.background:
        run(record)
        return
    record.mkdir(parents=True, exist_ok=True)
    pid_path = record / "prepare.pid"
    if pid_path.exists():
        try:
            os.kill(int(pid_path.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit(
                "Preparation process already exists; inspect it before restarting"
            )
    with (record / "prepare.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "glm53_setup",
                "prepare-image",
                "--record-dir",
                str(record),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(process.pid), encoding="ascii")
    print(json.dumps({"pid": process.pid, "record": str(record)}))


if __name__ == "__main__":
    main()
