"""Operate one TP=2 rank with explicit fabric settings and saved validation."""

import argparse
import ipaddress
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import fabric
from .config import ROOT, load_lock


def validate_site(site):
    if type(site.get("rank")) is not int or site["rank"] not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    for key in ("head_ip", "local_ip"):
        address = ipaddress.IPv4Address(site[key])
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError(f"{key} must be a fabric IPv4 address")
    if (site["rank"] == 0) != (site["head_ip"] == site["local_ip"]):
        raise ValueError("Head must own head_ip; worker must have a different address")
    for key in ("interface", "hca"):
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", site.get(key, "")):
            raise ValueError(f"Set a concrete {key} from the local device inventory")
    if site["interface"].startswith("wl"):
        raise ValueError("The real-model profile requires RoCE, not Wi-Fi")
    if type(site.get("gid_index")) is not int or site["gid_index"] < 0:
        raise ValueError("gid_index must be nonnegative")
    for key in ("api_port", "master_port"):
        if type(site.get(key)) is not int or not 1024 <= site[key] <= 65535:
            raise ValueError(f"Invalid {key}")
    if site["api_port"] == site["master_port"]:
        raise ValueError("API and rendezvous ports must differ")
    fabric.rails(site)


def serve_args(site, model_path):
    validate_site(site)
    args = [
        "serve",
        str(model_path),
        "--served-model-name",
        "glm-5.3-flash-nvidia",
        "--distributed-executor-backend",
        "mp",
        "--nnodes",
        "2",
        "--tensor-parallel-size",
        "2",
        "--node-rank",
        str(site["rank"]),
        "--master-addr",
        site["head_ip"],
        "--master-port",
        str(site["master_port"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(site["api_port"]),
        "--language-model-only",
        "--enforce-eager",
        "--kv-cache-dtype",
        "fp8",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "512",
        "--gpu-memory-utilization",
        "0.80",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--reasoning-parser",
        "glm45",
        "--tool-call-parser",
        "glm47",
        "--enable-auto-tool-choice",
    ]
    if site["rank"] == 1:
        args.append("--headless")
    return args


def fabric_env(site):
    validate_site(site)
    rails = fabric.rails(site)
    hcas = (
        site["hca"]
        if len(rails) == 1
        else ",".join(f"{r['hca']}:{r['port']}" for r in rails)
    )
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_HOST_IP": site["local_ip"],
        "NCCL_NET": "IB",
        "NCCL_IB_DISABLE": "0",
        "NCCL_SOCKET_IFNAME": "=" + site["interface"],
        "GLOO_SOCKET_IFNAME": site["interface"],
        "NCCL_IB_HCA": "=" + hcas,
        "NCCL_IB_GID_INDEX": str(site["gid_index"]),
        "NCCL_IB_ROCE_VERSION_NUM": "2",
        "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_DEBUG": "INFO",
    }


def snapshot_from_state(state, lock):
    if state.get("status") != "complete" or any(
        state.get(key) != lock[key] for key in ("model", "revision")
    ):
        raise ValueError("Matching fixed-revision download must be complete")
    return Path(state["snapshot"])


def valid_kernel_receipt(receipt, lock):
    return (
        receipt.get("passed") is True
        and receipt.get("kind") == "tp2-kernel-validation"
        and receipt.get("image") == lock["image"]
        and receipt.get("model_revision") == lock["revision"]
    )


def docker_args(site, lock, snapshot, cache, root):
    relative = snapshot.resolve().relative_to(cache.resolve())
    name = f"glm53-nvidia-rank{site['rank']}"
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--init",
        "--restart",
        "no",
        "--label",
        "glm53.setup.model=glm53-nvidia",
        "--label",
        f"glm53.setup.rank={site['rank']}",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--cap-add",
        "IPC_LOCK",
        "--ulimit",
        "memlock=-1:-1",
        "--device",
        "/dev/infiniband:/dev/infiniband",
        "-v",
        f"{cache.resolve()}:/hf:ro",
        "-v",
        f"{(root / 'state' / 'runtime-cache').resolve()}:/root/.cache",
    ]
    for key, value in fabric_env(site).items():
        args += ["-e", f"{key}={value}"]
    return args + [
        "--entrypoint",
        "vllm",
        lock["image"],
        *serve_args(site, "/hf/" + relative.as_posix()),
    ]


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def fabric_checks(site):
    validate_site(site)
    return fabric.checks(site, run)


def preflight(site, lock, snapshot):
    checks = {
        "snapshot_config": (snapshot / "config.json").is_file(),
        **fabric_checks(site),
    }
    receipt_path = ROOT / "state" / "kernel-validation.json"
    checks["kernel_validation"] = receipt_path.exists() and valid_kernel_receipt(
        json.loads(receipt_path.read_text()), lock
    )
    meminfo = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    available_gib = int(meminfo["MemAvailable"].split()[0]) / 1024**2
    # 87.743 GiB payload/rank + 8 GiB KV candidate + 8 GiB OS reserve + ~4 GiB
    # provisional runtime allowance; measured peak remains a separate gate.
    checks["startup_memory"] = available_gib >= 108
    existing = run("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()
    checks["rank_container_absent"] = f"glm53-nvidia-rank{site['rank']}" not in existing
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "available_gib": available_gib,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["plan", "preflight", "start", "stop", "status"]
    )
    parser.add_argument("--site", type=Path, default=ROOT / "state/site.json")
    args = parser.parse_args(argv)
    site = json.loads(args.site.read_text())
    validate_site(site)
    name = f"glm53-nvidia-rank{site['rank']}"
    if args.action in ("stop", "status"):
        info = json.loads(run("docker", "inspect", name))[0]
        if (info["Config"].get("Labels") or {}).get(
            "glm53.setup.model"
        ) != "glm53-nvidia":
            raise SystemExit("Container is not owned by this deployment")
        if args.action == "stop":
            print(run("docker", "stop", name))
        else:
            print(
                json.dumps(
                    {"image_id": info["Image"], "state": info["State"]}, indent=2
                )
            )
        return
    lock = load_lock()
    status = json.loads((ROOT / "state/download-status.json").read_text())
    snapshot = snapshot_from_state(status, lock)
    command = docker_args(
        site, lock, snapshot, Path.home() / ".cache/huggingface", ROOT
    )
    if args.action == "plan":
        print(json.dumps(command, indent=2))
        return
    result = preflight(site, lock, snapshot)
    record = (
        ROOT
        / "records"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-preflight")
    )
    record.mkdir(parents=True)
    (record / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)
    if args.action == "start":
        (ROOT / "state/runtime-cache").mkdir(parents=True, exist_ok=True)
        (record / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(run(*command))


if __name__ == "__main__":
    main()
