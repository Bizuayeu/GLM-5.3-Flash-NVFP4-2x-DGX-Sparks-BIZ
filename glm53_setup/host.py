"""Host-side helpers shared by the server launcher: site validation, serve arguments, fabric checks, snapshot resolution and subprocess execution."""

import ipaddress
import json
import mmap
import re
import subprocess
from pathlib import Path

from . import fabric


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
    hcas = ",".join(f"{r['hca']}:{r['port']}" for r in rails)
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


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def meminfo_gib(text, field):
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name == field:
            return int(value.split()[0]) / 1024**2
    raise ValueError(f"/proc/meminfo has no {field}")


def available_gib():
    return meminfo_gib(Path("/proc/meminfo").read_text(), "MemAvailable")


def free_blocks_gib(buddyinfo, page_size, min_bytes):
    """Free memory held in buddy blocks of at least min_bytes, summed over zones.

    NVRM allocates without reclaiming the page cache and needs contiguous 2 MiB
    blocks, so MemAvailable can stay high while such an allocation fails.
    """
    order = 0
    while page_size << order < min_bytes:
        order += 1
    total, zones = 0, 0
    for line in buddyinfo.splitlines():
        if not line.strip():
            continue
        _, found, counts = line.partition(" zone ")
        fields = counts.split()[1:]
        if not found or not fields or not all(f.isdigit() for f in fields):
            raise ValueError("Unrecognized /proc/buddyinfo line")
        zones += 1
        total += sum(
            int(count) * (page_size << index)
            for index, count in enumerate(fields)
            if index >= order
        )
    if not zones:
        raise ValueError("/proc/buddyinfo lists no zones")
    return total / 1024**3


def memory_sample():
    """Observation only: an unreadable sample is recorded, never a stop condition."""
    try:
        return {
            "mem_free_gib": meminfo_gib(Path("/proc/meminfo").read_text(), "MemFree"),
            "free_2mib_gib": free_blocks_gib(
                Path("/proc/buddyinfo").read_text(), mmap.PAGESIZE, 2 * 1024**2
            ),
        }
    except Exception as error:  # noqa: BLE001 - any unreadable sample is recorded, never raised
        return {"memory_sample_error": type(error).__name__}


def running_containers():
    """Inspections of running containers; one that exits or is removed meanwhile is dropped.

    A container still listed after its inspection failed raises, so the guard fails closed.
    """
    inspections = []
    for container in run("docker", "ps", "-q").split():
        try:
            info = json.loads(run("docker", "inspect", container))[0]
        except subprocess.CalledProcessError:
            if container in run("docker", "ps", "-q").split():
                raise
            continue
        if info["State"]["Running"]:
            inspections.append(info)
    return inspections


def foreign_gpu_containers(inspections, label):
    """Running containers that request a GPU and lack this launcher's label.

    Both `--gpus` and CDI (`--device nvidia.com/gpu=...`) requests appear in
    HostConfig.DeviceRequests; the GPU device nodes never appear in Devices.
    """
    return sorted(
        info["Name"].lstrip("/")
        for info in inspections
        if info["HostConfig"].get("DeviceRequests")
        and not (info["Config"].get("Labels") or {}).get(label)
    )


def fabric_checks(site):
    validate_site(site)
    return fabric.checks(site, run)
