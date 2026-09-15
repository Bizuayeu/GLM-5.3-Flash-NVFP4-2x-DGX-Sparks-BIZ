"""Host-side helpers shared by the server launcher: site validation, serve arguments, fabric checks, snapshot resolution and subprocess execution."""

import ipaddress
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


def available_gib():
    rows = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return int(rows["MemAvailable"].split()[0]) / 1024**2


def fabric_checks(site):
    validate_site(site)
    return fabric.checks(site, run)
