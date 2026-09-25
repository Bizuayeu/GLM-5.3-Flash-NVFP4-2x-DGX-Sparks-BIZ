"""Host-side helpers shared by the server launcher: fabric checks, snapshot resolution, memory samples, container inspection and subprocess execution."""

import json
import mmap
import subprocess
from pathlib import Path

from . import fabric


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


def container_memory_sample(
    container_id, cgroup_root=Path("/sys/fs/cgroup"), proc_root=Path("/proc")
):
    """Observation only: what the container's own processes hold.

    On GB10 the GPU shares host memory and device allocations are charged to
    neither the cgroup nor a process, so a falling MemAvailable beside a flat
    cgroup and flat RSS points at the device side, and rising RSS at the host side.
    """
    try:
        scope = cgroup_root / "system.slice" / f"docker-{container_id}.scope"
        sample = {
            "container_cgroup_gib": int((scope / "memory.current").read_text())
            / 1024**3
        }
        rss = anon = 0
        for pid in (scope / "cgroup.procs").read_text().split():
            try:
                status = (proc_root / pid / "status").read_text()
            except OSError:
                continue  # exited between the two reads
            for line in status.splitlines():
                name, _, value = line.partition(":")
                if name == "VmRSS":
                    rss += int(value.split()[0])
                elif name == "RssAnon":
                    anon += int(value.split()[0])
        sample["container_rss_gib"] = rss / 1024**2
        sample["container_anon_gib"] = anon / 1024**2
        return sample
    except Exception as error:  # noqa: BLE001 - any unreadable sample is recorded, never raised
        return {"container_memory_error": type(error).__name__}


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
    fabric.validate_site(site)
    return fabric.checks(site, run)
