"""Read the CUDA caching allocator from a serving worker (dev `/collective_rpc`).

On GB10 the GPU shares the host's memory, so a growing allocator shows up as
falling MemAvailable while the worker's RSS and its container's cgroup stay
flat. This extension answers whether such growth is inside torch's allocator
(reserved bytes, retries, segments) or somewhere else.

``host_stats`` is the other side: a worker whose own anonymous memory grows.
glibc's ``mallinfo2`` splits the heap into bytes in use, free bytes it keeps
and mapped blocks, torch reports its pinned host cache, and ``trim=True`` calls
``malloc_trim(0)`` between two readings, so retained free memory, live objects
and memory outside malloc can be told apart.
"""

import ctypes

GIB = 2**30


def summarize(stats, mem_get_info):
    """Reduce ``torch.cuda.memory_stats()`` and ``mem_get_info()`` to one row."""

    def gib(key):
        return round(stats.get(key, 0) / GIB, 3)

    free, total = mem_get_info
    return {
        "reserved_gib": gib("reserved_bytes.all.current"),
        "reserved_peak_gib": gib("reserved_bytes.all.peak"),
        "allocated_gib": gib("allocated_bytes.all.current"),
        "active_gib": gib("active_bytes.all.current"),
        "inactive_split_gib": gib("inactive_split_bytes.all.current"),
        "segments": stats.get("segment.all.current", 0),
        "alloc_retries": stats.get("num_alloc_retries", 0),
        "ooms": stats.get("num_ooms", 0),
        "device_allocs": stats.get("num_device_alloc", 0),
        "device_frees": stats.get("num_device_free", 0),
        "device_free_gib": round(free / GIB, 3),
        "device_total_gib": round(total / GIB, 3),
    }


MIB = 2**20
MALLINFO2_FIELDS = (
    "arena",
    "ordblks",
    "smblks",
    "hblks",
    "hblkhd",
    "usmblks",
    "fsmblks",
    "uordblks",
    "fordblks",
    "keepcost",
)


class Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in MALLINFO2_FIELDS]


def summarize_host(heap, status, pinned):
    """One row from mallinfo2 fields, /proc/self/status text and torch's pinned cache."""

    def mib(value):
        return round(value / MIB, 1)

    resident = {
        key: int(value.split()[0]) * 1024
        for key, _, value in (line.partition(":") for line in status.splitlines())
        if key in ("RssAnon", "RssFile", "RssShmem")
    }
    row = {
        "rss_anon_mib": mib(resident.get("RssAnon", 0)),
        "rss_file_mib": mib(resident.get("RssFile", 0)),
        "rss_shmem_mib": mib(resident.get("RssShmem", 0)),
        "heap_in_use_mib": mib(heap["uordblks"]),
        "heap_free_retained_mib": mib(heap["fordblks"]),
        "heap_arena_mib": mib(heap["arena"]),
        "heap_mmapped_mib": mib(heap["hblkhd"]),
    }
    if pinned is not None:
        row["pinned_reserved_mib"] = mib(pinned.get("reserved_bytes.current", 0))
        row["pinned_allocated_mib"] = mib(pinned.get("allocated_bytes.current", 0))
    return row


def read_host():
    import torch

    libc = ctypes.CDLL("libc.so.6")
    libc.mallinfo2.restype = Mallinfo2
    info = libc.mallinfo2()
    heap = {name: getattr(info, name) for name in MALLINFO2_FIELDS}
    with open("/proc/self/status") as stream:
        status = stream.read()
    reader = getattr(torch.cuda, "host_memory_stats", None)
    return summarize_host(heap, status, reader() if reader else None)


def trim_heap():
    return ctypes.CDLL("libc.so.6").malloc_trim(0)


class MemoryProbeWorker:
    def host_stats(self, trim=False):
        row = {"rank": self.rank}
        if trim:
            row["before_trim"] = read_host()
            row["trim_released"] = trim_heap()
        row.update(read_host())
        return row

    def allocator_stats(self):
        import torch

        row = summarize(torch.cuda.memory_stats(), torch.cuda.mem_get_info())
        row["rank"] = self.rank
        return row
