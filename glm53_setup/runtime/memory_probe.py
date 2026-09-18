"""Read the CUDA caching allocator from a serving worker (dev `/collective_rpc`).

On GB10 the GPU shares the host's memory, so a growing allocator shows up as
falling MemAvailable while the worker's RSS and its container's cgroup stay
flat. This extension answers whether such growth is inside torch's allocator
(reserved bytes, retries, segments) or somewhere else.
"""

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


class MemoryProbeWorker:
    def allocator_stats(self):
        import torch

        row = summarize(torch.cuda.memory_stats(), torch.cuda.mem_get_info())
        row["rank"] = self.rank
        return row
