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


def census(limit=12):
    """Live Python objects by type and the CPU tensors among them.

    A heap that grows with objects Python can see shows up here; one that grows
    without them is held by native code.
    """
    import gc
    from collections import Counter

    import torch

    counts = Counter()
    tensors = tensor_bytes = 0
    storages = set()
    for item in gc.get_objects():
        counts[type(item).__name__] += 1
        # type(), not isinstance(): a dead weakref proxy raises on __class__.
        if issubclass(type(item), torch.Tensor) and item.device.type == "cpu":
            tensors += 1
            storage = item.untyped_storage()
            if storage.data_ptr() not in storages:
                storages.add(storage.data_ptr())
                tensor_bytes += storage.nbytes()
    return {
        "objects": sum(counts.values()),
        "cpu_tensors": tensors,
        "cpu_tensor_mib": round(tensor_bytes / MIB, 1),
        "top_types": dict(counts.most_common(limit)),
    }


FA2_STAGES = (
    "off",
    "unique",
    "mask",
    "lengths",
    "unpack",
    "bitmap",
    "compact",
    "plan",
    "full",
)
_fa2_original = {}


def fa2_stage(stage):
    """Run only part of the FA2 path in this process, the rest on the reference path.

    ``off`` is the reference computation. ``unique``, ``mask``, ``lengths`` and
    ``unpack`` each add one operation of the compaction alone; ``bitmap`` is a
    compaction without ``torch.unique``; ``compact`` is the whole compaction,
    ``plan`` adds FlashInfer's ``plan()``, ``full`` restores the real function.
    The partial stages throw their work away, so the served numbers are the
    reference path's.
    """
    import torch

    from glm53_setup.runtime import fa2_attention as fa2
    from glm53_setup.runtime.reference_attention import unpack_latent

    if stage not in FA2_STAGES:
        raise ValueError(f"stage must be one of {FA2_STAGES}")
    if not _fa2_original:
        _fa2_original.update(run=fa2.sparse_nope_fa2, use=fa2.use_fa2)
    fa2.sparse_nope_fa2, fa2.use_fa2 = _fa2_original["run"], _fa2_original["use"]
    if stage == "full":
        return stage
    inner = []

    def unpack(query, flat_cache, rows):
        ckv = torch.empty(
            (rows.numel(), 1, 512), dtype=torch.bfloat16, device=query.device
        )
        for start in range(0, rows.numel(), fa2.UNPACK_ROWS):
            part = rows[start : start + fa2.UNPACK_ROWS].long()
            ckv[start : start + part.numel(), 0] = unpack_latent(flat_cache[part])

    def partial(query, packed_cache, physical_indices, scale):
        import glm53_reference

        if stage != "off":
            flat_cache = packed_cache.reshape(-1, 656)
            valid = physical_indices >= 0
        if stage == "unique":
            torch.unique(physical_indices.clamp_min(0), return_inverse=True)
        elif stage == "mask":
            physical_indices[valid].to(torch.int32)
        elif stage == "lengths":
            valid.sum(dim=1).to(torch.int32).to("cpu")
        elif stage == "unpack":
            unpack(query, flat_cache, physical_indices[:, -1].clamp_min(0))
        elif stage == "bitmap":
            used = torch.zeros(
                flat_cache.shape[0], dtype=torch.bool, device=query.device
            )
            used[physical_indices.clamp_min(0).long().reshape(-1)] = True
            rows = used.nonzero().squeeze(1)
            position = torch.cumsum(used, dim=0, dtype=torch.int32) - 1
            position[physical_indices[valid].long()]
            valid.sum(dim=1).to(torch.int32).to("cpu")
            unpack(query, flat_cache, rows)
        elif stage in ("compact", "plan"):
            rows, kv_indices, lengths = fa2.compact_candidates(physical_indices)
            host_lengths = lengths.to("cpu")
            unpack(query, flat_cache, rows)
        if stage == "plan":
            fa2._wrapper(query.device).plan(
                torch.arange(query.shape[0] + 1, dtype=torch.int32),
                fa2.indptr(host_lengths),
                kv_indices,
                host_lengths,
                query.shape[1],
                512,
                0,
                1,
                False,
                scale,
                q_data_type=query.dtype,
                kv_data_type=torch.bfloat16,
            )
        inner.append(True)
        try:
            return glm53_reference.sparse_nope_reference(
                query, packed_cache, physical_indices, scale
            )
        finally:
            inner.pop()

    fa2.sparse_nope_fa2 = partial
    fa2.use_fa2 = lambda query_rows: not inner and _fa2_original["use"](query_rows)
    return stage


TRACED = r"(^|\.)(embed_tokens|norm|lm_head|layers\.\d+(\.[A-Za-z_]+){0,2})$"


def trace_differences(reference, rows, limit=24):
    """Compare two fingerprint sequences of one request, in execution order.

    Each entry is ``[module, rows, fingerprint]``. Identical requests make the
    same calls until something differs, so the first differing entry names the
    module that made the difference: everything executed before it matched.
    """
    differing = []
    modules = {}
    for order, (old, new) in enumerate(zip(reference, rows)):
        if old != new:
            if len(differing) < limit:
                differing.append(
                    {"order": order, "module": new[0], "rows": new[1], "was": old[:2]}
                )
            modules[new[0]] = modules.get(new[0], 0) + 1
    return {
        "calls": [len(reference), len(rows)],
        "first": differing[0] if differing else None,
        "differing": differing,
        "modules": dict(sorted(modules.items(), key=lambda item: -item[1])[:limit]),
    }


class MemoryProbeWorker:
    def trace_begin(self):
        """Fingerprint every traced module output (and the token ids) of what runs next.

        Fingerprints are two byte sums kept on the GPU, so a whole request fits
        where clones of the outputs would not.
        """
        import re

        import torch

        pattern = re.compile(TRACED)
        state = {"handles": [], "names": [], "rows": [], "prints": []}

        def fingerprint(tensor):
            data = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
            return data.sum(dtype=torch.int64) * 1000003 + data[1::3].sum(
                dtype=torch.int64
            )

        def note(name, tensor):
            state["names"].append(name)
            state["rows"].append(int(tensor.shape[0]) if tensor.ndim else 1)
            state["prints"].append(fingerprint(tensor))

        def hook(name):
            def after(module, args, output):
                if name.endswith("embed_tokens") and args:
                    note(name + ":input_ids", args[0])
                items = output if isinstance(output, (tuple, list)) else [output]
                for item in items:
                    if isinstance(item, torch.Tensor) and item.is_floating_point():
                        note(name, item)
                        break

            return after

        # The draft model hangs off the runner under a version-dependent name:
        # take every other torch module reachable one attribute deep.
        main = self.get_model()
        models = {"": main}
        runner = getattr(self, "model_runner", None)
        for attribute, holder in vars(runner).items() if runner is not None else ():
            for candidate in (holder, getattr(holder, "model", None)):
                if isinstance(candidate, torch.nn.Module) and candidate is not main:
                    if all(candidate is not known for known in models.values()):
                        models[f"{attribute}:"] = candidate
        for prefix, model in models.items():
            for name, module in model.named_modules():
                if pattern.search(name):
                    state["handles"].append(
                        module.register_forward_hook(hook(prefix + name))
                    )
        self.probe_trace = state
        return {
            "rank": self.rank,
            "hooked": len(state["handles"]),
            "models": list(models),
        }

    def trace_end(self, keep=False):
        """Stop tracing; ``keep`` stores the run as the reference, otherwise compare to it."""
        import torch

        state = self.__dict__.pop("probe_trace")
        for handle in state["handles"]:
            handle.remove()
        prints = torch.stack(state["prints"]).cpu().tolist() if state["prints"] else []
        rows = [list(item) for item in zip(state["names"], state["rows"], prints)]
        if keep:
            self.probe_reference = rows
            return {"rank": self.rank, "kept": len(rows)}
        return {"rank": self.rank, **trace_differences(self.probe_reference, rows)}

    def zero_moe_scratch(self):
        """Diagnostic: hand the Marlin MoE kernel zeroed scratch buffers on every call.

        If repeats become identical, the kernel reads scratch memory it did not write.
        """
        from vllm.model_executor.layers.fused_moe.experts import marlin_moe

        if not hasattr(self, "probe_zeroed"):
            original = marlin_moe.fused_marlin_moe
            self.probe_zeroed = counts = {"calls": 0, "buffers": 0}

            def zeroed(*args, **kwargs):
                counts["calls"] += 1
                for key in (
                    "intermediate_cache13",
                    "intermediate_cache2",
                    "output",
                    "workspace",
                ):
                    if kwargs.get(key) is not None:
                        kwargs[key].zero_()
                        counts["buffers"] += 1
                return original(*args, **kwargs)

            marlin_moe.fused_marlin_moe = zeroed
        return {"rank": self.rank, **self.probe_zeroed}

    def fa2_stage(self, stage):
        return {"rank": self.rank, "stage": fa2_stage(stage)}

    def host_census(self):
        return {"rank": self.rank, **census()}

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
