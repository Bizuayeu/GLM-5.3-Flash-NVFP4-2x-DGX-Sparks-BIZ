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
import json
import threading
import traceback
from pathlib import Path

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


def gathers_cache_rows(query_rows):
    """Whether a traced attention call may gather the cache rows it touches.

    The gather is rows x candidates x 656 bytes on the device: megabytes for a
    verification step, gigabytes for a prefill chunk, per MLA layer. On GB10 that
    memory is the host's, and the supervisor's two-second watch is too slow for it.
    """
    from glm53_setup.runtime.fa2_attention import DECODE_MAX_ROWS

    return query_rows <= DECODE_MAX_ROWS


def whole_words(nbytes):
    """Bytes of a tensor that a fingerprint reads as int64 words; the rest is a tail.

    ``bytes.sum(dtype=int64)`` materialises the cast, eight times the tensor on
    the device; a sum over int64 words (wrapping) allocates nothing.
    """
    return nbytes - nbytes % 8


def flat_bytes(tensor, torch):
    """A tensor's bytes as one row; flattened first, so a scalar (0-dim) can be viewed."""
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def tensor_fingerprint(tensor):
    """Two byte sums of a tensor, kept on its device; equal bits give equal prints."""
    import torch

    data = flat_bytes(tensor, torch)
    if data.storage_offset() % 8:
        # A view as int64 needs an aligned offset; a copy (one times the
        # tensor) keeps the fingerprint independent of where it sat.
        data = data.clone()
    split = whole_words(data.numel())
    words = data[:split].view(torch.int64)
    tail = data[split:].to(torch.int64).sum()
    return words.sum() * 1000003 + words[1::3].sum() + tail


LAYER = r"^((?:[A-Za-z_]+:)?(?:[A-Za-z_]+\.)*layers\.\d+)\."


def layer_of(name):
    """The layer a tensor belongs to, or its first component outside the layers."""
    import re

    match = re.match(LAYER, name)
    if match:
        return match.group(1)
    head = name.split(".")
    return ".".join(head[:2]) if head[0] == "model" and len(head) > 2 else head[0]


def digest_summary(rows):
    """Per-layer and overall digests of ``[name, elements, fingerprint]`` rows.

    Each digest is the SHA-256 of the sorted ``name:fingerprint`` lines, so the
    order the tensors were walked in does not matter and one changed tensor
    changes its layer's digest and the overall one.
    """
    import hashlib

    layers = {}
    for name, _, fingerprint in rows:
        layers.setdefault(layer_of(name), []).append(f"{name}:{fingerprint}")

    def digest(lines):
        return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()[:16]

    return {
        "tensors": len(rows),
        "elements": sum(int(elements) for _, elements, _ in rows),
        "layers": {layer: digest(lines) for layer, lines in sorted(layers.items())},
        "overall": digest([f"{name}:{fingerprint}" for name, _, fingerprint in rows]),
    }


def digest_differences(reference, rows):
    """Which tensors moved between two digests of the same model, by name."""
    before = {name: fingerprint for name, _, fingerprint in reference}
    after = {name: fingerprint for name, _, fingerprint in rows}
    differing = sorted(n for n in before if n in after and before[n] != after[n])
    layers = {}
    for name in differing:
        layers[layer_of(name)] = layers.get(layer_of(name), 0) + 1
    return {
        "differing": differing,
        "missing": sorted(n for n in before if n not in after),
        "added": sorted(n for n in after if n not in before),
        "same": sum(1 for n in before if n in after and before[n] == after[n]),
        "layers": layers,
    }


def probe_models(worker, module_type):
    """The main model and every other torch module one attribute deep in the runner.

    The draft model hangs off the runner under a version-dependent name.
    """
    main = worker.get_model()
    models = {"": main}
    runner = getattr(worker, "model_runner", None)
    for attribute, holder in vars(runner).items() if runner is not None else ():
        for candidate in (holder, getattr(holder, "model", None)):
            if isinstance(candidate, module_type) and candidate is not main:
                if all(candidate is not known for known in models.values()):
                    models[f"{attribute}:"] = candidate
    return models


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


INDEXER_SHAPES = {
    "heads": 32,
    "dim": 128,
    "kpool": 4,
    "select": 512,
    "pools": 540,
    "block": 64,
}


def bytes_digest(tensor, torch):
    """SHA-256 of a tensor's bytes, 16 hex digits; equal bits give equal digests."""
    import hashlib

    return hashlib.sha256(
        flat_bytes(tensor, torch).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def indexer_kernel_hashes(
    torch, kpool_ops, deep_gemm, sms, seed=0, shapes=INDEXER_SHAPES
):
    """Hash the kpool indexer's computations on fixed inputs, inside this process.

    The request trace of 2026-09-23 named one rank's copy of the replicated
    indexer as the first call that differed between two launches on identical
    inputs, and thirteen fresh processes on one GB10 computed these same
    kernels bit-identically. Run on both ranks of a live launch, this says
    whether the serving processes themselves agree, and which computation does
    not: the fp32 head gate and bf16 gate score, the fused FWHT quantisation,
    the pool cache's prefill write and decode tail update, and DeepGEMM's paged
    MQA logits over that cache with the stable top-k of its rows.
    """
    heads, dim, kpool = shapes["heads"], shapes["dim"], shapes["kpool"]
    select, pools, block = shapes["select"], shapes["pools"], shapes["block"]
    hidden = 4096
    dev = torch.device("cuda")

    def bf16(*shape, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=g).to(torch.bfloat16).to(dev)

    def f32(*shape, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=g).to(dev)

    hashes = {}
    wp = f32(hidden, heads, seed=seed + 21) * 0.02
    gate = bf16(dim, hidden, seed=seed + 22) * 0.02
    for rows, s in ((4, 23), (2048, 24)):
        h = bf16(rows, hidden, seed=seed + s)
        hashes[f"head_gate_fp32_rows{rows}"] = bytes_digest(
            torch.mm(h.float(), wp), torch
        )
        hashes[f"gate_score_bf16_rows{rows}"] = bytes_digest(
            torch.nn.functional.linear(h, gate), torch
        )
    for rows, s in ((4, 25), (2048, 26)):
        q_fp8, q_scale = kpool_ops.fwht128_quant_fp8(
            bf16(rows * heads, dim, seed=seed + s)
        )
        hashes[f"fwht_q_fp8_rows{rows}"] = bytes_digest(q_fp8, torch)
        hashes[f"fwht_q_scale_rows{rows}"] = bytes_digest(q_scale, torch)
    num_blocks = pools // block + 2
    cache = torch.zeros(num_blocks, block, dim + 4, dtype=torch.uint8, device=dev)
    ape = f32(kpool, dim, seed=seed + 27) * 0.1
    kpool_ops.kpool_compress_and_write_cache(
        cache,
        bf16(pools, kpool, dim, seed=seed + 28),
        bf16(pools, kpool, dim, seed=seed + 29),
        ape,
        torch.arange(pools, dtype=torch.int64, device=dev),
        pool_size=kpool,
        head_dim=dim,
        round_scale=True,
    )
    hashes["compress_write_prefill"] = bytes_digest(cache, torch)
    tail = torch.zeros(num_blocks, 2, kpool, dim, dtype=torch.bfloat16, device=dev)
    int32 = lambda rows: torch.tensor(rows, dtype=torch.int32, device=dev)  # noqa: E731
    kpool_ops.kpool_decode_update_and_maybe_write_cache_batched(
        cache,
        tail,
        int32([[3, 3, 3, 3]]),
        bf16(1, 4, dim, seed=seed + 30),
        bf16(1, 4, dim, seed=seed + 31),
        ape,
        int32([[-1, -1, -1, pools]]),
        int32([[2160, 2161, 2162, 2163]]),
        kpool,
        head_dim=dim,
        round_scale=True,
    )
    hashes["decode_update_cache"] = bytes_digest(cache, torch)
    hashes["decode_update_tail"] = bytes_digest(tail, torch)
    q_fp8, _ = kpool_ops.fwht128_quant_fp8(bf16(4 * heads, dim, seed=seed + 32))
    weights = (f32(4, heads, seed=seed + 33) * 0.05).contiguous()
    context = int32([[pools - 3, pools - 2, pools - 1, pools]])
    blocks = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(1, -1)
    meta = deep_gemm.get_paged_mqa_logits_metadata(context, block, sms)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_fp8.view(1, 4, heads, dim), None),
        cache.unsqueeze(-2),
        weights,
        context,
        blocks,
        meta,
        block * 1024,
        clean_logits=False,
    )
    valid = logits[:, :pools].float().clone()
    order = torch.sort(valid, dim=-1, descending=True, stable=True).indices[:, :select]
    hashes["paged_mqa_logits"] = bytes_digest(valid, torch)
    hashes["paged_mqa_topk_set"] = bytes_digest(order.sort(dim=-1).values, torch)
    torch.cuda.synchronize()
    return hashes


def indexer_layer(model, layer):
    """The kpool indexer of one MLA layer and the attention that owns it, by name."""
    suffix = f"layers.{layer}.self_attn.indexer"
    for name, module in model.named_modules():
        if name.endswith(suffix):
            owner = dict(model.named_modules())[name[: -len(".indexer")]]
            return module, owner
    raise KeyError(f"no module named *.{suffix}")


def indexer_rope(owner):
    """The indexer's rotary embedding and where it was found, or ``(None, "none")``.

    The attention module keeps it as ``indexer_rope_emb``; the MLA wrapper it
    builds keeps the same object as ``indexer_rotary_emb`` (1.11.1 looked on the
    owner only and recorded no rope hash on the served model).
    """
    for path in ("indexer_rope_emb", "mla_attn.indexer_rotary_emb"):
        found = owner
        for part in path.split("."):
            found = getattr(found, part, None)
        if found is not None:
            return found, path
    return None, "none"


def indexer_stage_hashes(torch, glm_attention, model, layer, seed=0, rows=2048):
    """Hash the stages between the indexer's projections and its op, with the layer's own weights.

    The deep trace of 2026-09-23 (Probe4, state 2) showed the two ranks' copies
    of the replicated indexer agreeing on every projection output and on the
    fp8 query, the head gate and the hidden state, and disagreeing on the key
    handed to the indexer op at prefill in eight of the eleven MLA layers.
    Between the projection and the op stand the compiled layer norm
    (``_fused_indexer_k_norm``, an Inductor kernel) and the indexer's rotary
    embedding; this runs both on fixed inputs with the layer's weights, the norm
    also in eager fp32 as the reference.

    The served call normalises ``kw[:, :head_dim]``, a strided view of the
    projection output, and Inductor compiles a strided input into a kernel of
    its own, autotuned per host (2026-09-24: ``XBLOCK`` 8 on one host and 1 on
    the other, which differ in bits). ``k_norm_served_rows*`` hash that call at
    prefill, two-sequence and one-sequence decode widths; ``k_norm_compiled``
    (a contiguous input, the 1.11.1 key) runs a different kernel and stays for
    comparison with its records.
    """
    indexer, owner = indexer_layer(model, layer)
    dim = indexer.head_dim
    dev = indexer.k_norm.weight.device
    norm = indexer.k_norm
    projection = indexer.wk_weights_proj
    width = getattr(projection, "input_size", None) or projection.weight.shape[1]

    def bf16(*shape, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=g).to(torch.bfloat16).to(dev)

    def compiled_norm(x):
        return glm_attention._fused_indexer_k_norm(
            x, norm.weight, norm.bias, dim, norm.eps
        )

    x = bf16(rows, dim, seed=seed + 41)
    hashes = {}
    hashes["k_norm_compiled"] = bytes_digest(compiled_norm(x), torch)
    hashes["k_norm_eager_fp32"] = bytes_digest(
        torch.nn.functional.layer_norm(
            x.float(), (dim,), norm.weight, norm.bias, norm.eps
        ).type_as(x),
        torch,
    )
    rope, hashes["rope_source"] = indexer_rope(owner)
    if rope is not None and indexer.rope_dim > 0:
        positions = torch.arange(rows, device=dev)
        q_pe = bf16(rows, indexer.n_head, indexer.rope_dim, seed=seed + 42)
        k_pe = bf16(rows, 1, indexer.rope_dim, seed=seed + 43)
        q_out, k_out = rope(positions, q_pe, k_pe)
        hashes["rope_q"] = bytes_digest(q_out, torch)
        hashes["rope_k"] = bytes_digest(k_out, torch)
    for n in (rows, 8, 4):
        kw, _ = projection(bf16(n, width, seed=seed + 44))
        if n == rows:
            hashes["wk_weights_proj"] = bytes_digest(kw, torch)
            hashes["k_norm_served_eager_fp32"] = bytes_digest(
                torch.nn.functional.layer_norm(
                    kw[:, :dim].float(), (dim,), norm.weight, norm.bias, norm.eps
                ).type_as(kw),
                torch,
            )
        hashes[f"k_norm_served_rows{n}"] = bytes_digest(
            compiled_norm(kw[:, :dim]), torch
        )
    torch.cuda.synchronize()
    return hashes


def autotuner_config(config):
    return {
        **config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def autotuner_rows(objects, autotuner_type, pattern=None):
    """The Inductor autotuners among ``objects``: each kernel's file, candidate configs and served launchers.

    ``launchers`` is what the process actually runs: after a cached best config
    or a benchmark, one. ``configs`` is empty once precompiled (torch 2.13 drops
    the candidates). Read from a live worker, it says which config each rank
    serves without inferring it from the files on disk.
    """
    rows = []
    for obj in objects:
        try:
            if not isinstance(obj, autotuner_type):
                continue
        except ReferenceError:  # a weak proxy whose referent is gone
            continue
        name = (getattr(obj, "inductor_meta", None) or {}).get("kernel_name", "")
        if pattern and pattern not in name:
            continue
        filename = getattr(obj, "filename", None)
        hints = getattr(obj, "size_hints", None)
        rows.append(
            {
                "kernel": name,
                "file": "/".join(Path(filename).parts[-2:]) if filename else None,
                "size_hints": dict(hints) if hints else None,
                "configs": [autotuner_config(c) for c in obj.configs or []],
                "launchers": [
                    autotuner_config(launcher.config)
                    for launcher in getattr(obj, "launchers", None) or []
                ],
                # "hit" (a saved best config), "miss" (benchmarked in this
                # process), "only 1 config", or None before any lookup.
                "cache": (getattr(obj, "autotune_cache_info", None) or {}).get(
                    "autotune_cache_state"
                ),
                # What codegen wrote into the kernel: True filters reduction
                # configs to one before any timing.
                "deterministic": (getattr(obj, "inductor_meta", None) or {}).get(
                    "deterministic"
                ),
            }
        )
    return sorted(rows, key=lambda r: (r["kernel"], r["file"] or ""))


INDUCTOR_SETTINGS = (
    "deterministic",
    "batch_invariant",
    "compile_threads",
    "worker_start_method",
    "autotune_local_cache",
    "force_disable_caches",
)


class WatchedOverride:
    """Stands in for a torch config entry's ``user_override`` ContextVar and records who writes it.

    torch 2.13 keeps each config override in a ContextVar that ``setattr``,
    ``config.patch`` and ``load_config`` all reach through ``set``/``reset``, so
    this one seam catches every writer, with its stack.
    """

    def __init__(self, inner, name, writes, limit):
        self.inner, self.name, self.writes, self.limit = inner, name, writes, limit

    def _note(self, op, value):
        if len(self.writes) < self.limit:
            self.writes.append(
                {
                    "name": self.name,
                    "op": op,
                    "value": None if op == "reset" else repr(value),
                    "thread": threading.current_thread().name,
                    "stack": [
                        line.strip() for line in traceback.format_stack()[-14:-2]
                    ],
                }
            )

    def get(self, *default):
        return self.inner.get(*default)

    def set(self, value):
        self._note("set", value)
        return self.inner.set(value)

    def reset(self, token):
        self._note("reset", None)
        return self.inner.reset(token)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def watch_config_entry(config_module, name, writes, limit=40):
    """Record every write to one torch config entry (once per entry)."""
    entry = config_module._config[name]
    if not isinstance(entry.user_override, WatchedOverride):
        entry.user_override = WatchedOverride(entry.user_override, name, writes, limit)


# Writes to Inductor's deterministic switch in this process, from the moment the
# worker loads this module (before the model is loaded).
CONFIG_WRITES = []
try:
    import torch._inductor.config as _inductor_config

    for _name in ("deterministic", "batch_invariant"):
        watch_config_entry(_inductor_config, _name, CONFIG_WRITES)
except Exception:  # noqa: BLE001 - no torch here (tests), or a torch without _config
    pass


def inductor_state(inductor_config, environ):
    """The Inductor settings codegen reads in this process, next to the TORCHINDUCTOR_* environment.

    On 2026-09-24 a worker whose environment carried TORCHINDUCTOR_DETERMINISTIC=1
    generated every kernel with ``'deterministic': False``; this reads both sides
    in the serving process instead of a fresh one.
    """
    config = {k: getattr(inductor_config, k, None) for k in INDUCTOR_SETTINGS}
    env = {k: v for k, v in sorted(environ.items()) if k.startswith("TORCHINDUCTOR_")}
    wanted = env.get("TORCHINDUCTOR_DETERMINISTIC") == "1"
    state = {
        "config": config,
        "environ": env,
        "environ_disagrees": wanted != bool(config["deterministic"]),
    }
    entry = getattr(inductor_config, "_config", {}).get("deterministic")
    if entry is not None:
        override = entry.user_override.get()
        unset = type(override) is object  # torch's _UNSET_SENTINEL
        state["entry"] = {
            "default": entry.default,
            "user_override": None if unset else repr(override),
            "unset": unset,
            # env_value_force wins over any override: set by inductor_pin.
            "forced": None
            if type(entry.env_value_force) is object
            else repr(entry.env_value_force),
        }
    if hasattr(inductor_config, "codegen_config"):
        state["codegen_config"] = inductor_config.codegen_config()[:4000]
    return state


def autotuner_differences(ranks):
    """Kernel files whose served launchers differ between the ranks (a file one rank lacks counts)."""
    per_rank = [
        {
            r["file"]: json.dumps(r["launchers"], sort_keys=True)
            for r in row["autotuners"]
        }
        for row in ranks
    ]
    files = sorted({f for launchers in per_rank for f in launchers})
    return [f for f in files if len({launchers.get(f) for launchers in per_rank}) > 1]


def kernel_hash_differences(ranks):
    """Which hashes differ between the ranks of one launch (each row ``{"rank", "hashes"}``)."""
    keys = sorted({k for row in ranks for k in row["hashes"]})
    differing = [k for k in keys if len({row["hashes"].get(k) for row in ranks}) > 1]
    return {
        "ranks": [row["rank"] for row in ranks],
        "agree": not differing,
        "differing": differing,
    }


class MemoryProbeWorker:
    def trace_begin(self, pattern=None, inputs=False, sync=False, functions=True):
        """Fingerprint what the traced modules take and return in what runs next.

        Fingerprints are two byte sums kept on the GPU, so a whole request fits
        where clones would not. Every tensor of an output is taken, integer ones
        too (an indexer returns candidate indices); ``inputs`` adds the tensors a
        module receives; ``functions`` also wraps the sparse NoPE attention, which
        is a function and not a module; ``sync`` synchronises after every traced
        call, so a difference that then disappears was a race between streams.
        """
        import re
        import sys

        import torch

        pattern = re.compile(pattern or TRACED)
        state = {"handles": [], "names": [], "rows": [], "prints": [], "patched": []}

        fingerprint = tensor_fingerprint

        def note(name, tensor):
            if tensor.numel() == 0 or tensor.is_meta:
                return
            state["names"].append(name)
            state["rows"].append(int(tensor.shape[0]) if tensor.ndim else 1)
            # Candidate indices come out of the top-k kernel in a different order
            # every call and are sorted before attention: compare them as sets.
            if tensor.ndim == 2 and tensor.dtype in (torch.int32, torch.int64):
                tensor = tensor.sort(dim=-1).values
            state["prints"].append(fingerprint(tensor))

        def note_all(name, values):
            items = values if isinstance(values, (tuple, list)) else [values]
            for index, item in enumerate(items):
                if isinstance(item, torch.Tensor):
                    note(f"{name}[{index}]" if index else name, item)

        def hook(name):
            def after(module, args, output):
                if inputs or name.endswith("embed_tokens"):
                    note_all(name + ":in", args)
                note_all(name, output)
                if sync:
                    torch.cuda.synchronize()

            return after

        if functions:
            # The attention core is a plain function; callers may hold it by name,
            # so replace it in every module that refers to the same object.
            import glm53_reference

            original = glm53_reference.sparse_nope_reference

            def traced(query, packed_cache, physical_indices, scale, **kwargs):
                note("sparse_nope:query", query)
                # Physical slots depend on where the blocks of this request landed;
                # the number of candidates per row and the rows' contents do not.
                note(
                    "sparse_nope:candidates_per_row", (physical_indices >= 0).sum(dim=1)
                )
                if gathers_cache_rows(query.shape[0]):
                    flat = physical_indices.reshape(-1)
                    touched = packed_cache.reshape(-1, 656)[flat.clamp_min(0).long()]
                    # Padding gathers slot 0, which belongs to whoever wrote it last.
                    touched[flat < 0] = 0
                    # One row per query row, so the entry's row count is the call's.
                    note("sparse_nope:cache_rows", touched.view(query.shape[0], -1))
                result = original(
                    query, packed_cache, physical_indices, scale, **kwargs
                )
                note("sparse_nope:output", result)
                if sync:
                    torch.cuda.synchronize()
                return result

            for module in list(sys.modules.values()):
                for attribute, value in list(getattr(module, "__dict__", {}).items()):
                    if value is original:
                        setattr(module, attribute, traced)
                        state["patched"].append((module, attribute, original))

        models = probe_models(self, torch.nn.Module)
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
            "functions": len(state["patched"]),
            "models": list(models),
        }

    def trace_end(self, keep=False, export=False):
        """Stop tracing; ``keep`` stores the run as the reference, otherwise compare to it.

        ``export`` returns the rows themselves, so a later launch can be compared
        with ``trace_differences`` against a record instead of this process.
        """
        import torch

        state = self.__dict__.pop("probe_trace")
        for handle in state["handles"]:
            handle.remove()
        for module, attribute, original in state["patched"]:
            setattr(module, attribute, original)
        prints = torch.stack(state["prints"]).cpu().tolist() if state["prints"] else []
        rows = [list(item) for item in zip(state["names"], state["rows"], prints)]
        if keep:
            self.probe_reference = rows
            return {
                "rank": self.rank,
                "kept": len(rows),
                **({"rows": rows} if export else {}),
                "function_entries": sum(
                    row[0].startswith("sparse_nope:") for row in rows
                ),
                # The largest call the run carried, and the largest whose cache
                # rows were gathered: the cost of a reading is set by the former.
                "max_rows": max((row[1] for row in rows), default=0),
                "max_gathered_rows": max(
                    (row[1] for row in rows if row[0] == "sparse_nope:cache_rows"),
                    default=0,
                ),
            }
        return {"rank": self.rank, **trace_differences(self.probe_reference, rows)}

    def weight_digest(self, tensors=False):
        """Fingerprint every parameter and buffer as loaded, per layer and overall.

        Taken after a launch, it says whether two launches computed from the same
        bits; ``tensors`` returns the rows, so a later launch's record names the
        tensors that moved (``digest_differences``). The prints stay on the device
        until one transfer at the end, as the trace does; ``copied`` lists the
        tensors whose fingerprint needed a copy (non-contiguous or unaligned).
        """
        import torch

        names, elements, prints, copied = [], [], [], []
        models = probe_models(self, torch.nn.Module)
        for prefix, model in models.items():
            for name, tensor in (*model.named_parameters(), *model.named_buffers()):
                if not tensor.is_contiguous() or tensor.storage_offset() % 8:
                    copied.append(prefix + name)
                names.append(prefix + name)
                elements.append(int(tensor.numel()))
                prints.append(tensor_fingerprint(tensor))
        values = torch.stack(prints).cpu().tolist() if prints else []
        rows = [list(item) for item in zip(names, elements, values)]
        result = {
            "rank": self.rank,
            "models": list(models),
            "copied": copied,
            "summary": digest_summary(rows),
        }
        if tensors:
            result["rows"] = rows
        return result

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

    def autotuners(self, pattern=None):
        """The Inductor autotuners alive in this worker and the configs they serve."""
        import gc

        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        rows = autotuner_rows(gc.get_objects(), CachingAutotuner, pattern)
        return {"rank": self.rank, "autotuners": rows}

    def inductor_state(self):
        """Inductor's settings and TORCHINDUCTOR_* environment in this worker, and one fresh compile.

        The compile (a layer norm 96 wide, a shape the model does not use) shows
        whether codegen in this process now writes the deterministic mode into
        the kernels it generates.
        """
        import gc
        import os

        import torch
        import torch._inductor.config as inductor
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        row = inductor_state(inductor, os.environ)
        row["rank"] = self.rank
        row["config_writes"] = list(CONFIG_WRITES)
        try:
            before = {
                r["file"] for r in autotuner_rows(gc.get_objects(), CachingAutotuner)
            }

            def norm(x, w, b):
                return torch.nn.functional.layer_norm(x.float(), (96,), w, b, 1e-6).to(
                    x.dtype
                )

            compiled = torch.compile(norm, dynamic=True)
            with torch.inference_mode():
                x = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
                w = torch.ones(96, device="cuda")
                b = torch.zeros(96, device="cuda")
                compiled(x, w, b)
                compiled(x[:32], w, b)
                torch.cuda.synchronize()
            row["probe_kernels"] = [
                r
                for r in autotuner_rows(gc.get_objects(), CachingAutotuner)
                if r["file"] not in before
            ]
        except Exception as error:  # noqa: BLE001
            row["probe_error"] = repr(error)[:300]
        return row

    def kernel_hashes(self, seed=0, layer=19):
        """The indexer's computations on fixed inputs, hashed inside this worker.

        Compared across the ranks of one launch (``kernel_hash_differences``)
        and with an earlier launch's record, it names the computation that a
        launch in another numerical state does differently, on the spot.
        """
        import torch
        from vllm.models.glm5next.nvidia.ops import kpool_compress
        from vllm.utils import deep_gemm

        sms = torch.cuda.get_device_properties(0).multi_processor_count
        # Served forwards run in inference mode, and Dynamo guards on grad mode:
        # outside it the compiled leaves would be traced and compiled anew.
        with torch.inference_mode():
            hashes = indexer_kernel_hashes(torch, kpool_compress, deep_gemm, sms, seed)
            # 1.11.1: the stages of one real layer, with its weights. A failure
            # here names itself instead of failing the whole answer.
            try:
                from vllm.models.glm5next.nvidia import attention as glm_attention

                stages = indexer_stage_hashes(
                    torch, glm_attention, self.get_model(), layer, seed
                )
                hashes.update({f"layer{layer}_{k}": v for k, v in stages.items()})
            except Exception as error:  # noqa: BLE001
                hashes[f"layer{layer}_error"] = repr(error)[:300]
        return {"rank": self.rank, "seed": seed, "sms": sms, "hashes": hashes}
