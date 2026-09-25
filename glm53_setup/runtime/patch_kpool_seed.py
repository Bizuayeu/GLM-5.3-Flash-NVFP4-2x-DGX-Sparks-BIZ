"""Source-pinned patch: the kpool prefill seed addresses tail blocks by the tail's own strides.

The indexer's raw-tail cache shares the indexer cache allocation with the indexer's padded block
stride (38,016 B per block for GLM-5.3-Flash), but vLLM 385dce36's ``_kpool_tail_seed_kernel``
addresses blocks as a dense ``[num_blocks, 2, KPOOL, HEAD_DIM]`` array (2,048 B per block). Every
prefill therefore leaves the request's own tail block unseeded and writes the last tokens' raw
key and gate, 2,048 B per block, into the low-numbered indexer block at ``block * 2048 / 38016``,
which may belong to another live or cached request. The MLA KV is untouched; the damage reaches the
sparse top-k selection once a context exceeds ``index_topk``. Upstream: vllm-project/vllm pull
request #57477 (merged 2026-09-20 as ``db1bfdd4``), which addresses blocks through
``tail.stride(0)`` and ``tail.stride(1)`` as the decode kernel already did; this is the same change
on the pinned source.
"""

# cc-defer: carries the upstream fix on the pinned source; drop it (and the
# Dockerfile RUN) when the vLLM pin moves past db1bfdd4.

from . import pinned_patch
from .pinned_patch import replace_once

TARGET = "models/glm5next/nvidia/ops/kpool_compress.py"
SOURCE_SHA256 = "bd6f1d7b5e88a10d9756799a744d3e201421b6caabd32f7b3319d31904a91a0a"
MISMATCH = "kpool seed source hash mismatch"
RECORD = "glm53-kpool-seed-patch.json"
SIGNATURE = "    tail_ptr,\n    n_tokens,\n    HEAD_DIM: tl.constexpr,\n"
STRIDED_SIGNATURE = (
    "    tail_ptr,\n"
    "    n_tokens,\n"
    "    TAIL_BLOCK_ELEMS: tl.constexpr,\n"
    "    KPOOL_HEAD: tl.constexpr,\n"
    "    HEAD_DIM: tl.constexpr,\n"
)
BASE = "    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM\n"
STRIDED_BASE = "    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM\n"
SCORE = "    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)\n"
STRIDED_SCORE = "    tl.store(tail_ptr + base + KPOOL_HEAD + offs, s, mask=m)\n"
CHECKS = (
    "    assert tail_kv_cache.dtype == torch.bfloat16\n"
    "    assert key.dtype == torch.bfloat16\n"
)
STRIDE_CHECKS = (
    "    assert tail_kv_cache.dtype == torch.bfloat16\n"
    "    assert tail_kv_cache.ndim == 4 and tail_kv_cache.shape[1] == 2\n"
    "    assert tail_kv_cache.stride(3) == 1 and tail_kv_cache.stride(2) == head_dim\n"
    "    assert key.dtype == torch.bfloat16\n"
)
LAUNCH = "        tail_kv_cache,\n        n,\n        HEAD_DIM=head_dim,\n"
STRIDED_LAUNCH = (
    "        tail_kv_cache,\n"
    "        n,\n"
    "        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n"
    "        KPOOL_HEAD=tail_kv_cache.stride(1),\n"
    "        HEAD_DIM=head_dim,\n"
)
HEADER = (
    "# Modified by GLM setup: the kpool prefill seed addresses tail blocks by the tail's strides\n"
    "# (vllm-project/vllm #57477). Original vLLM Apache-2.0 notices below remain applicable.\n"
)


def patch_text(text):
    if STRIDED_BASE in text:
        raise ValueError("kpool seed patch already applied")
    patched = text
    for old, new in (
        (SIGNATURE, STRIDED_SIGNATURE),
        (BASE, STRIDED_BASE),
        (SCORE, STRIDED_SCORE),
        (CHECKS, STRIDE_CHECKS),
        (LAUNCH, STRIDED_LAUNCH),
    ):
        patched = replace_once(patched, old, new)
    patched = HEADER + patched
    compile(patched, TARGET, "exec")
    return patched


def prepare(package):
    return pinned_patch.prepare(package, TARGET, SOURCE_SHA256, MISMATCH, patch_text)


def main(argv=None):
    pinned_patch.main(
        argv,
        doc=__doc__,
        target=TARGET,
        sha256=SOURCE_SHA256,
        prepare=prepare,
        record=RECORD,
    )


if __name__ == "__main__":
    main()
