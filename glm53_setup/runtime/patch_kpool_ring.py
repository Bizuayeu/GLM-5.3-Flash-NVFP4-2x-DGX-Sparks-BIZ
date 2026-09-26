"""Source-pinned patch: the kpool raw-tail ring holds the drafts that follow a pool-completing one.

vLLM 385dce36 gives every request one circular tail block of ``index_kpool`` (4) slots and stashes
each speculative draft's raw key and gate into ``pos % kpool`` before the draft is verified. With
``num_speculative_tokens >= 2``, a draft that completes a pool can be rejected while the drafts
behind it have already overwritten the slots of that pool's earlier, committed keys; the redo of the
rejected position then compresses the wrong keys into the indexer cache. The damage is to sparse
top-k selection once decode has built pools past ``index_topk`` (2,048 tokens); the MLA KV is
untouched. Upstream: vllm-project/vllm pull request #58454 (merged 2026-09-25 as ``2617fe93``),
which widens the ring to ``kpool * next_power_of_2(cdiv(kpool + num_spec, kpool))`` slots (4
without speculation, 8 for depths 1 to 4, 16 for depth 5) and addresses the tail by the ring
instead of the pool; upstream calls it a partial fix of issues #56868 and #56605. This is the same
change on the pinned source, in two files: the tail cache spec, and the seed and decode kernels of
``kpool_compress.py``. The kernel file is the one ``patch_kpool_seed`` already rewrote, so its hash
is that of the seed patch's output and this patch applies only after it.
"""

# cc-defer: carries the upstream fix on the pinned source; the ring bound is
# upstream's and ends at depth 5 (the launcher's MTP ceiling). Drop it (and the
# Dockerfile RUN) when the vLLM pin moves past 2617fe93, together with
# patch_kpool_seed.

import argparse
import hashlib
import json
from pathlib import Path

from .pinned_patch import default_package, replace_once

ATTENTION = "models/glm5next/nvidia/attention.py"
KERNELS = "models/glm5next/nvidia/ops/kpool_compress.py"
# ATTENTION as pinned; KERNELS as patch_kpool_seed leaves it (its SOURCE_SHA256 is the pinned file).
SOURCES = {
    ATTENTION: "400d7063db2aa27314c79dce688060710f5ef7bdbafbddfc6698eaaaea3b3a8f",
    KERNELS: "969a6681500a8e20a66715d45b108a9b0e16a4fc9fbd481cdd388445a97a47eb",
}
RECORD = "glm53-kpool-ring-patch.json"
HEADER = (
    "# Modified by GLM setup: the kpool raw-tail ring spans the speculative drafts\n"
    "# (vllm-project/vllm #58454). Original vLLM Apache-2.0 notices below remain applicable.\n"
)

IMPORT = "from vllm.utils.deep_gemm import PAGED_MQA_PAGE_SIZES\n"
RING_IMPORT = IMPORT + "from vllm.utils.math_utils import cdiv, next_power_of_2\n"
SPEC = (
    "        # [block, head, state, content] cache view.\n"
    "        return KpoolTailSpec(\n"
    "            block_size=self._index_kpool,\n"
)
RING_SPEC = (
    "        # [block, head, state, content] cache view.\n"
    "        # Drafts are stashed before acceptance. With a one-pool ring, the\n"
    "        # drafts behind a rejected pool-completing draft overwrite the keys\n"
    "        # read by its redo.\n"
    "        span = self._index_kpool + vllm_config.num_speculative_tokens\n"
    "        ring = self._index_kpool * next_power_of_2(cdiv(span, self._index_kpool))\n"
    "        # ring must divide the attention block size (a multiple of 128).\n"
    "        assert self.cache_config.block_size % ring == 0, (\n"
    '            f"Glm5NextTailCache: cache_config.block_size "\n'
    '            f"({self.cache_config.block_size}) must be a multiple of the "\n'
    '            f"tail ring ({ring})"\n'
    "        )\n"
    "        return KpoolTailSpec(\n"
    "            block_size=ring,\n"
)
DOC = "    ``index_kpool`` slots per request, overwritten in place by ``pos % kpool``\n"
RING_DOC = "    ``ring`` slots per request, overwritten in place by ``pos % ring``\n"
WINDOW = "            sliding_window=self._index_kpool,\n"
RING_WINDOW = "            sliding_window=ring,\n"

# (old, new, occurrences) on the seed-patched kpool_compress.py.
KERNEL_EDITS = (
    (
        "    KPOOL: tl.constexpr,\n    BLOCK_D: tl.constexpr,\n",
        "    KPOOL: tl.constexpr,\n    RING: tl.constexpr,\n    BLOCK_D: tl.constexpr,\n",
        1,
    ),
    (
        '    """Copy token ``i``\'s raw K + gate into its request\'s tail block.\n',
        '    """Copy token ``i``\'s raw K + gate into its request\'s tail ring.\n',
        1,
    ),
    (
        "    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is\n"
        "    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.\n",
        "    slot < 0). ``tslot = block * RING + pos % RING``; the destination is\n"
        "    ``tail[block, {0:K, 1:score}, pos % RING, :]``.\n",
        1,
    ),
    (
        "    blk = t // KPOOL  # t >= 0 here, so trunc == floor\n",
        "    blk = t // RING  # t >= 0 here, so trunc == floor\n",
        1,
    ),
    (
        "    if ahead >= 0 and ahead // KPOOL == blk:\n",
        "    if ahead >= 0 and ahead // RING == blk:\n",
        1,
    ),
    (
        "    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM\n",
        "    base = blk * TAIL_BLOCK_ELEMS + (t % RING) * HEAD_DIM\n",
        1,
    ),
    (
        "        KPOOL=kpool,\n        BLOCK_D=triton.next_power_of_2(head_dim),\n",
        "        KPOOL=kpool,\n"
        "        RING=tail_kv_cache.shape[2],\n"
        "        BLOCK_D=triton.next_power_of_2(head_dim),\n",
        1,
    ),
    (
        "    POOL_SIZE: tl.constexpr,\n    TAIL_BLOCK_ELEMS: tl.constexpr,\n",
        "    POOL_SIZE: tl.constexpr,\n"
        "    RING: tl.constexpr,\n"
        "    TAIL_BLOCK_ELEMS: tl.constexpr,\n",
        1,
    ),
    (
        "    programs are independent (distinct tail blocks). With NEXT_N < POOL_SIZE\n"
        "    (the spec-verify case: NEXT_N ~= num_spec+1, POOL_SIZE=16) at most one\n"
        "    completion can occur per request per call, but the ordered loop is correct\n"
        "    for any NEXT_N.\n",
        "    programs are independent (distinct tail blocks). RING >= POOL_SIZE.\n",
        1,
    ),
    (
        "        phys_slot = safe_pos % POOL_SIZE\n",
        "        phys_slot = safe_pos % RING\n",
        1,
    ),
    (
        "        block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE\n",
        "        block = tl.maximum(tail_slot, 0).to(tl.int64) // RING\n",
        1,
    ),
    (
        "                phys = (pool_logical_start + pool_slot) % POOL_SIZE\n",
        "                phys = (pool_logical_start + pool_slot) % RING\n",
        2,  # the max pass and the softmax pass read the same pool
    ),
    (
        "    assert tail_kv_cache.shape[2] == pool_size\n",
        "    ring = tail_kv_cache.shape[2]\n"
        "    assert ring >= pool_size and ring % pool_size == 0, (ring, pool_size)\n",
        1,
    ),
    (
        "        POOL_SIZE=pool_size,\n        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n",
        "        POOL_SIZE=pool_size,\n"
        "        RING=ring,\n"
        "        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n",
        1,
    ),
)
# The seed patch's rewrite; its absence means the pristine pinned kernel file.
SEEDED = "    base = blk * TAIL_BLOCK_ELEMS + "


def replace_exactly(text, old, new, count):
    """Like ``replace_once`` for an anchor the source repeats a known number of times."""
    if text.count(old) != count:
        raise ValueError("Patch anchor count differs; refusing source drift")
    return text.replace(old, new)


def patch_attention(text):
    if "tail ring (" in text:
        raise ValueError("kpool ring patch already applied")
    patched = replace_once(text, IMPORT, RING_IMPORT)
    patched = replace_once(patched, DOC, RING_DOC)
    patched = replace_once(patched, SPEC, RING_SPEC)
    patched = HEADER + replace_once(patched, WINDOW, RING_WINDOW)
    compile(patched, ATTENTION, "exec")
    return patched


def patch_kernels(text):
    if "RING: tl.constexpr" in text:
        raise ValueError("kpool ring patch already applied")
    if SEEDED not in text:
        raise ValueError("kpool ring patch needs the kpool seed patch first")
    patched = text
    for old, new, count in KERNEL_EDITS:
        patched = replace_exactly(patched, old, new, count)
    patched = HEADER + patched
    compile(patched, KERNELS, "exec")
    return patched


PATCHES = {ATTENTION: patch_attention, KERNELS: patch_kernels}


def prepare(package):
    """Check both files against their pinned hashes before patching either."""
    originals = {name: (package / name).read_bytes() for name in SOURCES}
    for name, data in originals.items():
        if hashlib.sha256(data).hexdigest() != SOURCES[name]:
            raise ValueError("kpool ring source hash mismatch: " + name)
    return {
        name: PATCHES[name](data.decode("utf-8")).encode("utf-8")
        for name, data in originals.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    package = args.package or default_package()
    outputs = prepare(package)
    row = {
        name: {
            "source_sha256": SOURCES[name],
            "patched_sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in outputs.items()
    }
    if not args.check:
        for name, data in outputs.items():
            (package / name).write_bytes(data)
        (package.parent / RECORD).write_text(json.dumps(row, indent=2))
    print(json.dumps({"files": row, "check_only": args.check}))


if __name__ == "__main__":
    main()
