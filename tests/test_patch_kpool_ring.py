import hashlib
import tempfile
import unittest
from pathlib import Path

from glm53_setup.runtime import patch_kpool_ring, patch_kpool_seed

# The lines of vLLM 385dce36 models/glm5next/nvidia/attention.py that the patch touches.
ATTENTION = '''import torch

from vllm.utils.deep_gemm import PAGED_MQA_PAGE_SIZES
from vllm.v1.kv_cache_interface import KpoolTailSpec, MLAAttentionSpec


class Glm5NextTailCache(DeepseekV32IndexerCache):
    """Paged circular buffer for the kpool indexer's in-progress (tail) pool.

    Holds the trailing incomplete pool's raw K + gate score: one block of
    ``index_kpool`` slots per request, overwritten in place by ``pos % kpool``
    as decode/spec-decode advances.
    """

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        # The two head slots form [K, gate score] in the generic
        # [block, head, state, content] cache view.
        return KpoolTailSpec(
            block_size=self._index_kpool,
            num_kv_heads=2,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=self._index_kpool,
        )
'''

# The lines of vLLM 385dce36 models/glm5next/nvidia/ops/kpool_compress.py that the seed patch
# and this patch touch, before either.
KERNELS = '''import torch
import triton
import triton.language as tl


@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy token ``i``'s raw K + gate into its request's tail block.

    Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
    ahead belongs to a different tail block (or is past the batch / padding,
    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is
    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.
    """
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // KPOOL  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    if ahead >= 0 and ahead // KPOOL == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)


def kpool_seed_tail_cache(tail_kv_cache, key, gate_score, tslot, kpool, head_dim=128):
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    n = tslot.shape[0]
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


@triton.jit
def _kpool_decode_update_batched_kernel(
    tail_kv_ptr,
    POOL_SIZE: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
):
    """One program per request; iterates its NEXT_N verify tokens in order.

    programs are independent (distinct tail blocks). With NEXT_N < POOL_SIZE
    (the spec-verify case: NEXT_N ~= num_spec+1, POOL_SIZE=16) at most one
    completion can occur per request per call, but the ordered loop is correct
    for any NEXT_N.
    """
    for t in tl.range(0, NEXT_N):
        slot = safe_pos % POOL_SIZE
        phys_slot = safe_pos % POOL_SIZE
        block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE
        if pos_valid & (slot == POOL_SIZE - 1):
            for pool_slot in tl.static_range(0, POOL_SIZE):
                phys = (pool_logical_start + pool_slot) % POOL_SIZE
            for pool_slot in tl.static_range(0, POOL_SIZE):
                phys = (pool_logical_start + pool_slot) % POOL_SIZE


def kpool_decode_update_and_maybe_write_cache_batched(tail_kv_cache, pool_size):
    assert tail_kv_cache.shape[2] == pool_size
    _kpool_decode_update_batched_kernel[(1,)](
        tail_kv_cache,
        POOL_SIZE=pool_size,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
    )


def expand_pools_and_append_tail(pool_size):
    _expand_kernel[(1,)](
        POOL_SIZE=pool_size,
    )
'''


def seeded():
    return patch_kpool_seed.patch_text(KERNELS)


class AttentionTests(unittest.TestCase):
    def test_the_tail_spec_is_sized_by_the_ring(self):
        patched = patch_kpool_ring.patch_attention(ATTENTION)
        self.assertIn(
            "from vllm.utils.math_utils import cdiv, next_power_of_2\n", patched
        )
        self.assertIn(
            "        span = self._index_kpool + vllm_config.num_speculative_tokens\n",
            patched,
        )
        self.assertIn(
            "        ring = self._index_kpool * next_power_of_2(cdiv(span, self._index_kpool))\n",
            patched,
        )
        self.assertIn("assert self.cache_config.block_size % ring == 0", patched)
        self.assertIn("            block_size=ring,\n", patched)
        self.assertIn("            sliding_window=ring,\n", patched)
        self.assertNotIn("=self._index_kpool,", patched)
        self.assertTrue(patched.startswith("# Modified by GLM setup"))

    def test_the_ring_follows_the_upstream_sizes(self):
        # The formula of the patched spec: 4 without speculation, 8 for 1..4, 16 for 5.
        patched = patch_kpool_ring.patch_attention(ATTENTION)
        span_line = next(line for line in patched.splitlines() if "span =" in line)
        ring_line = next(line for line in patched.splitlines() if "ring =" in line)
        expected = {0: 4, 1: 8, 2: 8, 3: 8, 4: 8, 5: 16}
        for k, ring in expected.items():
            with self.subTest(k=k):
                scope = {
                    "cdiv": lambda a, b: -(-a // b),
                    "next_power_of_2": lambda n: 1 << (n - 1).bit_length(),
                    "self": type("Cache", (), {"_index_kpool": 4})(),
                    "vllm_config": type("Config", (), {"num_speculative_tokens": k})(),
                }
                exec(span_line.strip() + "\n" + ring_line.strip(), scope)
                self.assertEqual(scope["ring"], ring)
                # The served cache.block_size of 256 keeps the upstream assert true.
                self.assertEqual(256 % scope["ring"], 0)

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_kpool_ring.patch_attention(
                ATTENTION.replace(
                    "sliding_window=self._index_kpool", "sliding_window=4"
                )
            )
        with self.assertRaises(ValueError):
            patch_kpool_ring.patch_attention(
                patch_kpool_ring.patch_attention(ATTENTION)
            )


class KernelTests(unittest.TestCase):
    def test_the_seed_kernel_addresses_the_ring(self):
        patched = patch_kpool_ring.patch_kernels(seeded())
        seed = patched[
            patched.index("def _kpool_tail_seed_kernel") : patched.index(
                "def kpool_seed_tail_cache"
            )
        ]
        self.assertIn("    RING: tl.constexpr,\n", seed)
        self.assertIn("    blk = t // RING ", seed)
        self.assertIn("    if ahead >= 0 and ahead // RING == blk:\n", seed)
        self.assertIn(
            "    base = blk * TAIL_BLOCK_ELEMS + (t % RING) * HEAD_DIM\n", seed
        )
        # Whether a token is in the tail still looks one pool ahead.
        self.assertIn("tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens", seed)
        self.assertIn("        RING=tail_kv_cache.shape[2],\n", patched)

    def test_the_decode_kernel_addresses_the_ring(self):
        patched = patch_kpool_ring.patch_kernels(seeded())
        self.assertIn("        slot = safe_pos % POOL_SIZE\n", patched)
        self.assertIn("        phys_slot = safe_pos % RING\n", patched)
        self.assertIn(
            "        block = tl.maximum(tail_slot, 0).to(tl.int64) // RING\n", patched
        )
        self.assertEqual(
            patched.count("phys = (pool_logical_start + pool_slot) % RING\n"), 2
        )
        self.assertNotIn("% POOL_SIZE\n                ", patched)
        self.assertIn(
            "    assert ring >= pool_size and ring % pool_size == 0, (ring, pool_size)\n",
            patched,
        )
        self.assertIn("        POOL_SIZE=pool_size,\n        RING=ring,\n", patched)
        self.assertNotIn("assert tail_kv_cache.shape[2] == pool_size", patched)
        # Other kernels keyed on POOL_SIZE are left alone.
        self.assertTrue(
            patched.endswith(
                KERNELS[KERNELS.index("def expand_pools_and_append_tail") :]
            )
        )
        self.assertTrue(
            patched.startswith("# Modified by GLM setup: the kpool raw-tail ring")
        )

    def test_refuses_the_unseeded_pristine_source(self):
        with self.assertRaisesRegex(ValueError, "seed patch first"):
            patch_kpool_ring.patch_kernels(KERNELS)

    def test_refuses_drifted_or_already_patched_source(self):
        # One of the two pool reads gone: the anchor count no longer matches.
        drifted = seeded().replace(
            "phys = (pool_logical_start + pool_slot) % POOL_SIZE\n", "phys = 0\n", 1
        )
        with self.assertRaises(ValueError):
            patch_kpool_ring.patch_kernels(drifted)
        with self.assertRaises(ValueError):
            patch_kpool_ring.patch_kernels(patch_kpool_ring.patch_kernels(seeded()))


class PrepareTests(unittest.TestCase):
    def package(self, directory, attention, kernels):
        package = Path(directory) / "vllm"
        for name, text in (
            (patch_kpool_ring.ATTENTION, attention),
            (patch_kpool_ring.KERNELS, kernels),
        ):
            (package / name).parent.mkdir(parents=True, exist_ok=True)
            (package / name).write_text(text, encoding="utf-8")
        return package

    def test_both_files_are_checked_before_either_is_patched(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.package(directory, ATTENTION, seeded())
            hashes = {
                name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                for name in patch_kpool_ring.SOURCES
            }
            original = dict(patch_kpool_ring.SOURCES)
            try:
                patch_kpool_ring.SOURCES.update(hashes)
                outputs = patch_kpool_ring.prepare(package)
                self.assertEqual(set(outputs), set(hashes))
                patch_kpool_ring.SOURCES[patch_kpool_ring.KERNELS] = "0" * 64
                with self.assertRaisesRegex(
                    ValueError, "kpool ring source hash mismatch"
                ):
                    patch_kpool_ring.prepare(package)
            finally:
                patch_kpool_ring.SOURCES.clear()
                patch_kpool_ring.SOURCES.update(original)

    def test_the_kernel_hash_is_the_seed_patch_output(self):
        # The pinned hashes differ: the ring patch never takes the pristine kernel file.
        self.assertNotEqual(
            patch_kpool_ring.SOURCES[patch_kpool_ring.KERNELS],
            patch_kpool_seed.SOURCE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
