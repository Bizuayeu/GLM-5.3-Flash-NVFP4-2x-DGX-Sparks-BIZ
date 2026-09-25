import unittest

from glm53_setup.runtime import patch_kpool_seed

# The lines of vLLM 385dce36 models/glm5next/nvidia/ops/kpool_compress.py that the patch touches.
SOURCE = """import torch
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
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    blk = t // KPOOL
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
    TAIL_BLOCK_ELEMS: tl.constexpr,
):
    pass
"""


class PatchTests(unittest.TestCase):
    def test_the_seed_addresses_blocks_by_the_tail_strides(self):
        patched = patch_kpool_seed.patch_text(SOURCE)
        self.assertIn(
            "    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM\n", patched
        )
        self.assertIn("tl.store(tail_ptr + base + KPOOL_HEAD + offs, s", patched)
        self.assertIn("TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),", patched)
        self.assertIn("KPOOL_HEAD=tail_kv_cache.stride(1),", patched)
        self.assertNotIn("(blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM", patched)
        self.assertNotIn("base + KPOOL * HEAD_DIM", patched)
        self.assertTrue(patched.startswith("# Modified by GLM setup"))

    def test_the_wrapper_refuses_a_tail_it_cannot_address(self):
        patched = patch_kpool_seed.patch_text(SOURCE)
        self.assertIn("assert tail_kv_cache.ndim == 4", patched)
        self.assertIn("tail_kv_cache.stride(2) == head_dim", patched)

    def test_the_decode_kernel_is_left_alone(self):
        # The decode kernel already takes TAIL_BLOCK_ELEMS; only the seed kernel changes.
        patched = patch_kpool_seed.patch_text(SOURCE)
        decode = patched[patched.index("def _kpool_decode_update_batched_kernel") :]
        self.assertEqual(decode, SOURCE[SOURCE.index("def _kpool_decode_update") :])

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_kpool_seed.patch_text(SOURCE.replace("t % KPOOL) * HEAD_DIM", "t)"))
        with self.assertRaises(ValueError):
            patch_kpool_seed.patch_text(patch_kpool_seed.patch_text(SOURCE))


if __name__ == "__main__":
    unittest.main()
