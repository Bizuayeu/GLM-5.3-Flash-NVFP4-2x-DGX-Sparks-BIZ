import unittest

from glm53_setup.runtime import patch_slot_mapping

# The lines of vLLM 385dce36 v1/worker/gpu/block_table.py that the patch touches.
SOURCE = """import triton.language as tl


def _compute_slot_mappings_kernel():
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        block_indices = tl.where(
            mapping_enabled, local_positions // kernel_block_size, 0
        )
        block_offsets = local_positions % kernel_block_size
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=is_local,
            other=0,
        )
        slot_ids = block_numbers * kernel_block_size + block_offsets
        if CP_SIZE != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
"""


class PatchTests(unittest.TestCase):
    def test_the_table_read_is_bounded_by_the_row_width(self):
        patched = patch_slot_mapping.patch_text(SOURCE)
        self.assertIn("in_range = block_indices < block_table_stride", patched)
        # A position the row cannot index reads nothing and gets no slot.
        self.assertIn("mask=is_local & in_range,", patched)
        self.assertIn("slot_ids = tl.where(in_range, slot_ids, PAD_ID)", patched)
        self.assertNotIn("            mask=is_local,\n", patched)
        self.assertTrue(patched.startswith("# Modified by GLM setup"))

    def test_the_bound_is_computed_before_the_read_that_uses_it(self):
        patched = patch_slot_mapping.patch_text(SOURCE)
        self.assertLess(
            patched.index("in_range = "), patched.index("block_numbers = tl.load(")
        )
        self.assertLess(
            patched.index("slot_ids = block_numbers * kernel_block_size"),
            patched.index("slot_ids = tl.where(in_range"),
        )

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_slot_mapping.patch_text(
                SOURCE.replace("mask=is_local,", "mask=local,")
            )
        with self.assertRaises(ValueError):
            patch_slot_mapping.patch_text(patch_slot_mapping.patch_text(SOURCE))


if __name__ == "__main__":
    unittest.main()
