import unittest

from glm53_setup.runtime import patch_prefix_dedup

# The lines of vLLM 385dce36 v1/core/block_pool.py that the patch touches.
SOURCE = """from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockPool:
    def _insert_block_hash(
        self,
        block_hash_with_group_id,
        block,
        num_tokens,
    ) -> None:
        if block.block_hash == block_hash_with_group_id:
            return

        if self.cached_block_hash_to_block.contain(
            block_hash_with_group_id, block.block_id
        ):
            return

        if block.block_hash is None:
            block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)
"""


class PatchTests(unittest.TestCase):
    def test_the_pool_asks_before_registering_a_page_under_a_cached_hash(self):
        patched = patch_prefix_dedup.patch_text(SOURCE)
        self.assertIn("skip_duplicate_page as glm53_skip_duplicate_page", patched)
        self.assertIn("if glm53_skip_duplicate_page(", patched)
        self.assertTrue(patched.startswith("# Modified by GLM setup"))
        # The question comes after the pinned checks and before the insert.
        self.assertLess(
            patched.index("block_hash_with_group_id, block.block_id"),
            patched.index("if glm53_skip_duplicate_page("),
        )
        self.assertLess(
            patched.index("if glm53_skip_duplicate_page("),
            patched.index("self.cached_block_hash_to_block.insert("),
        )

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_prefix_dedup.patch_text(
                SOURCE.replace("block.block_id\n", "block.id\n")
            )
        with self.assertRaises(ValueError):
            patch_prefix_dedup.patch_text(patch_prefix_dedup.patch_text(SOURCE))


if __name__ == "__main__":
    unittest.main()
