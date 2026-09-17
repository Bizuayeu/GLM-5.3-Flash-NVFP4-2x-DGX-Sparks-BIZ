import importlib.util
import os
import unittest
from unittest.mock import patch

from glm53_setup.runtime import patch_moe_order
from glm53_setup.runtime.moe_token_order import moe_order_enabled

SOURCE = """import torch

def fused_marlin_moe(topk_ids, block_size_m, global_num_experts, expert_map):
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        global_num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )

    assert activation is not None
    return sorted_token_ids
"""


class SwitchTests(unittest.TestCase):
    def test_on_by_default_and_strict_about_values(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLM53_CANONICAL_MOE_ORDER", None)
            self.assertTrue(moe_order_enabled())
        for value, expected in (("1", True), ("0", False)):
            with patch.dict(os.environ, {"GLM53_CANONICAL_MOE_ORDER": value}):
                self.assertIs(moe_order_enabled(), expected)
        with patch.dict(os.environ, {"GLM53_CANONICAL_MOE_ORDER": "yes"}):
            with self.assertRaises(ValueError):
                moe_order_enabled()


class PatchTests(unittest.TestCase):
    def test_inserts_the_guarded_call_once_and_still_compiles(self):
        patched = patch_moe_order.patch_text(SOURCE)
        self.assertEqual(patched.count("canonical_expert_order("), 1)
        self.assertIn("moe_order_enabled() and expert_map is None", patched)
        self.assertLess(
            patched.index("ignore_invalid_experts=True"),
            patched.index("canonical_expert_order("),
        )
        self.assertLess(
            patched.index("canonical_expert_order("),
            patched.index("assert activation is not None"),
        )
        self.assertTrue(patched.startswith("# Modified by GLM setup"))

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_moe_order.patch_text(SOURCE.replace("ignore_invalid", "skip_invalid"))
        with self.assertRaises(ValueError):
            patch_moe_order.patch_text(patch_moe_order.patch_text(SOURCE))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
class OrderTests(unittest.TestCase):
    def test_orders_inside_each_expert_and_leaves_the_unwritten_tail(self):
        import torch

        from glm53_setup.runtime.moe_token_order import canonical_expert_order

        # Blocks of four, experts ascending as the align kernel emits them: expert 2
        # fills one block, expert 5 two. 10 is padding (the number of real entries, as
        # the kernel writes it), the last block is unwritten.
        sorted_ids = torch.tensor([9, 3, 10, 4, 8, 1, 6, 5, 2, 10, 10, 0, 7, 7, 7, 7])
        before = sorted_ids.clone()
        expert_ids = torch.tensor([2, 5, 5, 0])
        result = canonical_expert_order(sorted_ids, expert_ids, torch.tensor(12))
        self.assertEqual(
            result.tolist(), [3, 4, 9, 10, 0, 1, 2, 5, 6, 8, 10, 10, 7, 7, 7, 7]
        )
        self.assertEqual(result.dtype, before.dtype)
        self.assertTrue(torch.equal(sorted_ids, before))


if __name__ == "__main__":
    unittest.main()
