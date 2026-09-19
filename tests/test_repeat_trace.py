import importlib.util
import unittest

from glm53_setup.validation.run_repeat_trace import HOOKED, summarize


def row(order, module, max_abs, shape_changed=False):
    return {
        "order": order,
        "module": module,
        "call": 0,
        "rows": 4,
        "max_abs": max_abs,
        "differing_elements": int(max_abs > 0),
        "shape_changed": shape_changed,
    }


class RepeatTraceTests(unittest.TestCase):
    def test_hooks_layers_and_their_parts_but_not_the_leaves(self):
        base = "language_model.model.layers.3"
        for name in (
            "language_model.model.embed_tokens",
            base,
            base + ".self_attn",
            base + ".self_attn.indexer",
            base + ".mlp",
            base + ".mlp.gate",
            base + ".mlp.experts",
            base + ".mlp.shared_experts",
            "language_model.model.norm",
            "language_model.lm_head",
        ):
            self.assertTrue(HOOKED.search(name), name)
        for name in (
            base + ".self_attn.o_proj",
            base + ".mlp.shared_experts.down_proj",
            base + ".input_layernorm",
            "language_model.model.layers",
            "",
        ):
            self.assertFalse(HOOKED.search(name), name)

    def test_the_first_difference_in_execution_order_is_named(self):
        rows = [
            row(1, "layers.0", 0.0),
            row(2, "layers.1.mlp.experts", 0.25),
            row(3, "layers.1.mlp", 0.5),
            row(4, "layers.1.mlp.experts", 0.125),
        ]
        result = summarize(rows)
        self.assertEqual(result["calls_compared"], 4)
        self.assertEqual(result["calls_differing"], 3)
        self.assertEqual(result["first_difference"]["module"], "layers.1.mlp.experts")
        self.assertEqual(
            result["modules_differing"]["layers.1.mlp.experts"],
            {"calls": 2, "max_abs": 0.25},
        )

    def test_identical_passes_and_changed_shapes(self):
        self.assertIsNone(summarize([row(1, "layers.0", 0.0)])["first_difference"])
        changed = summarize([row(1, "layers.0", 0.0, shape_changed=True)])
        self.assertEqual(changed["first_difference"]["module"], "layers.0")


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
class AlignWrapperTests(unittest.TestCase):
    """The diagnostics order slots exactly as the served patch does.

    The buffers are the worst-case ones of the served model: 18 slots and 6 expert
    blocks for a block of 4, so neither length divides into the other. A block size
    guessed from their ratio (3) gives slots to the wrong expert; with lengths that
    divide, that mistake would stay green.
    """

    def align(self):
        import torch

        sorted_ids = torch.tensor(
            [9, 3, 12, 4, 8, 1, 6, 5, 2, 12, 12, 0, 7, 7, 7, 7, 7, 7]
        )
        expert_ids = torch.tensor([2, 5, 5, 0, 0, 0])
        return sorted_ids, expert_ids, torch.tensor(12)

    def expected(self):
        from glm53_setup.runtime.moe_token_order import canonical_expert_order

        sorted_ids, expert_ids, padded = self.align()
        return canonical_expert_order(sorted_ids, expert_ids, padded, 4).tolist()

    def test_order_only_takes_the_block_size_the_kernel_was_given(self):
        from glm53_setup.validation.run_repeat_trace import ordered_align

        for call in (
            lambda f: f("topk_ids", 4, 288),
            lambda f: f("topk_ids", block_size=4, num_experts=288),
        ):
            result = call(ordered_align(lambda *a, **k: self.align()))
            self.assertEqual(result[0].tolist(), self.expected())

    def test_watching_canonicalises_with_the_same_function(self):
        import torch

        from glm53_setup.validation.run_repeat_trace import watched_align

        log = []
        result = watched_align(lambda *a, **k: self.align(), log, True)("topk_ids", 4)
        self.assertEqual(result[0].tolist(), self.expected())

        def shuffled(*args, **kwargs):  # the same sets, another order inside experts
            sorted_ids, expert_ids, padded = self.align()
            sorted_ids[:4] = sorted_ids[:4].flip(0)
            sorted_ids[4:12] = sorted_ids[4:12].flip(0)
            return sorted_ids, expert_ids, padded

        watched_align(shuffled, log, False)("topk_ids", 4)
        self.assertNotEqual(log[0]["sorted_token_ids"], log[1]["sorted_token_ids"])
        self.assertEqual(log[0]["as_a_set_per_expert"], log[1]["as_a_set_per_expert"])
        self.assertEqual(log[1]["num_tokens_post_padded"], 12)
        self.assertTrue(torch.equal(result[1], self.align()[1]))


if __name__ == "__main__":
    unittest.main()
