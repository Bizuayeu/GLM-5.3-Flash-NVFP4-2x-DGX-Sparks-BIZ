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


if __name__ == "__main__":
    unittest.main()
