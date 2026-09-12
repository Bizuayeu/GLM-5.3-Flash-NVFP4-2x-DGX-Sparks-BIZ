import copy
import unittest

from glm53_setup.validation.make_fixture import fixture_config, keep_tensor


class FixtureTests(unittest.TestCase):
    def test_selects_only_first_four_layers_and_shared_language_weights(self):
        for name in [
            "model.language_model.layers.0.x",
            "model.language_model.layers.3.mlp.experts.287.down_proj.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.hc_head_fn",
            "lm_head.weight",
        ]:
            self.assertTrue(keep_tensor(name), name)
        for name in [
            "model.language_model.layers.4.x",
            "model.language_model.layers.45.enorm.weight",
            "model.visual.blocks.0.x",
        ]:
            self.assertFalse(keep_tensor(name), name)
        with self.assertRaises(ValueError):
            keep_tensor("model.language_model.layers.bad.weight")

    def test_keeps_dimensions_and_original_config_unchanged(self):
        source = {
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 45,
                "num_nextn_predict_layers": 1,
                "hidden_size": 4096,
                "n_routed_experts": 288,
                "layer_types": ["linear_attention"] * 3
                + ["deepseek_sparse_attention"]
                + ["linear_attention"] * 41,
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
                "indexer_types": ["full"] * 45,
                "linear_attn_config": {
                    "kda_layers": [0, 1, 2, 4],
                    "full_attn_layers": [3, 7],
                },
            },
        }
        before = copy.deepcopy(source)
        result = fixture_config(source)
        text = result["text_config"]
        self.assertEqual(source, before)
        self.assertEqual(text["num_hidden_layers"], 4)
        self.assertEqual(text["num_nextn_predict_layers"], 0)
        self.assertEqual(text["hidden_size"], 4096)
        self.assertEqual(text["n_routed_experts"], 288)
        self.assertEqual(text["linear_attn_config"]["kda_layers"], [0, 1, 2])
        self.assertEqual(text["linear_attn_config"]["full_attn_layers"], [3])
        self.assertTrue(result["_test_fixture_only"])
        source["text_config"]["layer_types"][3] = "linear_attention"
        with self.assertRaises(ValueError):
            fixture_config(source)


if __name__ == "__main__":
    unittest.main()
