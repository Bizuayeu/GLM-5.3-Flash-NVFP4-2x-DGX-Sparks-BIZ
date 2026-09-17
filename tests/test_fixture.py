import copy
import unittest

from glm53_setup.validation.make_fixture import (
    fixture_config,
    fixture_name,
    keep_tensor,
)


class FixtureTests(unittest.TestCase):
    def test_eight_layer_fixture_keeps_one_sparse_layer_per_pp_stage(self):
        pattern = ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
        source = {
            "model_type": "glm5_next",
            "text_config": {
                "layer_types": pattern * 2 + ["linear_attention"],
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 6,
                "indexer_types": ["full"] * 9,
                "linear_attn_config": {
                    "kda_layers": [0, 1, 2, 4, 5, 6, 8],
                    "full_attn_layers": [3, 7],
                },
            },
        }
        before = copy.deepcopy(source)
        result = fixture_config(source, layers=8)
        self.assertEqual(source, before)
        self.assertEqual(result["_fixture_source"]["layers"], list(range(8)))
        self.assertEqual(result["text_config"]["num_hidden_layers"], 8)
        self.assertEqual(result["text_config"]["layer_types"], pattern * 2)
        self.assertEqual(
            result["text_config"]["linear_attn_config"]["kda_layers"],
            [0, 1, 2, 4, 5, 6],
        )
        self.assertTrue(keep_tensor("model.language_model.layers.7.x", layers=8))
        self.assertFalse(keep_tensor("model.language_model.layers.8.x", layers=8))
        for layers in (True, 0, 5, 45):
            with self.assertRaises(ValueError):
                fixture_config(source, layers=layers)

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


class MtpFixtureTests(unittest.TestCase):
    SOURCE = {
        "model_type": "glm5_next",
        "quantization_config": {"quant_algo": "NVFP4", "ignore": ["lm_head"]},
        "text_config": {
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 1,
            "layer_types": (["linear_attention"] * 3 + ["deepseek_sparse_attention"])
            * 11
            + ["linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 43,
            "indexer_types": ["full"] * 46,
            "linear_attn_config": {"kda_layers": [0, 1, 2, 4], "full_attn_layers": [3]},
        },
    }

    def test_draft_layer_follows_the_kept_layers_and_stays_unquantized(self):
        before = copy.deepcopy(self.SOURCE)
        result = fixture_config(self.SOURCE, with_mtp=True)
        text = result["text_config"]
        self.assertEqual(self.SOURCE, before)
        self.assertEqual(text["num_hidden_layers"], 4)
        self.assertEqual(text["num_nextn_predict_layers"], 1)
        self.assertEqual(text["layer_types"][4], "deepseek_sparse_attention")
        self.assertEqual(text["mlp_layer_types"], ["dense"] * 3 + ["sparse"] * 2)
        self.assertEqual(len(text["indexer_types"]), 5)
        self.assertIn("*.layers.4.*", result["quantization_config"]["ignore"])
        self.assertEqual(
            result["_fixture_source"]["mtp_layer"], {"source": 45, "as": 4}
        )

    def test_draft_tensors_are_renamed_and_nothing_else_is_added(self):
        name = "model.language_model.layers.45.enorm.weight"
        self.assertEqual(
            fixture_name(name, mtp_source=45),
            "model.language_model.layers.4.enorm.weight",
        )
        self.assertIsNone(fixture_name(name))
        self.assertIsNone(
            fixture_name("model.language_model.layers.44.x", mtp_source=45)
        )
        self.assertEqual(
            fixture_name("model.language_model.layers.3.x", mtp_source=45),
            "model.language_model.layers.3.x",
        )
        self.assertEqual(fixture_name("lm_head.weight"), "lm_head.weight")

    def test_without_the_option_the_fixture_is_unchanged(self):
        text = fixture_config(self.SOURCE)["text_config"]
        self.assertEqual(text["num_nextn_predict_layers"], 0)
        self.assertEqual(len(text["layer_types"]), 4)
