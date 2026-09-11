import copy
import fnmatch
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.prepare_mtp_view import inspect_mtp, metadata_configs


class MtpViewTests(unittest.TestCase):
    def test_impossible_header_length_is_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = "model.language_model.layers.45.enorm.weight"
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {key: "shard.safetensors"}})
            )
            (root / "shard.safetensors").write_bytes(struct.pack("<Q", 2**63))
            with self.assertRaisesRegex(ValueError, "header"):
                inspect_mtp(root, 45)

    def test_only_declared_mtp_layer_is_added_and_inputs_are_unchanged(self):
        config = {
            "model_type": "glm5_next",
            "text_config": {"num_hidden_layers": 45, "num_nextn_predict_layers": 1},
            "quantization_config": {"quant_algo": "NVFP4", "ignore": ["lm_head"]},
        }
        legacy = {"quantization": {"exclude_modules": ["lm_head"]}}
        before = copy.deepcopy((config, legacy))
        updated, aux, layer = metadata_configs(config, legacy, "test-revision")
        self.assertEqual((config, legacy), before)
        self.assertEqual(layer, 45)
        pattern = updated["quantization_config"]["ignore"][-1]
        self.assertEqual(aux["quantization"]["exclude_modules"][-1], pattern)
        self.assertTrue(
            fnmatch.fnmatch("model.layers.45.mtp_block.mlp.experts", pattern)
        )
        for i in range(45):
            self.assertFalse(
                fnmatch.fnmatch(f"language_model.model.layers.{i}.mlp.experts", pattern)
            )
        self.assertFalse(fnmatch.fnmatch("model.layers.450.mlp.experts", pattern))

    def test_quantized_mtp_cannot_be_silently_treated_as_bf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = "model.language_model.layers.45.mlp.experts.0.up_proj.weight"
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {key: "shard.safetensors"}})
            )
            header = json.dumps(
                {key: {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
            ).encode()
            (root / "shard.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + bytes(4)
            )
            with self.assertRaisesRegex(ValueError, "not wholly BF16"):
                inspect_mtp(root, 45)
