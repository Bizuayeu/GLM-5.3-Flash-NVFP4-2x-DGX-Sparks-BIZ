import importlib.util
import math
import unittest

from glm53_setup.validation.quant_error import (
    dequant_nvfp4,
    e4m3fn_table,
    module_kind,
    summarize,
    tensor_error,
)


class E4m3Tests(unittest.TestCase):
    def test_table_matches_the_format_corners(self):
        table = e4m3fn_table()
        self.assertEqual(len(table), 256)
        self.assertEqual(table[0x00], 0.0)
        self.assertEqual(table[0x01], 2.0**-9)
        self.assertEqual(table[0x38], 1.0)
        self.assertEqual(table[0xB8], -1.0)
        self.assertEqual(table[0x7E], 448.0)
        self.assertTrue(math.isnan(table[0x7F]))
        self.assertEqual(sum(math.isinf(v) for v in table), 0)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy required")
class QuantErrorTests(unittest.TestCase):
    def test_low_nibble_is_the_even_column_and_bit_three_is_the_sign(self):
        import numpy as np

        packed = np.zeros((1, 16), dtype=np.uint8)
        packed[0, 0] = 0x21  # columns 0, 1 = codes 1, 2 = 0.5, 1.0
        packed[0, 8] = 0xF9  # columns 16, 17 = codes 9, 15 = -0.5, -6.0
        scales = np.array([[0x38, 0x40]], dtype=np.uint8)  # 1.0, 2.0
        result = dequant_nvfp4(packed, scales, 0.5)
        self.assertEqual(result.shape, (1, 32))
        self.assertEqual(result[0, :3].tolist(), [0.25, 0.5, 0.0])
        self.assertEqual(result[0, 16:18].tolist(), [-0.5, -6.0])

    def test_rejects_scales_of_another_shape(self):
        import numpy as np

        with self.assertRaises(ValueError):
            dequant_nvfp4(
                np.zeros((2, 16), dtype=np.uint8), np.zeros((2, 1), dtype=np.uint8), 1.0
            )

    def test_error_names_the_worst_row(self):
        import numpy as np

        original = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        restored = original.copy()
        restored[2, 1] = 0.5
        row = tensor_error(original, restored)
        self.assertEqual(row["worst_row"], 2)
        self.assertAlmostEqual(row["worst_row_relative"], 0.5)
        self.assertAlmostEqual(row["relative_frobenius"], 0.5 / math.sqrt(26))
        self.assertAlmostEqual(row["max_abs_over_amax"], 0.125)
        with self.assertRaises(ValueError):
            tensor_error(original, restored[:2])
        restored[0, 0] = np.nan
        with self.assertRaises(ValueError):
            tensor_error(original, restored)

    def test_kinds_drop_the_layer_number(self):
        name = "model.language_model.layers.3.self_attn.indexer.wq_b.weight"
        self.assertEqual(module_kind(name), "self_attn.indexer.wq_b")
        self.assertEqual(module_kind("lm_head.weight"), "lm_head")
        rows = [
            {"kind": "a", "relative_frobenius": value} for value in (0.1, 0.3, 0.2)
        ] + [{"kind": "b", "relative_frobenius": 0.5}]
        self.assertEqual(
            summarize(rows)["a"],
            {
                "tensors": 3,
                "max_relative_frobenius": 0.3,
                "median_relative_frobenius": 0.2,
            },
        )


if __name__ == "__main__":
    unittest.main()
