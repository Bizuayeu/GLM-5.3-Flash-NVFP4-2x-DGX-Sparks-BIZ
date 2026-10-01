import importlib.util
import types
import unittest

from glm53_setup.runtime import tp_padding


class FakeTensor:
    """Records the narrow a loader asks for; stands in for torch on CPU CI."""

    def __init__(self, *shape):
        self.shape = shape
        self.calls = []

    def size(self, dim):
        return self.shape[dim]

    def narrow(self, dim, start, length):
        self.calls.append((dim, start, length))
        return ("narrow", dim, start, length)


class MultipleTests(unittest.TestCase):
    def test_unset_is_one_and_a_value_must_be_a_positive_integer(self):
        self.assertEqual(tp_padding.pad_multiple({}), 1)
        self.assertEqual(tp_padding.pad_multiple({tp_padding.ENV: "1"}), 1)
        self.assertEqual(tp_padding.pad_multiple({tp_padding.ENV: "3"}), 3)
        for value in ("", "0", "-3", "1.5", "three"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tp_padding.pad_multiple({tp_padding.ENV: value})

    def test_only_sizes_that_do_not_divide_the_checkpoint_need_a_multiple(self):
        # 64 heads, a 2048 MoE width in 64-wide shares, a 154880 vocabulary.
        self.assertEqual(
            {tp: tp_padding.required_multiple(tp) for tp in (1, 2, 3, 4, 5, 6, 8)},
            {1: 1, 2: 1, 3: 3, 4: 1, 5: 5, 6: 6, 8: 1},
        )


class GeometryTests(unittest.TestCase):
    def test_the_padded_sizes_are_identity_at_one_and_the_plan_values_at_three(self):
        self.assertEqual(tp_padding.padded_heads(64, 1), 64)
        self.assertEqual(tp_padding.padded_heads(64, 3), 66)
        self.assertEqual(tp_padding.padded_moe_width(2048, 1), 2048)
        # 2112 = 3 x 704 and 704 = 11 x 64 (Stage 0 P1).
        self.assertEqual(tp_padding.padded_moe_width(2048, 3), 2112)
        self.assertEqual(tp_padding.padded_moe_width(1000, 1), 1000)
        self.assertEqual(tp_padding.vocab_padding_size(64, 1), 64)
        self.assertEqual(tp_padding.vocab_padding_size(64, 3), 192)

    def test_padding_is_idempotent(self):
        self.assertEqual(tp_padding.padded_heads(66, 3), 66)
        self.assertEqual(tp_padding.padded_moe_width(2112, 3), 2112)


def text_config(**overrides):
    fields = dict(
        num_attention_heads=64,
        num_key_value_heads=64,
        linear_num_heads=64,
        moe_intermediate_size=2048,
        head_dim=0,
        vocab_size=154880,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


class TextConfigTests(unittest.TestCase):
    def test_multiple_three_pads_heads_and_moe_width_and_leaves_the_rest(self):
        config = text_config()
        linear = {"num_heads": 64, "head_dim": 128}
        kwargs = {"linear_attn_config": linear}
        tp_padding.pad_text_config(config, kwargs, 3)
        self.assertEqual(
            (
                config.num_attention_heads,
                config.num_key_value_heads,
                config.linear_num_heads,
                config.moe_intermediate_size,
            ),
            (66, 66, 66, 2112),
        )
        self.assertEqual((config.head_dim, config.vocab_size), (0, 154880))
        # The dict wins over linear_num_heads on a later to_dict/from_dict round trip.
        self.assertEqual(
            kwargs["linear_attn_config"], {"num_heads": 66, "head_dim": 128}
        )
        self.assertEqual(linear["num_heads"], 64, "the caller's dict is not mutated")

    def test_multiple_one_changes_nothing(self):
        config, kwargs = text_config(), {"linear_attn_config": {"num_heads": 64}}
        tp_padding.pad_text_config(config, kwargs, 1)
        self.assertEqual(vars(config), vars(text_config()))
        self.assertEqual(kwargs, {"linear_attn_config": {"num_heads": 64}})

    def test_without_a_linear_attention_dict_only_the_attributes_change(self):
        config, kwargs = text_config(), {}
        tp_padding.pad_text_config(config, kwargs, 3)
        self.assertEqual(kwargs, {})
        self.assertEqual(config.linear_num_heads, 66)


class PadThenNarrowTests(unittest.TestCase):
    def test_multiple_one_takes_the_pinned_narrow_even_past_the_end(self):
        # Unset, the pinned loaders' narrow runs unchanged, errors included.
        tensor = FakeTensor(8192, 4096)
        result = tp_padding.pad_then_narrow(tensor, 0, 5632, 2816, 1)
        self.assertEqual(result, ("narrow", 0, 5632, 2816))
        self.assertEqual(tensor.calls, [(0, 5632, 2816)])

    def test_a_shard_inside_the_tensor_is_the_pinned_narrow(self):
        tensor = FakeTensor(8192, 4096)
        result = tp_padding.pad_then_narrow(tensor, 0, 2816, 2816, 3)
        self.assertEqual(result, ("narrow", 0, 2816, 2816))

    def test_a_shard_of_padding_only_is_refused(self):
        with self.assertRaises(ValueError):
            tp_padding.pad_then_narrow(FakeTensor(64), 0, 66, 22, 3)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
class TorchPadThenNarrowTests(unittest.TestCase):
    def test_the_last_shard_is_zero_extended_and_real_rows_are_kept(self):
        import torch

        # 64 KDA heads of 2 rows padded to 66 heads, 22 per rank.
        weight = torch.arange(128 * 3, dtype=torch.float32).reshape(128, 3) + 1
        shards = [
            tp_padding.pad_then_narrow(weight, 0, rank * 44, 44, 3) for rank in range(3)
        ]
        joined = torch.cat(shards)
        self.assertTrue(torch.equal(joined[:128], weight))
        self.assertTrue(torch.equal(joined[128:], torch.zeros(4, 3)))

    def test_packed_fp4_bytes_and_fp8_scales_pad_with_zeros_along_k(self):
        import torch

        # w2 of one expert: H x (2048 / 2) packed bytes and H x (2048 / 16) scales.
        packed = torch.randint(1, 256, (8, 1024), dtype=torch.uint8)
        scales = torch.rand(8, 128).to(torch.float8_e4m3fn)
        last = tp_padding.pad_then_narrow(packed, 1, 2 * 352, 352, 3)
        self.assertTrue(torch.equal(last[:, :320], packed[:, 704:]))
        self.assertEqual(int(last[:, 320:].count_nonzero()), 0)
        last = tp_padding.pad_then_narrow(scales, 1, 2 * 44, 44, 3)
        self.assertTrue(torch.equal(last[:, :40].float(), scales[:, 88:].float()))
        self.assertEqual(int(last[:, 40:].float().count_nonzero()), 0)

    def test_a_one_dimensional_parameter_pads_along_its_only_axis(self):
        import torch

        # A_log arrives 1-D and is viewed [1, 1, heads, 1] before the narrow.
        a_log = (torch.arange(64, dtype=torch.float32) + 1).view(1, 1, -1, 1)
        last = tp_padding.pad_then_narrow(a_log, 2, 44, 22, 3)
        self.assertEqual(tuple(last.shape), (1, 1, 22, 1))
        self.assertTrue(torch.equal(last[0, 0, :20, 0], a_log[0, 0, 44:, 0]))
        self.assertTrue(torch.equal(last[0, 0, 20:, 0], torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
