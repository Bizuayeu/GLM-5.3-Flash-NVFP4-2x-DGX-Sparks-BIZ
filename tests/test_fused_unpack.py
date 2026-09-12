import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
    "CUDA/Triton environment required",
)
class FusedUnpackTests(unittest.TestCase):
    def test_attention_with_padding_and_empty_query_matches_reference(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA device required")
        from glm53_setup.runtime.reference_attention import sparse_nope_reference

        torch.manual_seed(42)
        packed = torch.zeros((32, 656), dtype=torch.uint8, device="cuda")
        packed[:, :512] = (
            torch.randn((32, 512), device="cuda")
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        packed[:, 512:528] = torch.rand((32, 4), device="cuda").view(torch.uint8)
        query = torch.randn((9, 2, 512), device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(32, device="cuda").repeat(9, 1)
        indices[0] = -1
        indices[1, 3:] = -1
        with patch.dict("os.environ", {"GLM53_FUSED_UNPACK": "0"}):
            expected = sparse_nope_reference(query, packed, indices, 512**-0.5)
        with patch.dict("os.environ", {"GLM53_FUSED_UNPACK": "1"}):
            actual = sparse_nope_reference(query, packed, indices, 512**-0.5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)

    def test_all_fp8_codes_and_group_scales_match_reference(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA device required")
        from glm53_setup.runtime.fused_unpack import unpack_latent_cuda

        for rows in (1, 3, 97):
            packed = torch.zeros((rows, 656), dtype=torch.uint8, device="cuda")
            raw = torch.arange(256, device="cuda").to(torch.uint8).repeat(2)
            packed[:, :512] = raw
            scales = torch.tensor(
                [1.0, 0.5, 1.25, 0.0001], device="cuda", dtype=torch.float32
            ).repeat(rows, 1)
            packed[:, 512:528] = scales.view(torch.uint8)
            values = packed[:, :512].contiguous().view(torch.float8_e4m3fn).float()
            expected = (values.reshape(rows, 4, 128) * scales.unsqueeze(-1)).flatten(-2)
            actual = unpack_latent_cuda(packed)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            self.assertTrue(
                torch.equal(
                    torch.signbit(actual[expected == 0]),
                    torch.signbit(expected[expected == 0]),
                )
            )
        self.assertEqual(
            unpack_latent_cuda(
                torch.empty((0, 656), dtype=torch.uint8, device="cuda")
            ).shape,
            (0, 512),
        )
