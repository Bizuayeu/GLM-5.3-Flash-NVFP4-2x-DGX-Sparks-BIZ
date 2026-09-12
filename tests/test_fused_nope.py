import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
    "CUDA/Triton required",
)
class FusedNopeTests(unittest.TestCase):
    def test_fp64_reference_and_tail_candidates(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from glm53_setup.runtime.fused_nope import fused_nope_attention
        from glm53_setup.runtime.reference_attention import unpack_latent

        torch.manual_seed(71)
        cache = torch.zeros((2112, 656), dtype=torch.uint8, device="cuda")
        cache[:, :512] = (
            torch.randn((2112, 512), device="cuda")
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        cache[:, 512:528] = torch.rand((2112, 4), device="cuda").view(torch.uint8)
        decoded = unpack_latent(cache).double().cpu()
        for width in (0, 17, 65, 2048, 2051, 2176):
            query = torch.randn((3, 2, 576), dtype=torch.bfloat16, device="cuda")[
                ..., :512
            ]
            indices = torch.full((3, width), -1, dtype=torch.int32, device="cuda")
            count = min(width, len(cache))
            indices[0, :count] = torch.arange(count, dtype=torch.int32, device="cuda")
            if width:
                indices[1, -1] = 2050
            expected = torch.zeros((3, 2, 512), dtype=torch.bfloat16)
            for row in (0, 1):
                ids = indices[row].cpu()
                ids = ids[ids >= 0]
                if ids.numel():
                    kv = decoded[ids]
                    expected[row] = (
                        (query[row].double().cpu() @ kv.T * 512**-0.5).softmax(-1) @ kv
                    ).bfloat16()
            actual = fused_nope_attention(query, cache, indices, 512**-0.5).cpu()
            tolerance = (
                2
                * torch.finfo(torch.bfloat16).eps
                * max(1, expected.abs().max().item())
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
            self.assertEqual(torch.count_nonzero(actual[2]).item(), 0)
            if width:
                torch.testing.assert_close(
                    actual[1], decoded[2050].bfloat16().expand(2, -1), rtol=0, atol=0
                )
        invalid = torch.tensor([[-2], [2112], [0]], device="cuda", dtype=torch.int32)
        with self.assertRaises(ValueError):
            fused_nope_attention(query, cache, invalid, 1.0)
