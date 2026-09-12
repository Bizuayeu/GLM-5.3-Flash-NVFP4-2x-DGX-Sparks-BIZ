import importlib.util
import unittest

from glm53_setup.runtime.reference_attention import sparse_nope_reference


class QueryChunkContractTests(unittest.TestCase):
    def test_invalid_chunk_is_rejected_before_tensor_access(self):
        for value in (0, -1, True, 16, 128, "32"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "query_chunk"):
                    sparse_nope_reference(None, None, None, 1.0, query_chunk=value)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required")
class QueryChunkMathTests(unittest.TestCase):
    def test_batched_queries_preserve_masks_and_tail(self):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        cache = torch.zeros((64, 656), dtype=torch.uint8, device=device)
        cache[:, :512] = (
            torch.randn((64, 512), device=device)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        cache[:, 512:528] = torch.rand((64, 4), device=device).view(torch.uint8)
        query = torch.randn((65, 2, 512), device=device, dtype=torch.bfloat16)
        indices = torch.arange(64, device=device).repeat(65, 1)
        indices[0] = -1
        indices[1, :-1] = -1
        indices[2, ::2] = -1
        from glm53_setup.runtime.reference_attention import unpack_latent

        decoded = unpack_latent(cache).double().cpu()
        q_cpu = query.double().cpu()
        expected = torch.zeros_like(query, device="cpu")
        for row in range(65):
            selected = indices[row].cpu()
            selected = selected[selected >= 0]
            if selected.numel():
                kv = decoded[selected]
                expected[row] = (
                    (q_cpu[row] @ kv.T * 512**-0.5).softmax(-1) @ kv
                ).bfloat16()
        tolerance = (
            2 * torch.finfo(torch.bfloat16).eps * max(1, expected.abs().max().item())
        )
        for chunk in (32, 64):
            actual = sparse_nope_reference(
                query, cache, indices, 512**-0.5, query_chunk=chunk
            )
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=tolerance)
            self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)
            torch.testing.assert_close(
                actual[1].cpu(), decoded[-1].bfloat16().expand(2, -1), rtol=0, atol=0
            )
