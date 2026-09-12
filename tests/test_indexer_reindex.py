import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
    "CUDA/Triton required",
)
class ReindexGpuTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("vllm"), "Pinned vLLM required")
    def test_shared_noncontiguous_pool_matches_native_with_empty_prefix(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

        from glm53_setup.runtime.indexer_shared_pool import shared_pool_scores

        torch.manual_seed(19)
        q = torch.randn((32, 32, 128), device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn((256, 128), device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(256, device="cuda") + 0.1
        weights = torch.rand((32, 32), device="cuda")
        pool = torch.arange(1, 256, 2, dtype=torch.int32, device="cuda")
        limits = (torch.arange(32, dtype=torch.int32, device="cuda") * 8).contiguous()
        native = fp8_fp4_mqa_logits(
            (q, None),
            (k, scales),
            weights,
            torch.zeros_like(limits),
            limits,
            clean_logits=True,
        )
        expected = native.index_select(1, pool.long())
        actual = shared_pool_scores(q, k, scales, weights, pool, limits)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
        self.assertTrue(torch.isneginf(actual[0]).all())
        with self.assertRaises(ValueError):
            shared_pool_scores(q, k, scales, weights, pool.flip(0), limits)

    def test_candidate_scores_match_fp64_oracle_with_masking(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from glm53_setup.runtime.indexer_reindex import candidate_scores_cuda

        torch.manual_seed(73)
        for count in (1, 17, 128):
            q = torch.randn((3, 32, 128), device="cuda").to(torch.float8_e4m3fn)
            k = torch.randn((257, 128), device="cuda").to(torch.float8_e4m3fn)
            scale = torch.rand(257, device="cuda") + 0.1
            weights = torch.randn((3, 32), device="cuda")
            ids = torch.randint(0, 257, (3, count), device="cuda")
            ids[0, 0] = -1
            limits = torch.tensor([0, 128, 257], device="cuda")
            actual = candidate_scores_cuda(q, k, scale, weights, ids, limits)
            dense = torch.einsum(
                "qhd,pd->qph", q.double(), k.double() * scale.double()[:, None]
            ).relu()
            dense = (dense * weights.double()[:, None, :]).sum(-1)
            expected = dense.gather(1, ids.clamp_min(0))
            expected.masked_fill_((ids < 0) | (ids >= limits[:, None]), float("-inf"))
            torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-4)
            self.assertTrue(torch.isneginf(actual[0]).all())


if __name__ == "__main__":
    unittest.main()
