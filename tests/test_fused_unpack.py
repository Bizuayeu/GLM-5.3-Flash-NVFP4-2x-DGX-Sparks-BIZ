import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


class FusedUnpackSourceTests(unittest.TestCase):
    def test_element_count_is_a_run_time_argument(self):
        # A tl.constexpr element count compiles and keeps one kernel per distinct
        # size: a caller whose row count changes every call (the FA2 compaction)
        # then grows the worker's heap and the on-disk Triton cache without bound.
        source = (
            Path(__file__).resolve().parents[1] / "glm53_setup/runtime/fused_unpack.py"
        )
        kernel = next(
            node
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name == "_unpack"
        )
        constexpr = [
            arg.arg
            for arg in kernel.args.args
            if arg.annotation is not None and "constexpr" in ast.unparse(arg.annotation)
        ]
        self.assertEqual(constexpr, ["BLOCK"])


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

    def test_changing_row_counts_do_not_add_compiled_kernels(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA device required")
        from glm53_setup.runtime import fused_unpack

        def compiled():
            caches = getattr(fused_unpack._unpack, "device_caches", None)
            if caches is not None:
                return sum(len(entry[0]) for entry in caches.values())
            return sum(len(entry) for entry in fused_unpack._unpack.cache.values())

        packed = torch.zeros((4096, 656), dtype=torch.uint8, device="cuda")
        for rows in (
            16,
            32,
            1,
            3,
        ):  # the integer specialisations: multiple of 16, one, neither
            fused_unpack.unpack_latent_cuda(packed[:rows])
        before = compiled()
        for rows in range(100, 1100, 7):
            fused_unpack.unpack_latent_cuda(packed[:rows])
        self.assertEqual(compiled(), before)

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
