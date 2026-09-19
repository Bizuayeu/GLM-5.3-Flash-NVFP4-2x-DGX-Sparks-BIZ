import importlib.util
import os
import unittest
from unittest.mock import patch

from glm53_setup.runtime import patch_indexer_topk
from glm53_setup.runtime.stable_topk import stable_topk_enabled

HAS_CUDA = False
if importlib.util.find_spec("torch"):
    import torch

    HAS_CUDA = torch.cuda.is_available()

SOURCE = """import torch

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def indexer(logits, chunk, topk_dst, num_rows, select_k, seq_lens, topk_workspace, max_seq_len):
    if prefill:
        if xpu:
            xpu_ops.top_k_per_row_prefill(logits)
        else:
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_dst,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                select_k,
            )
    torch.ops._C.persistent_topk(
        logits,
        seq_lens,
        topk_dst,
        topk_workspace,
        select_k,
        max_seq_len,
    )
"""


class SwitchTests(unittest.TestCase):
    def test_on_by_default_and_strict_about_values(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLM53_STABLE_INDEXER_TOPK", None)
            self.assertTrue(stable_topk_enabled())
        for value, expected in (("1", True), ("0", False)):
            with patch.dict(os.environ, {"GLM53_STABLE_INDEXER_TOPK": value}):
                self.assertIs(stable_topk_enabled(), expected)
        with patch.dict(os.environ, {"GLM53_STABLE_INDEXER_TOPK": "yes"}):
            with self.assertRaises(ValueError):
                stable_topk_enabled()


class PatchTests(unittest.TestCase):
    def test_routes_both_kernels_through_the_switch_and_still_compiles(self):
        patched = patch_indexer_topk.patch_text(SOURCE)
        self.assertEqual(patched.count("glm53_decode_topk("), 1)
        self.assertEqual(patched.count("glm53_prefill_topk("), 1)
        self.assertNotIn("torch.ops._C.persistent_topk(", patched)
        self.assertNotIn("torch.ops._C.top_k_per_row_prefill(", patched)
        self.assertIn("xpu_ops.top_k_per_row_prefill(", patched)  # left alone
        self.assertTrue(patched.startswith("# Modified by GLM setup"))

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_indexer_topk.patch_text(
                SOURCE.replace("persistent_topk", "radix_topk")
            )
        with self.assertRaises(ValueError):
            patch_indexer_topk.patch_text(patch_indexer_topk.patch_text(SOURCE))


@unittest.skipUnless(HAS_CUDA, "needs torch with a CUDA device")
class DecodeTests(unittest.TestCase):
    def select(self, values, lengths, k=512, width=4096):
        from glm53_setup.runtime.stable_topk import stable_topk

        rows = len(lengths)
        logits = torch.full((rows, width), float("nan"), device="cuda")
        for row, length in enumerate(lengths):
            logits[row, :length] = values[row, :length]
        output = torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
        stable_topk(
            logits,
            torch.tensor([lengths], dtype=torch.int32, device="cuda"),
            output,
            k,
            max(lengths) * 4,
        )
        return output

    def test_picks_the_top_k_of_each_rows_valid_prefix(self):
        values = torch.randn((3, 4096), device="cuda")
        output = self.select(values, [600, 601, 700])
        for row, length in enumerate((600, 601, 700)):
            expected = set(torch.topk(values[row, :length], 512).indices.tolist())
            self.assertEqual(set(output[row].tolist()), expected)

    def test_a_tie_across_the_boundary_goes_to_the_lower_index_every_time(self):
        values = torch.arange(4096, 0, -1, device="cuda").float().repeat(2, 1)
        values[:, 508:520] = values[0, 508]  # twelve tied pools straddle rank 512
        first = self.select(values, [604, 604])
        self.assertEqual(sorted(first[0].tolist()), list(range(512)))
        for _ in range(50):
            self.assertTrue(torch.equal(self.select(values, [604, 604]), first))

    def test_a_row_shorter_than_k_lists_its_prefix_and_pads_with_minus_one(self):
        values = torch.randn((1, 4096), device="cuda")
        output = self.select(values, [100])
        self.assertEqual(sorted(output[0, :100].tolist()), list(range(100)))
        self.assertTrue(bool((output[0, 100:] == -1).all()))


@unittest.skipUnless(HAS_CUDA, "needs torch with a CUDA device")
class PrefillTests(unittest.TestCase):
    def run_prefill(self, logits, starts, ends, k=512):
        import vllm._C_stable_libtorch  # noqa: F401

        from glm53_setup.runtime.stable_topk import prefill_topk

        rows = logits.shape[0]
        output = torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
        prefill_topk(
            logits, starts, ends, output, rows, logits.stride(0), logits.stride(1), k
        )
        return output

    def test_rows_without_a_tie_keep_what_the_kernel_chose(self):
        import vllm._C_stable_libtorch  # noqa: F401

        rows, width, k = 300, 900, 512
        logits = torch.randn((rows, width), device="cuda")
        starts = torch.arange(rows, dtype=torch.int32, device="cuda") % 3
        ends = torch.full((rows,), 600, dtype=torch.int32, device="cuda") + starts * 100
        ends[0] = 40  # shorter than k
        kernel = torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
        torch.ops._C.top_k_per_row_prefill(
            logits, starts, ends, kernel, rows, logits.stride(0), logits.stride(1), k
        )
        ours = self.run_prefill(logits, starts, ends)
        self.assertTrue(
            torch.equal(kernel.sort(dim=-1).values, ours.sort(dim=-1).values)
        )

    def test_a_tied_row_goes_to_the_lower_index_every_time(self):
        rows = 300  # more than one slice of the tie check
        values = torch.arange(900, 0, -1, device="cuda").float().repeat(rows, 1)
        values[7, 508:520] = values[7, 508]
        values[299, 500:530] = values[299, 500]
        starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        ends = torch.full((rows,), 604, dtype=torch.int32, device="cuda")
        for _ in range(50):
            output = self.run_prefill(values, starts, ends)
            for row in (7, 299, 0):
                self.assertEqual(sorted(output[row].tolist()), list(range(512)))

    def test_switched_off_it_is_the_kernel_alone(self):
        values = torch.randn((8, 900), device="cuda")
        starts = torch.zeros(8, dtype=torch.int32, device="cuda")
        ends = torch.full((8,), 604, dtype=torch.int32, device="cuda")
        with patch.dict(os.environ, {"GLM53_STABLE_INDEXER_TOPK": "0"}):
            with patch(
                "glm53_setup.runtime.stable_topk.repair_boundary_ties"
            ) as repair:
                self.run_prefill(values, starts, ends)
        repair.assert_not_called()
