import importlib.util
import unittest

HAS_CUDA = False
if importlib.util.find_spec("torch"):
    import torch

    HAS_CUDA = torch.cuda.is_available()


@unittest.skipUnless(HAS_CUDA, "needs torch with a CUDA device")
class StableTopkTests(unittest.TestCase):
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
