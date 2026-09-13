import importlib.util
import unittest
from unittest.mock import Mock

from glm53_setup.runtime.lpa_query import ReferenceQueryMask


class QueryScopeTests(unittest.TestCase):
    def test_inactive_scope_is_an_unmodified_call(self):
        original = Mock(return_value="result")
        mask = ReferenceQueryMask(original)
        self.assertEqual(mask("query", "cache", "indices", 0.5), "result")
        original.assert_called_once_with("query", "cache", "indices", 0.5)

    def test_exception_does_not_leave_a_mask_active(self):
        mask = ReferenceQueryMask(Mock(return_value="normal"))
        with self.assertRaises(RuntimeError):
            with mask.activate(2, 4, 3):
                raise RuntimeError("cancelled")
        self.assertEqual(mask("q", "cache", "ids", 1), "normal")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch environment required")
class QuerySelectionTests(unittest.TestCase):
    def test_recomputed_exact_prefix_and_exact_tail_keep_all_candidates(self):
        import torch

        q = torch.ones(5, 1, 512)
        cache = torch.zeros(8, 656, dtype=torch.uint8)
        indices = torch.arange(15).reshape(5, 3)
        original = Mock(side_effect=lambda x, c, i, s: x * s)
        mask = ReferenceQueryMask(original)
        with mask.activate(2, 5, 7, start=2):
            output = mask(q, cache, indices, 0.5)
        expected = q * 0.5
        expected[2:4] = 0
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(original.call_count, 2)
        self.assertTrue(torch.equal(original.call_args_list[0].args[2], indices[:2]))
        self.assertTrue(torch.equal(original.call_args_list[1].args[2], indices[4:]))
        self.assertEqual(mask.counts, {7: 2})

    def test_only_unused_queries_are_skipped_and_all_candidates_are_retained(self):
        import torch

        q = torch.arange(4 * 2 * 512, dtype=torch.float32).reshape(4, 2, 512)
        cache = torch.zeros(8, 656, dtype=torch.uint8)
        indices = torch.arange(12).reshape(4, 3)
        original = Mock(side_effect=lambda x, c, i, s: x * s)
        mask = ReferenceQueryMask(original)
        with mask.activate(2, 4, 7):
            output = mask(q, cache, indices, 0.5)
        self.assertTrue(torch.equal(output[:2], torch.zeros_like(q[:2])))
        self.assertTrue(torch.equal(output[2:], q[2:] * 0.5))
        passed = original.call_args.args
        self.assertIs(passed[1], cache)
        self.assertTrue(torch.equal(passed[2], indices[2:]))
        self.assertEqual(mask.counts, {7: 2})

    def test_all_history_avoids_reference_kernel_and_shape_drift_fails(self):
        import torch

        q = torch.ones(4, 2, 512)
        cache = torch.zeros(8, 656, dtype=torch.uint8)
        indices = torch.zeros(4, 3, dtype=torch.int64)
        original = Mock()
        mask = ReferenceQueryMask(original)
        with mask.activate(4, 4, 7):
            output = mask(q, cache, indices, 1)
        original.assert_not_called()
        self.assertEqual(output.count_nonzero().item(), 0)
        with mask.activate(2, 5, 7), self.assertRaises(ValueError):
            mask(q, cache, indices, 1)


if __name__ == "__main__":
    unittest.main()
