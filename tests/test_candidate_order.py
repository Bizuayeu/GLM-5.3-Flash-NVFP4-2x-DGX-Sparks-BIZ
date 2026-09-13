import importlib.util
import itertools
import os
import unittest
from unittest.mock import patch

from glm53_setup.runtime.candidate_order import (
    candidate_order_enabled,
    canonical_logical_candidates,
)
from glm53_setup.runtime.patch_nope_reference import add_candidate_order


class CandidateOrderPolicyTests(unittest.TestCase):
    def test_default_enabled_and_explicit_rollback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(candidate_order_enabled())
        for value, expected in (("0", False), ("1", True)):
            with patch.dict(os.environ, {"GLM53_CANONICAL_CANDIDATES": value}):
                self.assertEqual(candidate_order_enabled(), expected)
        with patch.dict(os.environ, {"GLM53_CANONICAL_CANDIDATES": "yes"}):
            with self.assertRaises(ValueError):
                candidate_order_enabled()


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required")
class CandidateOrderTensorTests(unittest.TestCase):
    def test_all_permutations_preserve_membership_padding_and_input(self):
        import torch

        for dtype in (torch.int32, torch.int64):
            maximum = torch.iinfo(dtype).max
            rows = list(set(itertools.permutations((-1, 9, 0, 9, maximum))))
            backing = torch.zeros((len(rows), 10), dtype=dtype)
            ids = backing[:, ::2]
            ids.copy_(torch.tensor(rows, dtype=dtype))
            before = backing.clone()
            result = canonical_logical_candidates(ids)
            expected = [sorted(x for x in row if x >= 0) + [-1] for row in rows]
            self.assertEqual(result.tolist(), expected)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, ids.device)
            self.assertTrue(torch.equal(backing, before))

    def test_empty_rows_all_padding_and_invalid_inputs(self):
        import torch

        for shape in ((0, 7), (3, 0)):
            ids = torch.empty(shape, dtype=torch.int32)
            self.assertEqual(canonical_logical_candidates(ids).shape, ids.shape)
        ids = torch.full((2, 2051), -1, dtype=torch.int64)
        self.assertTrue(torch.equal(canonical_logical_candidates(ids), ids))
        for ids in (
            torch.tensor([[0, -2]]),
            torch.tensor([1, 0]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[True, False]]),
        ):
            with self.assertRaises(ValueError):
                canonical_logical_candidates(ids)

    def test_patched_backend_orders_logical_ids_before_mapping(self):
        import torch

        source = """class Backend:
    def __init__(self, model_type, ids):
        self.topk_indices_buffer = ids
        self.supports_quant_query_input = False
    def run(self, num_actual_toks, block_table):
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        return map_indices(topk_indices, block_table)
"""

        def map_indices(ids, tables):
            safe = ids.clamp_min(0).long()
            blocks = tables.gather(1, safe // 4)
            return (blocks * 4 + safe % 4).masked_fill(ids < 0, -1)

        namespace = {"map_indices": map_indices}
        exec(compile(add_candidate_order(source), "backend-fixture", "exec"), namespace)
        ids = torch.tensor([[5, -1, 1, 4], [3, 0, -1, -1]], dtype=torch.int32)
        before = ids.clone()
        tables = torch.tensor([[17, 9], [4, 12]])
        with patch.dict(os.environ, {}, clear=True):
            backend = namespace["Backend"]("glm5_next_text", ids)
            # Physical order is deliberately different from logical order.
            self.assertEqual(
                backend.run(2, tables).tolist(), [[69, 36, 37, -1], [16, 19, -1, -1]]
            )
            other = namespace["Backend"]("other_model", ids)
            self.assertTrue(torch.equal(other.run(2, tables), map_indices(ids, tables)))
        with patch.dict(os.environ, {"GLM53_CANONICAL_CANDIDATES": "0"}):
            backend = namespace["Backend"]("glm5_next_text", ids)
            self.assertTrue(
                torch.equal(backend.run(2, tables), map_indices(ids, tables))
            )
        self.assertTrue(torch.equal(ids, before))
        with self.assertRaisesRegex(ValueError, "anchor"):
            add_candidate_order(
                source.replace("self.supports_quant_query_input = False", "pass")
            )


if __name__ == "__main__":
    unittest.main()
