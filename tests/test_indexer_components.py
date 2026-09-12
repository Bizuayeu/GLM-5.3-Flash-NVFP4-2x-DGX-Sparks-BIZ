import importlib.util
import unittest

from glm53_setup.runtime.indexer_candidates import CandidateReuse


class ReuseTests(unittest.TestCase):
    def test_request_scope_copy_and_exception_cleanup(self):
        row = {
            "request_id": "a",
            "layer": 3,
            "query_position": 8,
            "coordinate_space": "logical_tokens",
            "indices": [1, 2, -1],
        }
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            with CandidateReuse("a", [(3, 7)]) as session:
                session.remember(row)
                row["indices"][0] = 8
                self.assertEqual(session.selection("a", 7, 8), [1, 2, -1])
                with self.assertRaises(ValueError):
                    session.selection("other", 7, 8)
                with self.assertRaises(KeyError):
                    session.selection("a", 7, 9)
                raise RuntimeError("cancelled")
        self.assertEqual(session.cache, {})
        with self.assertRaises(ValueError):
            session.selection("a", 7, 8)

    def test_invalid_pairs_and_physical_coordinates_are_rejected(self):
        for pairs in [[(7, 3)], [(3, 7), (4, 7)]]:
            with self.assertRaises(ValueError):
                CandidateReuse("a", pairs)
        with CandidateReuse("a", [(3, 7)]) as session:
            with self.assertRaises(ValueError):
                session.remember(
                    {
                        "request_id": "a",
                        "layer": 3,
                        "query_position": 8,
                        "coordinate_space": "physical_slots",
                        "indices": [1],
                    }
                )


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required")
class IndexerTensorTests(unittest.TestCase):
    def test_unique_pool_selection_and_incomplete_tail_boundaries(self):
        import torch

        from glm53_setup.runtime.indexer_candidates import (
            expand_pools_reference,
            select_candidate_pools,
        )

        selected = select_candidate_pools(
            torch.tensor([[3.0, 3.0, 2.0, 100.0, 3.0]]),
            torch.tensor([[2, 2, 0, -1, 1]]),
            4,
        )
        self.assertEqual(selected.tolist(), [[1, 2, 0, -1]])
        for length, expected in [
            (3, [0, 1, 2]),
            (4, [0, 1, 2, 3]),
            (5, [0, 1, 2, 3, 4]),
            (7, list(range(7))),
            (8, list(range(8))),
        ]:
            result = expand_pools_reference(
                torch.tensor([[0, 1, -1]]), torch.tensor([length])
            )
            self.assertEqual(result[result >= 0].tolist(), expected)
        with self.assertRaises(ValueError):
            select_candidate_pools(
                torch.tensor([[1.0, 2.0]]), torch.tensor([[0, 0]]), 1
            )

    def test_capture_clones_shared_buffer_and_restores_hooks(self):
        import torch

        from glm53_setup.runtime.indexer_capture import IndexerCapture

        class Indexer(torch.nn.Module):
            def forward(self, values, *, positions):
                return values

        module = Indexer()
        values = torch.tensor([[0, -1], [1, 2]])
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            with IndexerCapture({3: module}, "a", [2]) as capture:
                returned = module(values, positions=torch.tensor([0, 2]))
                self.assertIs(returned, values)
                values[1] = 0
                raise RuntimeError("cancelled")
        self.assertEqual(capture.report()["rows"][0]["indices"], [1, 2])
        self.assertEqual(len(module._forward_hooks), 0)
        self.assertEqual(len(module._forward_pre_hooks), 0)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            with IndexerCapture({3: module}, "b", [2], max_bytes=1):
                module(values, positions=torch.tensor([0, 2]))
        self.assertEqual(len(module._forward_hooks), 0)

    def test_reindex_subset_matches_dense_with_causality_and_padding(self):
        import torch

        from glm53_setup.runtime.indexer_candidates import candidate_scores_reference

        q = torch.tensor([[[1.0, -2.0], [2.0, 1.0]], [[1.0, 1.0], [-1.0, 2.0]]])
        k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [-2.0, 4.0]])
        scales = torch.tensor([0.5, 1.0, 2.0, 0.25])
        weights = torch.tensor([[1.0, -0.2], [0.7, 0.3]])
        ids = torch.tensor([[2, 0, -1], [1, 3, 1]])
        actual = candidate_scores_reference(
            q, k, scales, weights, ids, torch.tensor([3, 3])
        )
        dense = torch.stack(
            [
                torch.stack(
                    [
                        sum(
                            weights[i, h] * torch.dot(q[i, h], k[j] * scales[j]).relu()
                            for h in range(2)
                        )
                        for j in range(4)
                    ]
                )
                for i in range(2)
            ]
        )
        expected = torch.stack([dense[0, [2, 0, 0]], dense[1, [1, 3, 1]]])
        expected[0, 2] = expected[1, 1] = float("-inf")
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
