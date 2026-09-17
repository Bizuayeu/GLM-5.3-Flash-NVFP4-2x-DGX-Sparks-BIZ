import importlib.util
import math
import unittest

from glm53_setup.validation.compare_agreement import (
    candidate_shift,
    distribution_shift,
    kl_rows,
)
from glm53_setup.validation.run_agreement_fixture import (
    candidate_positions,
    cycle_tokens,
)


def capture(position, indices, layer=3):
    return {
        "request_id": "long-cycle",
        "layer": layer,
        "query_position": position,
        "coordinate_space": "logical_tokens",
        "indices": indices,
    }


class PromptShapeTests(unittest.TestCase):
    def test_candidate_positions_start_past_the_limit_and_end_on_the_last_query(self):
        positions = candidate_positions(2500)
        self.assertEqual(positions, [2304, 2368, 2432, 2496, 2499])
        self.assertEqual(candidate_positions(2305), [2304])

    def test_cycle_repeats_the_texts_in_order(self):
        self.assertEqual(cycle_tokens([[1, 2], [3]], 7), [1, 2, 3, 1, 2, 3, 1])
        with self.assertRaises(ValueError):
            cycle_tokens([[]], 4)


class CandidateShiftTests(unittest.TestCase):
    def test_reports_the_worst_query_and_ignores_padding(self):
        reference = [capture(10, [0, 1, 2, -1]), capture(20, [0, 1, 2, 3])]
        candidate = [capture(10, [2, 1, 0, -1]), capture(20, [0, 1, 2, 9])]
        result = candidate_shift(reference, candidate)
        self.assertEqual(result["queries"], 2)
        self.assertEqual(result["identical_sets"], 0.5)
        self.assertEqual(result["jaccard_min"], 3 / 5)
        self.assertEqual(result["jaccard_min_query"], [3, 20])

    def test_layers_are_reported_apart(self):
        reference = [capture(10, [0, 1], layer=3), capture(10, [0, 1], layer=7)]
        candidate = [capture(10, [0, 1], layer=3), capture(10, [0, 2], layer=7)]
        result = candidate_shift(reference, candidate)
        self.assertEqual(result["by_layer"]["3"]["jaccard_mean"], 1.0)
        self.assertEqual(result["by_layer"]["7"]["jaccard_mean"], 1 / 3)
        self.assertEqual(result["jaccard_min_query"], [7, 10])

    def test_rejects_captures_of_different_queries(self):
        with self.assertRaises(ValueError):
            candidate_shift([capture(10, [0])], [capture(11, [0])])
        with self.assertRaises(ValueError):
            candidate_shift([], [])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy required")
class DistributionShiftTests(unittest.TestCase):
    def test_kl_matches_the_closed_form_and_is_zero_on_itself(self):
        import numpy as np

        p = np.log(np.array([[0.5, 0.5], [0.9, 0.1]], dtype=np.float32))
        q = np.log(np.array([[0.25, 0.75], [0.9, 0.1]], dtype=np.float32))
        expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        kl = kl_rows(p, q)
        self.assertAlmostEqual(kl[0], expected, places=6)
        self.assertAlmostEqual(kl[1], 0.0, places=6)
        self.assertEqual(kl_rows(p, p).tolist(), [0.0, 0.0])

    def test_summary_counts_argmax_changes_across_blocks(self):
        import numpy as np

        p = np.log(np.array([[0.6, 0.4]] * 5, dtype=np.float32))
        q = p.copy()
        q[4] = np.log(np.array([0.4, 0.6], dtype=np.float32))
        result = distribution_shift(p, q, block=2)
        self.assertEqual(result["positions"], 5)
        self.assertEqual(result["argmax_agreement"], 0.8)
        self.assertEqual(result["kl_max_position"], 4)
        self.assertAlmostEqual(result["kl_max"], 0.2 * math.log(1.5), places=6)

    def test_rejects_shape_mismatch_and_infinities(self):
        import numpy as np

        p = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            kl_rows(p, p[:1])
        q = p.copy()
        q[0, 0] = -np.inf
        with self.assertRaises(ValueError):
            kl_rows(p, q)


if __name__ == "__main__":
    unittest.main()
