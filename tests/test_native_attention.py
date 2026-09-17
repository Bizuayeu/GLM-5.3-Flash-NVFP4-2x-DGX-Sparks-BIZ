import unittest

from glm53_setup.validation import benchmark_native_attention as native


class NativeAttentionRevisitTests(unittest.TestCase):
    def test_row_kinds_follow_the_v15_layout(self):
        self.assertEqual(native.row_kinds(3), ["full", "partial", "empty"])
        self.assertEqual(native.row_kinds(65)[:2], ["full", "partial"])
        self.assertEqual(native.row_kinds(65)[2:], ["empty"] * 63)

    def test_empty_row_errors_never_reach_filled_row_kinds(self):
        errors = native.row_kind_errors(
            [0.001, 0.002, 3.03125, 3.03125], ["full", "partial", "empty", "empty"]
        )
        self.assertEqual(errors, {"full": 0.001, "partial": 0.002, "empty": 3.03125})
        with self.assertRaises(ValueError):
            native.row_kind_errors([0.1], ["full", "empty"])

    def test_reduction_drops_only_the_lowest_ranked_pool_and_keeps_the_tail(self):
        ranking = list(range(511, -1, -1))  # pool 0 ranks last
        tail = [2048, 2049, 2050]
        reduced = native.reduce_candidates(ranking, tail)
        self.assertEqual(len(reduced), 2047)
        self.assertTrue({0, 1, 2, 3}.isdisjoint(reduced))
        self.assertEqual(reduced[-3:], tail)
        self.assertEqual(sorted(reduced[:-3]), list(range(4, 2048)))
        # A row that never filled 512 pools already fits; nothing is dropped.
        short = native.reduce_candidates(list(range(511)), tail)
        self.assertEqual(len(short), 511 * 4 + 3)
        self.assertEqual(short[:4], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
