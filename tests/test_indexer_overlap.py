import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from glm53_setup.validation.indexer_overlap import compare_candidates, main


class IndexerOverlapTests(unittest.TestCase):
    def row(self, indices, **kwargs):
        return {
            "request_id": "one",
            "query_position": 4,
            "coordinate_space": "logical_tokens",
            "indices": indices,
            **kwargs,
        }

    def test_padding_duplicates_and_empty_rows_do_not_inflate_overlap(self):
        result = compare_candidates(self.row([-1, 1, 2, 2]), self.row([2, 3, -1]))
        self.assertAlmostEqual(result["jaccard"], 1 / 3)
        self.assertEqual(result["target_recall"], 0.5)
        self.assertIsNone(compare_candidates(self.row([-1]), self.row([-1]))["jaccard"])

    def test_extra_pool_recall_requires_separately_captured_candidates(self):
        result = compare_candidates(
            self.row([1, 2], candidate_pool=[1, 2, 3, 4]), self.row([2, 3])
        )
        self.assertEqual(result["target_recall_in_pool"], 1)
        with self.assertRaises(ValueError):
            compare_candidates(
                self.row([1, 2], candidate_pool=[3, 4]), self.row([2, 3])
            )

    def test_dense_prefix_agreement_is_flagged_as_trivial(self):
        row = self.row(list(range(5)))
        result = compare_candidates(row, row)
        self.assertTrue(result["trivial_full_coverage"])
        self.assertEqual(result["jaccard"], 1)

    def test_unaligned_physical_or_future_candidates_are_rejected(self):
        for values in ({"request_id": "two"}, {"query_position": 5}, {"indices": [5]}):
            target = self.row([1])
            target.update(values)
            with self.assertRaises(ValueError):
                compare_candidates(self.row([1]), target)
        with self.assertRaises(ValueError):
            compare_candidates(
                self.row([1], coordinate_space="physical_slots"),
                self.row([1], coordinate_space="physical_slots"),
            )


class MainTests(unittest.TestCase):
    row = IndexerOverlapTests.row

    def run_main(self, pairs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.json"
            path.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main([str(path)])
        return out.getvalue()

    def test_each_aligned_pair_is_compared_in_order(self):
        pairs = [
            {"source": self.row([1, 2]), "target": self.row([2, 3])},
            {"source": self.row([4]), "target": self.row([4])},
        ]
        result = json.loads(self.run_main(pairs))
        self.assertEqual([r["jaccard"] for r in result], [1 / 3, 1])
        self.assertEqual(json.loads(self.run_main([])), [])

    def test_one_unaligned_pair_fails_the_whole_capture(self):
        pairs = [
            {"source": self.row([1]), "target": self.row([1])},
            {"source": self.row([1]), "target": self.row([1], request_id="two")},
        ]
        with self.assertRaises(ValueError):
            self.run_main(pairs)
