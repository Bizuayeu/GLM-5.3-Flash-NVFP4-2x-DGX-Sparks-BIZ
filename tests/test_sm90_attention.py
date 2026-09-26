import unittest

from glm53_setup.validation import benchmark_sm90_attention as sm90


class SM90AttentionContractTests(unittest.TestCase):
    def test_cases_cover_every_width_and_layout_with_empty_rows_last(self):
        cases = sm90.parity_cases()
        filled = [c for c in cases if not c["empty_row"]]
        empty = [c for c in cases if c["empty_row"]]
        # A kernel fault on an empty row must not cost the other results.
        self.assertEqual(cases, filled + empty)
        self.assertEqual(
            {(c["width"], c["contiguous"]) for c in filled},
            {(w, c) for w in sm90.WIDTHS if w for c in (True, False)},
        )
        self.assertEqual(
            {(c["width"], c["contiguous"]) for c in empty},
            {(w, c) for w in sm90.WIDTHS for c in (True, False)},
        )

    def test_rows_hold_all_candidates_a_sole_tail_and_an_empty_row(self):
        rows = sm90.case_rows({"width": 2051, "empty_row": True})
        self.assertEqual(rows[0], list(range(2051)))
        self.assertEqual(rows[1], [sm90.TAIL_SLOT])
        self.assertEqual(rows[2], [])
        self.assertEqual(len(sm90.case_rows({"width": 17, "empty_row": False})), 2)
        self.assertEqual(sm90.case_rows({"width": 0, "empty_row": True}), [[], [], []])
        self.assertLess(max(sm90.WIDTHS), sm90.SLOTS)

    def test_padded_rows_keep_the_tail_last_for_the_reference(self):
        padded = sm90.padded_rows([[0, 1, 2], [9], []], 3)
        self.assertEqual(padded, [[0, 1, 2], [-1, -1, 9], [-1, -1, -1]])

    def test_compacted_rows_become_page_size_one_kv_ranges(self):
        indptr, indices, lengths = sm90.compact([[4, 5, 6], [], [9]])
        self.assertEqual(indptr, [0, 3, 3, 4])
        self.assertEqual(indices, [4, 5, 6, 9])
        self.assertEqual(lengths, [3, 0, 1])


if __name__ == "__main__":
    unittest.main()
