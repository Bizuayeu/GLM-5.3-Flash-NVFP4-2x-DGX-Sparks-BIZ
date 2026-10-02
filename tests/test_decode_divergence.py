import contextlib
import io
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from tools import decode_divergence


def sample(ids, text=None):
    return {"token_ids": ids, "text": "".join(map(str, ids)) if text is None else text}


def compare(a, b):
    """Run the tool on two token records; return its exit code and printed lines."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, samples in (("a", a), ("b", b)):
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
            paths.append(str(path))
        out = io.StringIO()
        code = 0
        with (
            patch.object(sys, "argv", ["decode_divergence.py", *paths]),
            contextlib.redirect_stdout(out),
            # The tool leaves its input files to the garbage collector.
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", ResourceWarning)
            try:
                decode_divergence.main()
            except SystemExit as error:
                code = error.code
    return code, out.getvalue().splitlines()


class DecodeDivergenceTests(unittest.TestCase):
    def test_identical_runs_report_each_sample_identical(self):
        samples = [sample([1, 2, 3]), sample([4, 5])]
        code, lines = compare(samples, samples)
        self.assertIn(code, (0, None))
        self.assertEqual(
            lines,
            ["sample 0: identical (3 tokens)", "sample 1: identical (2 tokens)"],
        )

    def test_the_first_differing_token_and_text_are_reported(self):
        code, lines = compare(
            [sample([1, 2, 3, 4], "abcd")], [sample([1, 2, 9, 4], "abXd")]
        )
        self.assertEqual(
            lines[0],
            "sample 0: first differing token 2/4 (ids [3, 4] vs [9, 4]), char 2",
        )
        self.assertEqual(lines[1:], ["   A: 'abcd'", "   B: 'abXd'"])

    def test_a_shorter_first_run_differs_where_it_ends(self):
        _, lines = compare([sample([1, 2])], [sample([1, 2, 3])])
        self.assertEqual(
            lines[0], "sample 0: first differing token 2/2 (ids [] vs [3]), char 2"
        )

    # Until 1.26.2 a longer first run reported its own length, not where B ended.
    def test_a_shorter_second_run_differs_where_it_ends(self):
        _, lines = compare([sample([1, 2, 3])], [sample([1, 2])])
        self.assertTrue(
            lines[0].startswith("sample 0: first differing token 2/3 (ids [3] vs [])"),
            lines[0],
        )

    # Until 1.26.2 zip dropped the extra sample, so the runs read as identical.
    def test_differing_sample_counts_are_not_reported_as_matching(self):
        code, lines = compare([sample([1]), sample([2])], [sample([1])])
        self.assertTrue(
            code not in (0, None) or any("identical" not in line for line in lines),
            lines,
        )


if __name__ == "__main__":
    unittest.main()
