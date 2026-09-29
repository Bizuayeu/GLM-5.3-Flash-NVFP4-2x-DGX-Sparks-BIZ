import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tools import nccl_probe


class NcclProbeCliTests(unittest.TestCase):
    def test_help_does_not_import_gpu_dependencies(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as result,
        ):
            nccl_probe.main(["--help"])
        self.assertEqual(result.exception.code, 0)

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rank.json"
            output.write_text("previous evidence", encoding="utf-8")
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as result,
            ):
                nccl_probe.main(
                    ["--rank", "0", "--head", "10.53.0.1", "--output", str(output)]
                )
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous evidence")


class NcclProbeMathTests(unittest.TestCase):
    """What every rank must hold, for two ranks and for the three-node ring."""

    def test_an_all_reduce_of_pattern_plus_rank_plus_one(self):
        for pattern in range(7):
            # Two ranks: the 1.21.0 expectation, pattern * 2 + 3.
            self.assertEqual(nccl_probe.reduced(pattern, 2), pattern * 2 + 3)
            # N ranks: N * pattern + N(N+1)/2.
            self.assertEqual(nccl_probe.reduced(pattern, 3), 3 * pattern + 6)
            self.assertEqual(
                sum(pattern + rank + 1 for rank in range(3)),
                nccl_probe.reduced(pattern, 3),
            )

    def test_ten_timed_all_reduces_multiply_by_n_nine_times(self):
        for pattern in range(7):
            self.assertEqual(
                nccl_probe.repeated(nccl_probe.reduced(pattern, 2), 2, 9),
                (pattern * 2 + 3) * 512,
            )
            self.assertEqual(
                nccl_probe.repeated(nccl_probe.reduced(pattern, 3), 3, 9),
                (3 * pattern + 6) * 3**9,
            )

    def test_gather_and_reduce_scatter_slices(self):
        n = 4
        pattern = [i % 7 for i in range(n)]
        gathered = [
            [nccl_probe.contribution(p, rank) for p in pattern] for rank in range(3)
        ]
        self.assertEqual(gathered[2], [p + 3 for p in pattern])
        whole = [i % 7 for i in range(3 * n)]
        for rank in range(3):
            inputs = [
                [nccl_probe.contribution(w, sender) for w in whole]
                for sender in range(3)
            ]
            expected = [sum(column) for column in zip(*inputs, strict=True)][
                rank * n : (rank + 1) * n
            ]
            self.assertEqual(
                [nccl_probe.reduced(w, 3) for w in nccl_probe.shard(whole, rank, n)],
                expected,
            )

    def test_world_size_bounds_the_rank_and_stops_at_three(self):
        for argv in (
            ["--rank", "2", "--world-size", "2"],
            ["--rank", "0", "--world-size", "4"],
            ["--rank", "0", "--world-size", "1"],
        ):
            with (
                self.subTest(argv=argv),
                tempfile.TemporaryDirectory() as tmp,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as result,
            ):
                nccl_probe.main(
                    [*argv, "--head", "10.41.12.1", "--output", f"{tmp}/r.json"]
                )
            self.assertEqual(result.exception.code, 2)
