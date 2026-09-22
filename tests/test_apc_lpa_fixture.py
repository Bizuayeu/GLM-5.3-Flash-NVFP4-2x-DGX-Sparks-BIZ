import json
import subprocess
import sys
import unittest
from pathlib import Path

from glm53_setup.validation import run_apc_lpa_fixture as fixture

ROOT = Path(__file__).resolve().parents[1]


def worker(cached, skipped):
    return {
        "policy": {"policy": {"cached_tokens": cached}},
        "lpa": {"mla_queries_skipped": skipped} if skipped is not None else None,
    }


class PolicyAssertionTests(unittest.TestCase):
    def test_accepts_a_row_whose_hit_and_omission_match_on_every_worker(self):
        row = {"cached_tokens": 256, "workers": [worker(256, {3: 1000})] * 2}
        fixture.assert_policy(row, 256, 1000)
        row = {"cached_tokens": 0, "workers": [worker(0, {}), worker(0, None)]}
        fixture.assert_policy(row, 0, 0)

    def test_rejects_an_unexpected_hit_a_disagreeing_worker_or_wrong_omission(self):
        with self.assertRaisesRegex(ValueError, "Unexpected actual cache hit"):
            fixture.assert_policy({"cached_tokens": 1, "workers": []}, 0, 0)
        with self.assertRaisesRegex(ValueError, "Worker H differs"):
            fixture.assert_policy(
                {"cached_tokens": 256, "workers": [worker(0, {})]}, 256, 0
            )
        with self.assertRaisesRegex(ValueError, "Incorrect actual query omission"):
            fixture.assert_policy(
                {"cached_tokens": 256, "workers": [worker(256, {3: 999})]}, 256, 1000
            )
        with self.assertRaisesRegex(ValueError, "Incorrect actual query omission"):
            fixture.assert_policy(
                {"cached_tokens": 256, "workers": [worker(256, {3: 5})]}, 256, 0
            )


class JudgementTests(unittest.TestCase):
    def test_first_distribution_delta_compares_only_the_shared_top_logprobs(self):
        teacher = {"logprobs": [{"1": -0.1, "2": -2.0, "3": -3.0}]}
        altered = {"logprobs": [{"1": -0.4, "2": -2.0, "9": -0.5}]}
        self.assertAlmostEqual(fixture.first_distribution_delta(teacher, altered), 0.3)
        self.assertEqual(
            fixture.first_distribution_delta(teacher, {"logprobs": [{"9": -1.0}]}), 0
        )

    def test_history_positions_cross_the_block_boundary_and_span_the_prompt(self):
        self.assertEqual(
            fixture.history_positions(block=256, length=2000),
            [3, 4, 5, 200, 255, 256, 257, 1000, 1800],
        )
        # Duplicates collapse when the prompt is short.
        self.assertEqual(
            fixture.history_positions(block=4, length=50), [3, 4, 5, 25, 45]
        )

    def test_expected_omission_applies_the_tail_and_the_break_even(self):
        # eligible = prompt - tail - hit, omitted only above the break-even
        self.assertEqual(fixture.expected_omission(4096, 256), 4096 - 512 - 256)
        self.assertEqual(fixture.expected_omission(1536, 0), 0)  # exactly 1024
        self.assertEqual(fixture.expected_omission(1537, 0), 1025)
        self.assertEqual(fixture.expected_omission(300, 0), 0)  # shorter than tail
        self.assertEqual(fixture.expected_omission(2000, 1900), 0)

    def test_lpa_config_names_the_synthetic_projector_and_the_measurement_terms(self):
        config = json.loads(fixture.lpa_config(Path("/tmp/p.pt"), "ab" * 32))
        self.assertEqual(
            config,
            {
                "cut": 0,
                "tail": 512,
                "break_even": 1024,
                "projector_path": str(Path("/tmp/p.pt")),
                "projector_sha256": "ab" * 32,
                "skip_mla_queries": True,
            },
        )
        self.assertEqual(fixture.TAIL, config["tail"])
        self.assertEqual(fixture.BREAK_EVEN, config["break_even"])


class ImportTests(unittest.TestCase):
    def test_the_judgements_import_without_torch_or_vllm(self):
        code = (
            "import sys; sys.modules['torch'] = None; sys.modules['vllm'] = None\n"
            "from glm53_setup.validation import run_apc_lpa_fixture as f\n"
            "print(f.expected_omission(4096, 0), len(f.engine_kwargs(f.parser().parse_args("
            "['--fixture', 'x', '--output', 'y']))))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["3584", "18"])


if __name__ == "__main__":
    unittest.main()
