import contextlib
import hashlib
import io
import json
import math
import unittest
from unittest.mock import patch

from glm53_setup.validation import apc_history
from glm53_setup.validation.apc_history import (
    answer_matches,
    common_prefix,
    expected_omission,
    history_cases,
)


class HistoryCaseTests(unittest.TestCase):
    def test_partial_or_under_repeated_timings_cannot_pose_as_full_functional_validation(
        self,
    ):
        base = [
            "--config",
            "unused",
            "--corpus",
            "unused",
            "--corpus-sha256",
            "0" * 64,
            "--output",
            "unused",
            "--block-tokens",
            "4352",
        ]
        for options in (
            ["--case-ids", "edit-50"],
            ["--timing-only", "--case-ids", "edit-50", "--repeats", "1"],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                apc_history.main(base + options)
            self.assertEqual(caught.exception.code, 2)

    def test_edits_change_the_answer_before_the_common_question(self):
        cases, *_ = history_cases(["early", "middle", "late", "end"])
        self.assertEqual(len(cases), 7)
        for case in cases:
            if case["id"].startswith("edit"):
                self.assertNotEqual(case["expected"], case["prime_expected"])
                self.assertEqual(
                    case["prime"]["messages"][-1], case["request"]["messages"][-1]
                )
                self.assertNotEqual(
                    case["prime"]["messages"][1], case["request"]["messages"][1]
                )
        self.assertEqual(common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(common_prefix([1, 2], [1, 2, 3]), 2)

    def test_stale_or_mixed_answer_is_not_accepted_as_format_variation(self):
        self.assertTrue(answer_matches("**MIA110001**", "MIA110001"))
        self.assertFalse(answer_matches("MIA550005", "MIA110001"))
        self.assertFalse(answer_matches("MIA110001 or MIA550005", "MIA110001"))
        for malformed in (
            "MIA1100019",
            "XMIA110001",
            "MIA110001_suffix",
            "MIA110001-9",
        ):
            self.assertFalse(answer_matches(malformed, "MIA110001"))


class ExpectedOmissionTests(unittest.TestCase):
    def test_every_eligible_row_is_skipped_on_each_mla_layer_after_the_cut(self):
        self.assertEqual(
            expected_omission(32, 7), {"35": 7, "39": 7, "43": 7}
        )  # The calibration cut; 44 is the model's last layer.
        self.assertEqual(expected_omission(40, 7), {"43": 7})
        self.assertEqual(expected_omission(0, 1, layers=4), {"3": 1})


class PolicyHitTests(unittest.TestCase):
    """The admission check both ranks must agree on, lifted out of main().

    It used to be a closure over the profile and the collective RPC, so only a
    running two-rank pair could reach it. The arithmetic it performs -- how
    many positions are eligible for approximation once the tail and the cached
    prefix are removed -- decides what every APC/LPA history reading means.
    """

    PROFILE = {"lpa": {"tail": 256, "cut": 32, "break_even_tokens": 512}}

    def worker(self, *, cached, prompt, skipped=None, limit=None):
        return {
            "policy": {
                "policy": {
                    "cached_tokens": cached,
                    "prompt_tokens": prompt,
                    "shared_cache_limit": limit,
                }
            },
            "lpa": {"mla_queries_skipped": skipped} if skipped is not None else None,
        }

    def call(self, workers, length, mode):
        return apc_history.policy_hits(self.PROFILE, lambda _: workers, length, mode)

    def test_off_mode_skips_nothing_however_long_the_prompt(self):
        pair = [self.worker(cached=64, prompt=8192) for _ in range(2)]
        hit, workers = self.call(pair, 8192, "off")
        self.assertEqual(hit, 64)
        self.assertEqual(len(workers), 2)

    def test_auto_mode_approximates_what_the_tail_and_the_cache_leave(self):
        # 8192 - 256 tail - 64 already cached = 7872 eligible, above break-even,
        # on every fourth layer after the cut.
        skipped = {str(layer): 7872 for layer in range(33, 45) if layer % 4 == 3}
        pair = [
            self.worker(cached=64, prompt=8192, skipped=skipped, limit=64)
            for _ in range(2)
        ]
        self.assertEqual(self.call(pair, 8192, "auto")[0], 64)

    def test_a_prompt_under_the_break_even_point_is_left_exact(self):
        pair = [self.worker(cached=0, prompt=600, skipped={}) for _ in range(2)]
        self.assertEqual(self.call(pair, 600, "auto")[0], 0)

    def test_a_scheduler_that_counted_different_tokens_is_refused(self):
        pair = [self.worker(cached=0, prompt=999) for _ in range(2)]
        with self.assertRaises(ValueError):
            self.call(pair, 1000, "off")

    def test_ranks_that_disagree_on_the_joint_hit_are_refused(self):
        pair = [self.worker(cached=64, prompt=8192), self.worker(cached=0, prompt=8192)]
        with self.assertRaises(ValueError):
            self.call(pair, 8192, "off")

    def test_a_single_rank_is_never_a_pair(self):
        with self.assertRaises(ValueError):
            self.call([self.worker(cached=0, prompt=100)], 100, "off")

    def test_work_actually_skipped_must_match_what_the_policy_promised(self):
        pair = [
            self.worker(cached=0, prompt=8192, skipped={"35": 1}, limit=0)
            for _ in range(2)
        ]
        with self.assertRaises(ValueError):
            self.call(pair, 8192, "auto")


class CorpusTests(unittest.TestCase):
    def corpus(self):
        rows = [
            {"split": "validation", "text": "first"},
            {"split": "train", "text": "never"},
            {"split": "validation", "text": "second"},
        ]
        return "\n".join(json.dumps(row) for row in rows).encode()

    def test_only_the_validation_split_is_read_in_file_order(self):
        raw = self.corpus()
        digest = hashlib.sha256(raw).hexdigest()
        self.assertEqual(apc_history.validation_text(raw, digest), "first\n\nsecond")

    def test_a_corpus_other_than_the_pinned_one_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^Corpus hash differs$"):
            apc_history.validation_text(self.corpus(), "0" * 64)


class CacheLayoutTests(unittest.TestCase):
    """The running pair must hold the retention and block the run assumes."""

    def row(self, retention=8, blocks=(16,)):
        return {
            "retention_interval": retention,
            "groups": [{"block_size": size} for size in blocks],
        }

    def test_a_matching_pair_passes(self):
        layout = [self.row(blocks=(16, 64)), self.row(blocks=(64, 16))]
        apc_history.check_cache_layout(layout, 8, 64)

    def test_a_rank_with_another_retention_is_refused(self):
        with self.assertRaisesRegex(
            ValueError,
            "^Actual cache retention differs from the selected pinned "
            "baseline/candidate$",
        ):
            apc_history.check_cache_layout([self.row(), self.row(retention=4)], 8, 16)

    def test_the_block_is_the_joint_alignment_of_every_group(self):
        message = (
            "^Supplied scheduler block does not match the pinned worker "
            "group's joint alignment$"
        )
        for layout, block in (
            ([self.row(blocks=(16, 24))] * 2, 24),  # lcm is 48
            ([self.row(blocks=(16,)), self.row(blocks=(32,))], 32),  # ranks differ
        ):
            with self.subTest(block=block), self.assertRaisesRegex(ValueError, message):
                apc_history.check_cache_layout(layout, 8, block)
        apc_history.check_cache_layout([self.row(blocks=(16, 24))] * 2, 8, 48)

    def test_retention_is_checked_before_alignment(self):
        with self.assertRaisesRegex(ValueError, "retention"):
            apc_history.check_cache_layout([self.row(retention=1, blocks=(3,))], 8, 16)


class TimingCaseTests(unittest.TestCase):
    CASES = [{"id": "edit-10"}, {"id": "edit-50"}, {"id": "append"}]

    def test_the_named_cases_come_back_in_the_order_asked(self):
        chosen = apc_history.select_timing_cases(self.CASES, ["append", "edit-10"])
        self.assertEqual([case["id"] for case in chosen], ["append", "edit-10"])

    def test_an_unknown_or_repeated_id_is_refused(self):
        for ids in (["edit-10", "edit-10"], ["edit-90x"]):
            with (
                self.subTest(ids=ids),
                self.assertRaisesRegex(
                    ValueError, "^Unknown or duplicate timing case IDs$"
                ),
            ):
                apc_history.select_timing_cases(self.CASES, ids)


def completion(prompt_tokens=5, completion_tokens=1, token_ids=(7,), logprobs=(-0.5,)):
    return {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "choices": [
            {
                "token_ids": list(token_ids),
                "logprobs": {"token_logprobs": list(logprobs)},
            }
        ],
    }


class OneOutputTests(unittest.TestCase):
    def test_one_finite_token_on_the_encoded_prompt_is_complete(self):
        self.assertTrue(apc_history.one_output_complete(completion(), 5))

    def test_every_departure_makes_the_sample_incomplete(self):
        for name, response in {
            "two outputs counted": completion(completion_tokens=2),
            "another prompt length": completion(prompt_tokens=6),
            "two token ids": completion(token_ids=(7, 8)),
            "nonfinite logprob": completion(logprobs=(math.nan,)),
            "infinite logprob": completion(logprobs=(-math.inf,)),
        }.items():
            with self.subTest(case=name):
                self.assertFalse(apc_history.one_output_complete(response, 5))


class PrefillTurnTests(unittest.TestCase):
    """The timing sample: one exact output, timed between two metric reads."""

    PROFILE = {
        "api": {"served_model_name": "glm"},
        "runtime": {"seed": 42},
        "lpa": {"tail": 256, "cut": 32, "break_even_tokens": 512},
    }

    def run_turn(self, response):
        order = []
        worker = {
            "policy": {
                "policy": {
                    "cached_tokens": 2,
                    "prompt_tokens": 5,
                    "shared_cache_limit": None,
                }
            },
            "lpa": None,
        }

        def post(profile, path, body):
            order.append(path)
            self.body = body
            return response

        def rpc(method):
            order.append(method)
            return [worker, worker]

        def metrics(profile):
            order.append("metrics")
            return f"m{len(order)}"

        clock = iter([1.0, 1.25])
        with (
            patch.object(apc_history, "encode_prompt", return_value=[1, 2, 3, 4, 5]),
            patch.object(apc_history, "read_metrics", side_effect=metrics),
            patch.object(apc_history.server, "post", side_effect=post),
            patch.object(apc_history.time, "perf_counter", lambda: next(clock)),
            patch.object(apc_history.server_config, "request_body", lambda _, b: b),
        ):
            row = apc_history.prefill_turn(self.PROFILE, rpc, {"messages": []})
        return row, order

    def test_the_sample_is_timed_between_two_metric_reads(self):
        row, order = self.run_turn(completion())
        self.assertEqual(
            order, ["metrics", "/v1/completions", "apc_lpa_report", "metrics"]
        )
        self.assertEqual(
            list(row),
            [
                "seconds",
                "prompt_token_ids",
                "cached_tokens",
                "workers",
                "response",
                "metrics_before",
                "metrics_after",
            ],
        )
        self.assertEqual(row["seconds"], 0.25)
        self.assertEqual(row["cached_tokens"], 2)
        self.assertEqual((row["metrics_before"], row["metrics_after"]), ("m1", "m4"))
        self.assertEqual(self.body["prompt"], [1, 2, 3, 4, 5])
        self.assertEqual(self.body["vllm_xargs"], {apc_history.MODE_KEY: "off"})

    def test_an_incomplete_sample_is_refused(self):
        with self.assertRaisesRegex(
            ValueError, "^Incomplete one-output prefill sample$"
        ):
            self.run_turn(completion(completion_tokens=2))


class CommonPrefixCheckTests(unittest.TestCase):
    PRIME = {"prompt_token_ids": [1, 2, 3, 4]}

    def test_a_hit_within_the_common_prefix_returns_its_length(self):
        row = {"prompt_token_ids": [1, 2, 9, 4], "cached_tokens": 2}
        self.assertEqual(apc_history.checked_common_prefix(self.PRIME, row, "x"), 2)

    def test_a_hit_beyond_it_fails_with_the_callers_message(self):
        row = {"prompt_token_ids": [1, 2, 9, 4], "cached_tokens": 3}
        with self.assertRaisesRegex(ValueError, "^Timing sample reused$"):
            apc_history.checked_common_prefix(self.PRIME, row, "Timing sample reused")


class BoundaryTests(unittest.TestCase):
    def test_lengths_straddle_one_and_two_blocks(self):
        self.assertEqual(
            apc_history.boundary_lengths(16), [3, 4, 5, 15, 16, 17, 31, 32, 33]
        )
        # Tiny blocks overlap the fixed short lengths; each length runs once.
        self.assertEqual(apc_history.boundary_lengths(2), [1, 2, 3, 4, 5])

    def rows(self, limit, cached):
        auto = {"workers": [{"policy": {"policy": {"shared_cache_limit": limit}}}]}
        return [auto, {"cached_tokens": cached}, {"cached_tokens": cached}]

    def test_an_approximated_request_must_leave_nothing_to_reuse(self):
        self.assertTrue(apc_history.boundary_contaminated(self.rows(0, 16)))

    def test_an_exact_request_or_an_empty_cache_is_clean(self):
        self.assertFalse(apc_history.boundary_contaminated(self.rows(None, 16)))
        self.assertFalse(apc_history.boundary_contaminated(self.rows(0, 0)))

    def test_only_the_first_rank_and_first_exact_repeat_decide(self):
        rows = self.rows(None, 0)
        rows[0]["workers"].append({"policy": {"policy": {"shared_cache_limit": 0}}})
        rows[2]["cached_tokens"] = 16
        self.assertFalse(apc_history.boundary_contaminated(rows))


class QualityTests(unittest.TestCase):
    def report(self):
        return {
            "cases": [
                {
                    "arms": [
                        {"prime": {"passed": True}, "result": {"passed": True}}
                        for _ in range(2)
                    ]
                }
            ],
            "revisits": [{"result": {"passed": True}}],
            "pressure": [{"passed": True}, {"passed": True}],
            "pressure_revisit": {"after": {"passed": True}},
        }

    def test_every_answer_right_passes(self):
        self.assertTrue(apc_history.quality_passed(self.report()))

    def test_any_wrong_answer_fails(self):
        for path in (
            ("cases", 0, "arms", 1, "prime"),
            ("cases", 0, "arms", 0, "result"),
            ("revisits", 0, "result"),
            ("pressure", 1),
            ("pressure_revisit", "after"),
        ):
            report = self.report()
            node = report
            for key in path:
                node = node[key]
            node["passed"] = False
            with self.subTest(path=path):
                self.assertFalse(apc_history.quality_passed(report))


class ParserTests(unittest.TestCase):
    def test_the_parser_is_reachable_without_running_main(self):
        args = apc_history.parser().parse_args(
            [
                "--config",
                "c",
                "--corpus",
                "k",
                "--corpus-sha256",
                "0" * 64,
                "--output",
                "o",
                "--block-tokens",
                "16",
            ]  # fmt: skip
        )
        self.assertEqual(
            (args.corpus_tokens, args.repeats, args.pressure_histories),
            (16000, 1, 12),
        )
        self.assertFalse(args.timing_only)


if __name__ == "__main__":
    unittest.main()
