import copy
import json
import math
import unittest

from glm53_setup import agreement

TOKENS = [11, 22, 33, 44]


def entry(*pairs):
    """A vLLM prompt_logprobs token map from (token_id, logprob) pairs."""
    return {
        str(token): {"logprob": logprob, "rank": rank, "decoded_token": f"t{token}"}
        for rank, (token, logprob) in enumerate(pairs, start=1)
    }


def response(prompt_logprobs):
    return {"choices": [{"prompt_logprobs": prompt_logprobs, "text": ""}]}


# Position 1 predicts 22 first, position 2 puts the actual 33 second,
# position 3 keeps the actual 44 outside the top-two but present.
GOOD = [
    None,
    entry((22, -0.1), (99, -2.5)),
    entry((55, -0.4), (33, -1.2)),
    entry((66, -0.3), (77, -1.9), (44, -4.0)),
]


class ParseTests(unittest.TestCase):
    def test_rows_are_sorted_by_logprob_and_carry_the_prompt_token(self):
        rows = agreement.parse_prompt_logprobs(GOOD, TOKENS)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], [(22, -0.1), (99, -2.5)])
        self.assertEqual(rows[2][2], (44, -4.0))
        # Ties break on token id so two runs sort identically.
        tied = [None, entry((5, -1.0), (3, -1.0))]
        self.assertEqual(
            agreement.parse_prompt_logprobs(tied, [1, 3]), [[(3, -1.0), (5, -1.0)]]
        )

    def test_shape_mistakes_raise_instead_of_scoring(self):
        cases = {
            "empty": ([], []),
            "length": (GOOD, TOKENS[:-1]),
            "first-not-none": ([entry((1, -1.0))] + GOOD[1:], TOKENS),
            "not-a-map": ([None, [], GOOD[2], GOOD[3]], TOKENS),
            "empty-map": ([None, {}, GOOD[2], GOOD[3]], TOKENS),
            "missing-prompt-token": (
                [None, entry((99, -0.1)), GOOD[2], GOOD[3]],
                TOKENS,
            ),
            "non-finite": (
                [None, entry((22, float("nan"))), GOOD[2], GOOD[3]],
                TOKENS,
            ),
        }
        for name, (logprobs, tokens) in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                agreement.parse_prompt_logprobs(logprobs, tokens)


class TeacherForcedTests(unittest.TestCase):
    def test_ranks_top1_top5_and_mean_nll(self):
        rows = agreement.parse_prompt_logprobs(GOOD, TOKENS)
        forced = agreement.teacher_forced(rows, TOKENS)
        self.assertEqual(forced["ranks"], [1, 2, 3])
        self.assertAlmostEqual(forced["top1"], 1 / 3)
        self.assertAlmostEqual(forced["top5"], 1.0)
        self.assertAlmostEqual(forced["mean_nll"], (0.1 + 1.2 + 4.0) / 3)
        self.assertEqual(forced["logprobs"], [-0.1, -1.2, -4.0])
        with self.assertRaises(ValueError):
            agreement.teacher_forced(rows, TOKENS[:-1])

    def test_top_k_is_a_parameter(self):
        rows = agreement.parse_prompt_logprobs(GOOD, TOKENS)
        forced = agreement.teacher_forced(rows, TOKENS, top_k=2)
        self.assertAlmostEqual(forced["top2"], 2 / 3)


class AgreementTests(unittest.TestCase):
    def test_identical_runs_agree_completely(self):
        rows = agreement.parse_prompt_logprobs(GOOD, TOKENS)
        result = agreement.agreement(rows, copy.deepcopy(rows))
        self.assertEqual(result["argmax_agreement"], 1.0)
        self.assertEqual(result["top5_overlap_mean"], 1.0)
        self.assertEqual(result["top5_overlap_min"], 1.0)
        self.assertIsNone(result["first_argmax_divergence"])
        self.assertEqual(
            agreement.drift([-1.0, -2.0], [-1.0, -2.0])["max_abs_logprob_move"], 0.0
        )

    def test_divergence_is_located_and_overlap_is_jaccard(self):
        rows = agreement.parse_prompt_logprobs(GOOD, TOKENS)
        changed = copy.deepcopy(rows)
        changed[1] = [(33, -0.9), (55, -1.0)]  # argmax flips, same top-2 set
        changed[2] = [(66, -0.3), (88, -1.0), (44, -4.5)]  # 77 replaced by 88
        result = agreement.agreement(rows, changed, top_k=2)
        self.assertAlmostEqual(result["argmax_agreement"], 2 / 3)
        self.assertEqual(result["first_argmax_divergence"], 2)
        self.assertAlmostEqual(result["top2_overlap_min"], 1 / 3)
        moved = agreement.drift([-0.1, -1.2, -4.0], [-0.1, -0.9, -4.5])
        self.assertAlmostEqual(moved["max_abs_logprob_move"], 0.5)
        self.assertAlmostEqual(moved["mean_abs_logprob_move"], 0.8 / 3)

    def test_mismatched_or_empty_inputs_raise(self):
        rows = agreement.parse_prompt_logprobs(GOOD, TOKENS)
        with self.assertRaises(ValueError):
            agreement.agreement(rows, rows[:-1])
        with self.assertRaises(ValueError):
            agreement.agreement([], [])
        with self.assertRaises(ValueError):
            agreement.drift([], [])


class RunTests(unittest.TestCase):
    def senders(self, prompt_logprobs=GOOD, fail_on=None):
        calls = {"tokenize": [], "complete": []}

        def tokenize(text):
            calls["tokenize"].append(text)
            return list(TOKENS)

        def complete(token_ids, top_k):
            calls["complete"].append((tuple(token_ids), top_k))
            if fail_on is not None and len(calls["complete"]) == fail_on:
                raise TimeoutError("slow")
            return response(copy.deepcopy(prompt_logprobs))

        return tokenize, complete, calls

    def test_scores_every_text_twice_and_checks_the_instrument(self):
        tokenize, complete, calls = self.senders()
        texts = {"a": "alpha", "b": "beta"}
        record = agreement.run(tokenize, complete, texts=texts, repeats=2)
        self.assertEqual(calls["tokenize"], ["alpha", "alpha", "beta", "beta"])
        self.assertTrue(all(k == agreement.TOP_K for _, k in calls["complete"]))
        self.assertEqual([row["name"] for row in record["texts"]], ["a", "a", "b", "b"])
        self.assertEqual([c["name"] for c in record["self_agreement"]], ["a", "b"])
        self.assertTrue(
            all(c["argmax_agreement"] == 1.0 for c in record["self_agreement"])
        )
        self.assertEqual(record["errors"], 0)
        self.assertIs(record["passed"], True)
        self.assertEqual(record["first_raw_response"], response(GOOD))
        self.assertAlmostEqual(record["texts"][0]["top1"], 1 / 3)

    def test_repeats_that_differ_are_data_not_a_failure(self):
        # A served full model does not repeat itself exactly; the run still passes
        # and the difference stays in self_agreement for the reader.
        answers = [copy.deepcopy(GOOD), copy.deepcopy(GOOD)]
        second = answers[1][1]
        top, other = sorted(second, key=lambda t: -second[t]["logprob"])[:2]
        second[top]["logprob"], second[other]["logprob"] = (
            second[other]["logprob"],
            second[top]["logprob"],
        )
        replies = iter(answers)
        record = agreement.run(
            lambda text: list(TOKENS),
            lambda ids, k: response(next(replies)),
            texts={"a": "alpha"},
            repeats=2,
        )
        self.assertLess(record["self_agreement"][0]["argmax_agreement"], 1.0)
        self.assertIs(record["passed"], True)

    def test_a_failed_request_is_recorded_and_the_run_does_not_pass(self):
        tokenize, complete, _ = self.senders(fail_on=2)
        record = agreement.run(tokenize, complete, texts={"a": "alpha"}, repeats=2)
        self.assertEqual(record["errors"], 1)
        self.assertEqual(record["texts"][1]["error"], "TimeoutError: slow")
        self.assertEqual(record["self_agreement"], [])
        self.assertIs(record["passed"], False)

    def test_bundled_texts_are_distinct_non_empty_and_bounded(self):
        self.assertEqual(set(agreement.TEXTS), {"ja-prose", "en-prose", "code", "math"})
        for name, text in agreement.TEXTS.items():
            with self.subTest(name=name):
                self.assertGreater(len(text), 200)
                # Well under the 4,608-token block: one text never registers a
                # prefix-cache block on the serving pair.
                self.assertLess(len(text), 4000)


class CompareRecordsTests(unittest.TestCase):
    def record(self, prompt_logprobs=GOOD, names=("a", "b"), tokens=TOKENS):
        return {
            "texts": [
                agreement.score_text(
                    name, list(tokens), response(copy.deepcopy(prompt_logprobs))
                )
                for name in names
            ]
        }

    def test_same_record_compares_to_full_agreement(self):
        reference = self.record()
        summary = agreement.compare_records(reference, copy.deepcopy(reference))
        self.assertEqual(summary["argmax_agreement"], 1.0)
        self.assertEqual(summary["max_abs_logprob_move"], 0.0)
        self.assertEqual([row["name"] for row in summary["texts"]], ["a", "b"])
        self.assertEqual(summary["missing_in_candidate"], [])

    def test_reference_read_back_from_json_still_agrees_fully(self):
        # On the host the reference arrives through read_json: rows become lists.
        reference = json.loads(json.dumps(self.record()))
        summary = agreement.compare_records(reference, self.record())
        self.assertEqual(summary["argmax_agreement"], 1.0)
        self.assertEqual(summary["max_abs_logprob_move"], 0.0)
        self.assertTrue(all(row["top5_overlap_min"] == 1.0 for row in summary["texts"]))

    def test_tokenization_change_and_missing_texts_are_named_not_scored(self):
        reference = self.record(names=("a", "b"))
        # The first token has no prediction, so changing it only alters tokenization.
        candidate = self.record(names=("a", "c"), tokens=[12, 22, 33, 44])
        summary = agreement.compare_records(reference, candidate)
        self.assertEqual(
            summary["texts"], [{"name": "a", "error": "tokenization differs"}]
        )
        self.assertEqual(summary["missing_in_candidate"], ["b"])
        self.assertEqual(summary["missing_in_reference"], ["c"])
        self.assertNotIn("argmax_agreement", summary)

    def test_errored_texts_are_skipped(self):
        reference = self.record(names=("a",))
        candidate = {"texts": [{"name": "a", "error": "TimeoutError: slow"}]}
        summary = agreement.compare_records(reference, candidate)
        self.assertEqual(summary["texts"], [])
        self.assertEqual(summary["missing_in_candidate"], ["a"])

    def test_weighted_summary_uses_positions(self):
        reference = self.record(names=("a",))
        changed = copy.deepcopy(reference)
        changed["texts"][0]["rows"][0] = [(99, -0.1), (22, -0.2)]
        summary = agreement.compare_records(reference, changed)
        self.assertTrue(math.isclose(summary["argmax_agreement"], 2 / 3))
        self.assertEqual(summary["texts"][0]["first_argmax_divergence"], 1)


if __name__ == "__main__":
    unittest.main()


class ServerSenderTests(unittest.TestCase):
    def profile(self, lpa=False, apc=True):
        return {
            "api": {"served_model_name": "glm"},
            "runtime": {"seed": 7},
            "lpa": {"enabled": lpa},
            "cache": {"prefix_caching": apc},
        }

    def test_senders_tokenize_text_then_score_the_same_token_ids(self):
        from glm53_setup import server

        posts = []

        def sender(profile, path, body):
            posts.append((path, body))
            return {"tokens": [1, 2, 3]} if path == "/tokenize" else response(GOOD)

        tokenize, complete = server.agreement_senders(self.profile(), sender)
        self.assertEqual(tokenize("text"), [1, 2, 3])
        complete([1, 2, 3], 5)
        self.assertEqual(posts[0], ("/tokenize", {"model": "glm", "prompt": "text"}))
        self.assertEqual(
            posts[1],
            (
                "/v1/completions",
                {
                    "model": "glm",
                    "prompt": [1, 2, 3],
                    "max_tokens": 1,
                    "temperature": 0,
                    "seed": 7,
                    "prompt_logprobs": 5,
                },
            ),
        )

    def test_native_lpa_is_refused_and_apc_first_lpa_is_allowed(self):
        from glm53_setup import server

        with self.assertRaises(ValueError):
            server.agreement_senders(self.profile(lpa=True, apc=False), lambda *a: None)
        server.agreement_senders(self.profile(lpa=True, apc=True), lambda *a: None)
