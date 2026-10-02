import threading
import unittest

from glm53_setup import prefix_gate

SEED = 7
TOKENS_PER_LINE = 33  # The fake tokenizer: a fixed cost per prompt line.


def count_tokens(request):
    return TOKENS_PER_LINE * request["messages"][0]["content"].count("\n") + 40


def split(request):
    """The prefix the fake server was sent, and the task asked of it."""
    text, question = request["messages"][0]["content"].rsplit("\n\n", 1)
    prefix = prefix_gate.build_prefix(text.count("\n") + 1, SEED)
    task = next(t for t in prefix_gate.tasks(prefix) if t["question"] == question)
    return prefix, task


def response(content, cached=1000, prompt=2000, details=True):
    usage = {"prompt_tokens": prompt}
    if details:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage,
    }


class FakeServer:
    """Answers every task correctly unless told otherwise; counts calls per salt."""

    def __init__(self, parallel, warm=None, cold_cached=0, warm_cached=1000):
        self.parallel, self.warm = parallel, warm
        self.cold_cached, self.warm_cached = cold_cached, warm_cached
        self.lock, self.calls, self.requests = threading.Lock(), 0, []

    def __call__(self, request):
        with self.lock:
            self.calls += 1
            phase = "cold" if self.calls <= self.parallel else "warm"
            self.requests.append(request)
        prefix, task = split(request)
        content = prefix_gate.answer(prefix, task)
        if phase == "warm" and self.warm is not None:
            content = self.warm(prefix, task, content)
        cached = self.cold_cached if phase == "cold" else self.warm_cached
        return response(content, cached=cached)


def run(server, limit=6000):
    return prefix_gate.run(
        server, count_tokens, max_prompt_tokens=limit, salt="s1", seed=SEED
    )


class PrefixTests(unittest.TestCase):
    def test_the_prefix_is_reproduced_by_its_seed_and_answers_are_unique(self):
        first = prefix_gate.build_prefix(400, SEED)
        self.assertEqual(first.text, prefix_gate.build_prefix(400, SEED).text)
        self.assertNotEqual(first.text, prefix_gate.build_prefix(400, SEED + 1).text)
        self.assertEqual(len(first.lines), 400)
        # Each record number appears on exactly one line, so a lookup has one answer.
        numbers = [line.split()[1] for line in first.lines]
        self.assertEqual(len(set(numbers)), 400)
        for task in prefix_gate.tasks(first):
            with self.subTest(task=task["question"][:40]):
                checked = prefix_gate.verify(
                    first, task, prefix_gate.answer(first, task)
                )
                self.assertEqual(checked, {"correct": True, "mismatches": []})
                for target in task["targets"]:
                    self.assertIn(first.codes[target], first.lines[target])

    def test_a_wrong_code_and_a_line_absent_from_the_prefix_are_mismatches(self):
        prefix = prefix_gate.build_prefix(400, SEED)
        lookup, quote = (
            next(t for t in prefix_gate.tasks(prefix) if t["kind"] == kind)
            for kind in ("lookup", "quote")
        )
        a, b = lookup["targets"]
        wrong = "ZZZZZ" if prefix.codes[b] != "ZZZZZ" else "YYYYY"
        checked = prefix_gate.verify(
            prefix,
            lookup,
            f"Record {a:05d}: {prefix.codes[a]}\nRecord {b:05d}: {wrong}",
        )
        self.assertFalse(checked["correct"])
        self.assertEqual(
            checked["mismatches"],
            [f"Record {b:05d}: {wrong} (truth {prefix.codes[b]})"],
        )
        phantom = prefix.lines[quote["targets"][0]].replace("retries=", "retries=9")
        checked = prefix_gate.verify(
            prefix, quote, prefix_gate.answer(prefix, quote) + "\n" + phantom
        )
        self.assertEqual(checked, {"correct": False, "mismatches": [phantom]})
        self.assertFalse(prefix_gate.verify(prefix, quote, "")["correct"])

    def test_a_cited_record_beyond_the_prefix_is_a_mismatch(self):
        prefix = prefix_gate.build_prefix(400, SEED)
        lookup = next(t for t in prefix_gate.tasks(prefix) if t["kind"] == "lookup")
        checked = prefix_gate.verify(
            prefix,
            lookup,
            prefix_gate.answer(prefix, lookup) + "\nRecord 00400: ABCDE",
        )
        self.assertEqual(
            checked,
            {"correct": False, "mismatches": ["Record 00400: ABCDE (no such record)"]},
        )

    def test_the_prefix_is_sized_to_the_token_limit_by_the_given_counter(self):
        prefix, used = prefix_gate.fit_prefix(count_tokens, 6000, SEED)
        self.assertLessEqual(used, 6000)
        self.assertGreater(used, 6000 - 2 * TOKENS_PER_LINE)
        self.assertEqual(
            used,
            max(
                count_tokens(prefix_gate.request(prefix, task, "s"))
                for task in prefix_gate.tasks(prefix)
            ),
        )
        with self.assertRaisesRegex(ValueError, "too short"):
            prefix_gate.fit_prefix(count_tokens, 500, SEED)

    def test_lengths_are_the_long_gate_and_the_short_trial(self):
        self.assertEqual(prefix_gate.LENGTHS, {"long": 99_000, "short": 14_025})


class GateTests(unittest.TestCase):
    def test_all_correct_with_warm_hits_passes_and_records_both_phases(self):
        server = FakeServer(parallel=6)
        result = run(server)
        self.assertEqual(result["verdict"], "pass")
        self.assertIs(result["passed"], True)
        self.assertEqual(result["reasons"], [])
        self.assertEqual(len(server.requests), 12)
        self.assertTrue(all(r["cache_salt"] == "s1" for r in server.requests))
        self.assertTrue(all(r["temperature"] == 0 for r in server.requests))
        for phase in ("cold", "warm"):
            self.assertEqual(result[phase]["accuracy"], 1.0)
            self.assertEqual(result[phase]["failures"], [])
        self.assertEqual(result["warm"]["cached_tokens"], [1000] * 6)
        self.assertEqual(result["cold"]["cached_tokens"], [0] * 6)
        self.assertEqual(result["prefix"]["seed"], SEED)
        self.assertEqual(result["cache_salt"], "s1")

    def test_one_phantom_line_in_one_warm_answer_is_prefix_cache_corruption(self):
        def phantom(prefix, task, content):
            if task["kind"] != "quote" or task["targets"][0] != len(prefix.lines) - 3:
                return content
            return content.replace("retries=", "retries=7", 1)

        result = run(FakeServer(parallel=6, warm=phantom))
        self.assertEqual(result["verdict"], "fail")
        self.assertIs(result["passed"], False)
        self.assertEqual(result["reasons"], ["prefix_cache_corruption"])
        self.assertEqual(result["cold"]["accuracy"], 1.0)
        self.assertAlmostEqual(result["warm"]["accuracy"], 5 / 6)
        [failure] = result["warm"]["failures"]
        self.assertEqual(failure["kind"], "quote")
        self.assertIn("last three lines", failure["request"])
        self.assertIn("retries=7", failure["reply"])
        [line] = failure["mismatches"]
        self.assertIn("retries=7", line)

    def test_correct_answers_without_a_warm_hit_never_pass(self):
        for server in (
            FakeServer(parallel=6, warm_cached=0),
            FakeServer(parallel=6, warm_cached=None),
        ):
            with self.subTest(cached=server.warm_cached):
                result = run(server)
                self.assertEqual(result["verdict"], "inconclusive")
                self.assertIs(result["passed"], False)
                self.assertEqual(result["reasons"], ["no_cache_hit"])

        def no_details(request):
            prefix, task = split(request)
            return response(prefix_gate.answer(prefix, task), details=False)

        result = run(no_details)
        self.assertEqual(result["reasons"], ["no_cache_hit"])
        self.assertEqual(result["warm"]["cached_tokens"], [None] * 6)

    def test_a_cold_misreading_or_a_failed_request_is_inconclusive(self):
        calls = []

        def cold_wrong(request):
            calls.append(request)
            prefix, task = split(request)
            content = prefix_gate.answer(prefix, task)
            if len(calls) == 1:
                content = "NONE"
            return response(content)

        result = prefix_gate.run(
            cold_wrong,
            count_tokens,
            max_prompt_tokens=6000,
            salt="s",
            seed=SEED,
            parallel=1,
        )
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(result["reasons"], ["cold_incorrect"])

        def broken(request):
            raise TimeoutError

        result = prefix_gate.run(
            broken, count_tokens, max_prompt_tokens=6000, salt="s", seed=SEED
        )
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertIn("request_error", result["reasons"])
        self.assertEqual(result["warm"]["failures"][0]["error"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
