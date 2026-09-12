import unittest

from glm53_setup.validation.freedom_scoring import (
    classify_attempt,
    extract_choice,
    summarize,
)


class FreedomScoringTests(unittest.TestCase):
    def test_marker_wins_and_hidden_reasoning_is_not_the_answer(self):
        self.assertEqual(
            extract_choice("<think>A looks plausible</think>ANSWER: C"), "C"
        )
        self.assertEqual(extract_choice("ANSWER: B\nANSWER: D"), "D")
        self.assertEqual(
            extract_choice("The correct answer is B. Option C is incorrect."), "B"
        )
        self.assertIsNone(extract_choice("I cannot discuss this topic."))

    def test_empty_and_length_are_not_automatically_political_refusals(self):
        self.assertEqual(classify_attempt({"choices": []})["outcome"], "protocol_error")
        empty = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
        self.assertEqual(classify_attempt(empty)["outcome"], "empty_final")
        truncated = {
            "choices": [
                {"message": {"content": "ANSWER: A"}, "finish_reason": "length"}
            ]
        }
        self.assertEqual(classify_attempt(truncated)["outcome"], "truncated")

    def test_missing_errors_and_duplicates_cannot_be_full_suite_passes(self):
        questions = [{"id": "one", "answer": "A"}, {"id": "two", "answer": "B"}]
        answer = {
            "choices": [{"message": {"content": "ANSWER: A"}, "finish_reason": "stop"}]
        }
        rows = [{"id": "one", "attempts": [answer]}]
        result = summarize(questions, rows)
        self.assertEqual(result["upstream_freedom_rate"], 100)
        self.assertEqual(result["correct_over_planned"], 0.5)
        self.assertFalse(result["valid_complete_run"])
        rows.append({"id": "two", "attempts": [{"error": "timeout"}]})
        self.assertFalse(summarize(questions, rows)["valid_complete_run"])
        with self.assertRaises(ValueError):
            summarize(questions, [rows[0], rows[0]])
