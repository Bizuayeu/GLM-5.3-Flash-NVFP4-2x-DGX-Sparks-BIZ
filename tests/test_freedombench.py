import contextlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup.validation import freedombench
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


class RunnerFailureTests(unittest.TestCase):
    def test_a_request_that_raises_leaves_a_failed_record(self):
        questions = [{"id": "one", "prompt": "?", "answer": "A"}]

        def ask(profile, body):
            raise RuntimeError("server went away")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            server = freedombench.server
            with (
                patch.object(freedombench, "os", types.SimpleNamespace(name="posix")),
                patch.object(freedombench.server_config, "load", return_value={}),
                patch.object(
                    freedombench, "load_questions", return_value=({}, questions)
                ),
                patch.object(freedombench, "load_lock", return_value={}),
                patch.object(
                    server,
                    "running_head",
                    return_value=({}, {"State": {"Running": True}, "Image": "i"}),
                ),
                patch.object(server, "request_lock", contextlib.nullcontext),
                patch.object(server, "ask", side_effect=ask),
                self.assertRaises(RuntimeError),
            ):
                freedombench.main(
                    ["--benchmark-dir", tmp, "--config", "c", "--output", str(output)]
                )
            record = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "RuntimeError('server went away')")
        self.assertEqual(record["summary"]["received"], 1)
        self.assertFalse(record["full_suite_complete"])
