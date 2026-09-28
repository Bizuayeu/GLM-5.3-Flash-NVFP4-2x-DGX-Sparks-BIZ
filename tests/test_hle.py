import contextlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup.validation import hle
from glm53_setup.validation.hle_scoring import exact, extract, normalize


def reply(content, finish="stop"):
    return {
        "choices": [
            {
                "message": {"content": content, "reasoning_content": "thinking"},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


ROWS = [
    {"id": "t1", "question": "Q1?", "image": "", "category": "Math"},
    {
        "id": "i1",
        "question": "Q2?",
        "image": "data:image/png;base64,AAAA",
        "category": "Chemistry",
    },
]


class ScoringTests(unittest.TestCase):
    def test_answer_and_confidence_come_from_the_final_content(self):
        text = "Explanation: because\nExact Answer: **42.**\nConfidence: 80%"
        self.assertEqual(extract(text), ("**42.**", 80.0))
        self.assertEqual(
            extract("**Exact Answer:** x\n**Confidence:** 55%"), ("x", 55.0)
        )

    def test_missing_answer_never_matches_an_empty_reference(self):
        self.assertEqual(extract("I think it is 3."), (None, None))
        self.assertEqual(extract(None), (None, None))
        self.assertIsNone(exact(None, ""))

    def test_normalization_matches_the_lineage_rule(self):
        self.assertEqual(normalize(" Ｆｅ. "), "fe")
        self.assertTrue(exact("Paris.", "paris"))
        self.assertFalse(exact("Lyon", "Paris"))

    def test_out_of_range_confidence_is_dropped(self):
        self.assertEqual(extract("Exact Answer: 1\nConfidence: 150%"), ("1", None))


class BudgetTests(unittest.TestCase):
    def test_the_answer_budget_is_freedombenchs(self):
        from glm53_setup.validation import freedombench

        self.assertEqual(hle.MAX_TOKENS, freedombench.REASONING_BUDGET)


class MessageTests(unittest.TestCase):
    def test_text_and_image_questions(self):
        self.assertEqual(hle.messages(ROWS[0])[1]["content"], "Q1?")
        parts = hle.messages(ROWS[1])[1]["content"]
        self.assertEqual(parts[0]["image_url"]["url"], ROWS[1]["image"])
        self.assertEqual(parts[1], {"type": "text", "text": "Q2?"})
        with self.assertRaises(ValueError):
            hle.messages(dict(ROWS[1], image="http://example.com/a.png"))

    def test_question_file_must_not_carry_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.jsonl"
            path.write_text(
                json.dumps(dict(ROWS[0], answer="x")) + "\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                hle.load_questions(path)


class RunnerTests(unittest.TestCase):
    def run_main(self, tmp, ask, extra=()):
        questions = Path(tmp) / "q.jsonl"
        questions.write_text(
            "".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8"
        )
        output = Path(tmp) / "out"
        server = hle.server
        with (
            patch.object(hle, "os", types.SimpleNamespace(name="posix")),
            patch.object(hle.server_config, "load", return_value={}),
            patch.object(hle.server_config, "fingerprint", return_value="f"),
            patch.object(hle, "load_lock", return_value={}),
            patch.object(
                server,
                "running_head",
                return_value=({}, {"State": {"Running": True}, "Image": "img"}),
            ),
            patch.object(server, "request_lock", contextlib.nullcontext),
            patch.object(server, "ask", side_effect=ask),
        ):
            hle.main(
                [
                    "--questions",
                    str(questions),
                    "--config",
                    "c",
                    "--output",
                    str(output),
                    "--label",
                    "axl",
                    *extra,
                ]
            )
        return output

    def test_answers_are_saved_per_question_and_marked_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_main(
                tmp, lambda p, b: reply("Exact Answer: 7\nConfidence: 50%")
            )
            saved = json.loads(
                (out / "answers" / "i1.json").read_text(encoding="utf-8")
            )
            status = json.loads((out / "status.json").read_text(encoding="utf-8"))
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["answer"], "7")
        self.assertTrue(saved["has_image"])
        self.assertEqual(saved["reasoning"], "thinking")
        self.assertTrue(saved["teacher_excluded"])
        self.assertTrue(manifest["teacher_excluded"])
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["summary"]["parsed"], 2)

    def test_resume_skips_answered_questions(self):
        calls = []

        def ask(profile, body):
            calls.append(body)
            if len(calls) == 2:
                raise ConnectionError("gone")
            return reply("Exact Answer: 1\nConfidence: 10%")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConnectionError):
                self.run_main(tmp, ask)
            out = Path(tmp) / "out"
            self.assertEqual(
                json.loads((out / "status.json").read_text(encoding="utf-8"))["status"],
                "failed",
            )
            self.assertEqual(len(list((out / "errors").iterdir())), 1)
            self.run_main(tmp, ask)
            status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(len(calls), 3)  # t1, failed i1, retried i1 only
        self.assertEqual(status["status"], "complete")

    def test_stop_file_halts_between_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "STOP").write_text("", encoding="utf-8")
            self.run_main(tmp, lambda p, b: self.fail("no request after STOP"))
            status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "stopped")
        self.assertEqual(status["summary"]["answered"], 0)

    def test_a_different_run_cannot_reuse_the_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_main(tmp, lambda p, b: reply("Exact Answer: 1\nConfidence: 10%"))
            with self.assertRaises(SystemExit):
                self.run_main(
                    tmp, lambda p, b: reply("x"), extra=["--max-tokens", "100"]
                )

    def test_distribution_sampling_and_a_longer_client_timeout_reach_the_request(self):
        # J2 case c (2026-09-28): the model card's sampling, 16,384 tokens, and a
        # timeout that covers 16,384 at the slowest measured decode (20.67 tok/s).
        seen = []

        def ask(profile, body):
            seen.append((profile, body))
            return reply("Exact Answer: 1\nConfidence: 10%")

        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_main(
                tmp,
                ask,
                extra=[
                    "--max-tokens",
                    "16384",
                    "--temperature",
                    "1.0",
                    "--top-p",
                    "0.95",
                    "--timeout",
                    "1200",
                ],
            )
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        profile, body = seen[0]
        self.assertEqual(body["temperature"], 1.0)
        self.assertEqual(body["top_p"], 0.95)
        self.assertEqual(body["max_tokens"], 16384)
        self.assertEqual(profile["generation"]["timeout_seconds"], 1200)
        self.assertEqual(manifest["sampling"], {"temperature": 1.0, "top_p": 0.95})
        self.assertEqual(manifest["timeout_seconds"], 1200)

    def test_profile_sampling_is_left_alone_unless_asked(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            self.run_main(
                tmp,
                lambda p, b: seen.append(b) or reply("Exact Answer: 1\nConfidence: 1%"),
            )
        self.assertNotIn("temperature", seen[0])
        self.assertNotIn("top_p", seen[0])

    def test_max_new_pauses_so_a_driver_can_cool_the_hosts_between_questions(self):
        calls = []

        def ask(profile, body):
            calls.append(body)
            return reply("Exact Answer: 1\nConfidence: 10%")

        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_main(tmp, ask, extra=["--max-new", "1"])
            first = json.loads((out / "status.json").read_text(encoding="utf-8"))
            self.run_main(tmp, ask, extra=["--max-new", "1"])
            second = json.loads((out / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(first["status"], "paused")
        self.assertEqual(first["summary"]["answered"], 1)
        self.assertEqual(second["status"], "complete")
        self.assertEqual(len(calls), 2)

    def test_truncation_is_counted_not_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_main(tmp, lambda p, b: reply(None, finish="length"))
            status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["summary"]["truncated"], 2)
        self.assertEqual(status["summary"]["parsed"], 0)


if __name__ == "__main__":
    unittest.main()
