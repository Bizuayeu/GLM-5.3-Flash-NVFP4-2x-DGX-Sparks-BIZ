import contextlib
import hashlib
import io
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


def _case(qid, case_type, arm, position, answer):
    return {
        "id": qid,
        "topic": "t",
        "kind": "benchmark",
        "case_type": case_type,
        "arm": arm,
        "position": position,
        "language": "en",
        "system_prompt": freedombench.SYSTEM_PROMPT if case_type == "M" else "Review.",
        "user": f"document for {qid}\n\nquestion",
        "answer": answer,
        "document_sha256": "d",
        "designated_distractor": None,
    }


def _case_file(tmp, cases, conditions=None, excluded=True):
    if conditions is None:
        conditions = {}
        for c in cases:
            key = freedombench.condition_key(c)
            conditions.setdefault(key, []).append(c["id"])
    doc = {
        "teacher_excluded": excluded,
        "case": "test",
        "benchmark": {"revision": "r"},
        "conditions": conditions,
        "cases": cases,
    }
    path = Path(tmp) / "cases.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class CaseFileTests(unittest.TestCase):
    cases = [
        _case("one", "M", "S", "middle", "B"),
        _case("ctrl", "M", "C", "middle", "A"),
        _case("one", "M", "K", "middle", "B"),
        _case("one", "E", "S", "middle", None),
    ]

    def test_a_condition_selects_its_cases_and_the_counter_arm_joins_s(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, digest = _case_file(tmp, self.cases)
            meta, questions = freedombench.load_cases(path, digest, "M-S-middle-en")
        self.assertEqual([q["id"] for q in questions], ["one", "ctrl"])
        self.assertEqual([q["arm"] for q in questions], ["S", "C"])
        self.assertEqual(questions[0]["prompt"], "document for one\n\nquestion")
        self.assertEqual(questions[0]["system_prompt"], freedombench.SYSTEM_PROMPT)
        self.assertEqual(meta["condition"], "M-S-middle-en")
        self.assertEqual(meta["sha256"], digest)

    def test_hash_list_and_prompt_mismatches_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, digest = _case_file(tmp, self.cases)
            with self.assertRaises(ValueError):
                freedombench.load_cases(path, "0" * 64, "M-S-middle-en")
            with self.assertRaises(ValueError):
                freedombench.load_cases(path, digest, "M-N-front-en")
            path, digest = _case_file(
                tmp, self.cases, conditions={"M-S-middle-en": ["ctrl", "one"]}
            )
            with self.assertRaises(ValueError):
                freedombench.load_cases(path, digest, "M-S-middle-en")
            other = [dict(self.cases[0], system_prompt="Something else.")]
            path, digest = _case_file(tmp, other)
            with self.assertRaises(ValueError):
                freedombench.load_cases(path, digest, "M-S-middle-en")
            path, digest = _case_file(tmp, self.cases, excluded=False)
            with self.assertRaises(ValueError):
                freedombench.load_cases(path, digest, "M-S-middle-en")

    def test_a_type_e_case_is_asked_once_with_its_own_prompts(self):
        sent = []
        reply = {
            "choices": [
                {"message": {"content": "1. 7Q-214, 17."}, "finish_reason": "stop"}
            ]
        }

        def ask(profile, body):
            sent.append(body)
            return dict(reply)

        with tempfile.TemporaryDirectory() as tmp:
            path, digest = _case_file(tmp, self.cases)
            output = Path(tmp) / "out"
            server = freedombench.server
            with (
                patch.object(freedombench, "os", types.SimpleNamespace(name="posix")),
                patch.object(freedombench.server_config, "load", return_value={}),
                patch.object(freedombench, "load_lock", return_value={}),
                patch.object(
                    server,
                    "running_head",
                    return_value=({}, {"State": {"Running": True}, "Image": "i"}),
                ),
                patch.object(server, "request_lock", contextlib.nullcontext),
                patch.object(server, "ask", side_effect=ask),
            ):
                freedombench.main(
                    [
                        "--cases",
                        str(path),
                        "--cases-sha256",
                        digest,
                        "--condition",
                        "E-S-middle-en",
                        "--config",
                        "c",
                        "--output",
                        str(output),
                    ]
                )
            record = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(len(sent), 1)
        self.assertEqual(
            sent[0]["messages"][0], {"role": "system", "content": "Review."}
        )
        self.assertEqual(record["condition"], "E-S-middle-en")
        self.assertEqual(record["language"], "en")
        self.assertTrue(record["teacher_excluded"])
        self.assertTrue(record["pilot"])
        self.assertFalse(record["full_suite_complete"])
        self.assertEqual(record["summary"]["answered"], 1)
        self.assertTrue(record["summary"]["valid_complete_run"])

    def test_the_pinned_suite_and_a_case_file_are_exclusive(self):
        with self.assertRaises(SystemExit):
            freedombench.main(["--benchmark-dir", "b", "--cases", "c", "--output", "o"])


_UPSTREAM = [
    ("alpha", "t1", "First question?", "right", ("w1", "w2", "w3"), "s1"),
    ("zeta", "t2", "Second question?", "yes", ("no1", "no2", "no3"), "s2"),
    ("eta", "t3", "Third question?", "true", ("f1", "f2", "f3"), "s3"),
]
_JA_PROMPT = "日本語の小テストです。最後は次の形式の行で終えてください。\nANSWER: X"
_JA_LINE = "一文字で答えてください。"
# Built by the pre-translation runner; the refactor must keep them.
GOLDEN_ANSWERS = ["D", "B", "A"]
GOLDEN_ALPHA = (
    "First question?\n\nA) w3\nB) w2\nC) w1\nD) right\n\n"
    "Answer with a single letter (A, B, C, or D)."
)


def _item(qid, topic, question, correct, distractors, source):
    return {
        "id": qid,
        "topic": topic,
        "source": source,
        "en": {"question": question, "correct": correct, "distractors": [*distractors]},
        "ja": {
            "question": f"ja:{question}",
            "correct": f"ja:{correct}",
            "distractors": [f"ja:{d}" for d in distractors],
        },
    }


def _suite(tmp, edit=None, version="ja-1"):
    """A pinned questions.py, a translation and both locks, with tmp as ROOT."""
    root = Path(tmp)
    (root / "freedombench").mkdir()
    (root / "config").mkdir()
    calls = "".join(f"    q{item!r},\n" for item in _UPSTREAM)
    source = f"QUESTIONS: tuple[Question, ...] = (\n{calls})\n"
    (root / "freedombench/questions.py").write_bytes(source.encode())
    lock = {
        "repository": "r",
        "revision": "v",
        "questions_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    (root / "config/freedombench.lock.json").write_text(json.dumps(lock), "utf-8")
    doc = {
        "teacher_excluded": True,
        "translation_version": "ja-1",
        "system_prompt": _JA_PROMPT,
        "instruction": _JA_LINE,
        "questions": [_item(*item) for item in _UPSTREAM],
    }
    if edit:
        edit(doc)
    path = root / "fb04.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    ja_lock = {
        **lock,
        "translation_version": version,
        "translation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "translation_path": "records/fb04.json",
    }
    (root / "config/freedombench-ja.lock.json").write_text(json.dumps(ja_lock), "utf-8")
    return root, path


class PinnedSuiteTests(unittest.TestCase):
    def test_the_english_prompts_are_the_upstream_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = _suite(tmp)
            with patch.object(freedombench, "ROOT", root):
                lock, questions = freedombench.load_questions(root)
        self.assertEqual(lock["revision"], "v")
        self.assertEqual([q["answer"] for q in questions], GOLDEN_ANSWERS)
        self.assertEqual(
            questions[0],
            {
                "id": "alpha",
                "topic": "t1",
                "prompt": GOLDEN_ALPHA,
                "answer": GOLDEN_ANSWERS[0],
                "source": "s1",
            },
        )

    def test_a_translation_keeps_the_english_order_and_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, path = _suite(tmp)
            with patch.object(freedombench, "ROOT", root):
                _, english = freedombench.load_questions(root)
                benchmark, japanese = freedombench.load_translation(root, path)
        self.assertEqual(benchmark["revision"], "v")
        self.assertEqual(benchmark["translation"]["translation_version"], "ja-1")
        for en, ja in zip(english, japanese, strict=True):
            self.assertEqual(
                (ja["id"], ja["topic"], ja["source"], ja["answer"]),
                (en["id"], en["topic"], en["source"], en["answer"]),
            )
            *body, line = ja["prompt"].split("\n")
            self.assertEqual(line, _JA_LINE)
            self.assertEqual(
                "\n".join(body).replace("ja:", ""),
                en["prompt"].rsplit("\n", 1)[0],
            )
            self.assertEqual(ja["system_prompt"], _JA_PROMPT)
            self.assertEqual(ja["language"], "ja-1")

    def test_a_translation_that_drifts_from_its_pins_is_refused(self):
        def drift(doc):
            doc["questions"][1]["en"]["question"] += "!"

        def two_distractors(doc):
            doc["questions"][0]["ja"]["distractors"].pop()

        def reorder(doc):
            doc["questions"][0:2] = doc["questions"][1::-1]

        cases = {
            "en drift": {"edit": drift},
            "distractor count": {"edit": two_distractors},
            "id order": {"edit": reorder},
            "version": {"version": "ja-2"},
        }
        for name, kwargs in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                root, path = _suite(tmp, **kwargs)
                with (
                    patch.object(freedombench, "ROOT", root),
                    self.assertRaises(ValueError),
                ):
                    freedombench.load_translation(root, path)
        with tempfile.TemporaryDirectory() as tmp:
            root, path = _suite(tmp)
            path.write_bytes(path.read_bytes() + b" ")
            with (
                patch.object(freedombench, "ROOT", root),
                self.assertRaisesRegex(ValueError, "hash"),
            ):
                freedombench.load_translation(root, path)

    def test_a_translation_needs_the_pinned_suite_and_runs_alone(self):
        for argv in (
            ["--translation", "t"],
            ["--translation", "t", "--cases", "c", "--cases-sha256", "h"],
            ["--translation", "t", "--benchmark-dir", "b", "--cases", "c"],
            ["--translation", "t", "--benchmark-dir", "b", "--limit", "2"],
        ):
            stderr = io.StringIO()
            with (
                self.subTest(argv),
                patch.object(freedombench, "os", types.SimpleNamespace(name="posix")),
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                freedombench.main([*argv, "--condition", "x", "--output", "o"])
            self.assertIn("--translation needs --benchmark-dir", stderr.getvalue())

    def test_a_translation_run_is_a_full_teacher_excluded_japanese_record(self):
        sent = []
        reply = {
            "choices": [{"message": {"content": "ANSWER: A"}, "finish_reason": "stop"}]
        }

        def ask(profile, body):
            sent.append(body)
            return dict(reply)

        with tempfile.TemporaryDirectory() as tmp:
            root, path = _suite(tmp)
            output = root / "out"
            server = freedombench.server
            with (
                patch.object(freedombench, "ROOT", root),
                patch.object(freedombench, "os", types.SimpleNamespace(name="posix")),
                patch.object(freedombench.server_config, "load", return_value={}),
                patch.object(freedombench, "load_lock", return_value={}),
                patch.object(
                    server,
                    "running_head",
                    return_value=({}, {"State": {"Running": True}, "Image": "i"}),
                ),
                patch.object(server, "request_lock", contextlib.nullcontext),
                patch.object(server, "ask", side_effect=ask),
            ):
                freedombench.main(
                    [
                        "--benchmark-dir",
                        tmp,
                        "--translation",
                        str(path),
                        "--config",
                        "c",
                        "--output",
                        str(output),
                    ]
                )
            record = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(len(sent), 3)
        self.assertEqual(
            sent[0]["messages"][0], {"role": "system", "content": _JA_PROMPT}
        )
        self.assertTrue(sent[0]["messages"][1]["content"].startswith("ja:"))
        self.assertEqual(record["language"], "ja-1")
        self.assertEqual(record["system_prompt"], _JA_PROMPT)
        self.assertEqual(
            record["benchmark"]["translation"]["translation_path"], "records/fb04.json"
        )
        self.assertTrue(record["teacher_excluded"])
        self.assertFalse(record["pilot"])
        self.assertTrue(record["full_suite_complete"])
