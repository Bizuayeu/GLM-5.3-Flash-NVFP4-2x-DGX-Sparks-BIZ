import unittest

from glm53_setup import mojibake

JAPANESE = "日本の春は桜が咲き、夏は祭りでにぎわう。" * 30
KOREAN = "한국의 봄에는 벚꽃이 피고 여름에는 축제가 열립니다. " * 20


def response(content, reasoning=None, finish="stop"):
    return {
        "choices": [
            {
                "message": {"content": content, "reasoning": reasoning},
                "finish_reason": finish,
            }
        ]
    }


class MojibakeTests(unittest.TestCase):
    def test_counts_each_invalid_kind_and_keeps_line_breaks_and_tabs(self):
        text = "a\N{REPLACEMENT CHARACTER}b\ud800c\x00d\x1be\n\r\tf"
        self.assertEqual(
            mojibake.count_invalid_text(text),
            {"replacement": 1, "lone_surrogate": 1, "control": 2},
        )
        self.assertEqual(
            mojibake.count_invalid_text("正常な文章"),
            {"replacement": 0, "lone_surrogate": 0, "control": 0},
        )

    def test_target_script_share_ignores_whitespace(self):
        self.assertGreater(mojibake.script_share(KOREAN, "ko"), 0.5)
        self.assertGreater(mojibake.script_share(JAPANESE, "ja"), 0.5)
        self.assertLess(
            mojibake.script_share("Spring brings cherry blossoms.", "ja"), 0.5
        )
        self.assertEqual(mojibake.script_share(" \n", "ko"), 0.0)

    def judge(self, language, result):
        return mojibake.judge(language, result)["verdict"]

    def test_long_target_language_without_invalid_characters_passes(self):
        self.assertEqual(self.judge("ja", response(JAPANESE)), "pass")
        self.assertEqual(self.judge("ko", response(KOREAN)), "pass")

    def test_one_invalid_character_fails_in_content_or_reasoning(self):
        self.assertEqual(
            self.judge("ko", response(KOREAN + "\N{REPLACEMENT CHARACTER}")), "fail"
        )
        self.assertEqual(
            self.judge("ja", response(JAPANESE, "考え\N{REPLACEMENT CHARACTER}")),
            "fail",
        )
        # A short answer that still carries a broken character is evidence.
        self.assertEqual(
            self.judge("ja", response("短い\N{REPLACEMENT CHARACTER}")), "fail"
        )

    def test_errors_empty_short_or_wrong_language_never_pass(self):
        for result in (
            RuntimeError("HTTP 500"),
            response(None),
            response(""),
            response(JAPANESE[:399]),
            response("Spring brings cherry blossoms. " * 20),
        ):
            with self.subTest(result=repr(result)[:40]):
                self.assertEqual(self.judge("ja", result), "inconclusive")

    def test_truncated_answer_is_counted_and_marked(self):
        row = mojibake.judge("ko", response(KOREAN, finish="length"))
        self.assertEqual(row["verdict"], "pass")
        self.assertIs(row["truncated"], True)

    def test_run_repeats_each_language_at_temperature_zero_and_aggregates(self):
        requests = []

        def ask(request):
            requests.append(request)
            language = "ja" if "日本" in request["messages"][0]["content"] else "ko"
            if language == "ko" and len(requests) == 5:
                return response(KOREAN + "\N{REPLACEMENT CHARACTER}")
            return response(JAPANESE if language == "ja" else KOREAN)

        record = mojibake.run(ask, repeats=3)
        self.assertEqual(len(requests), 6)
        self.assertTrue(all(r["temperature"] == 0 for r in requests))
        self.assertTrue(all("max_tokens" not in r for r in requests))
        self.assertEqual(
            [row["language"] for row in record["runs"]], ["ja"] * 3 + ["ko"] * 3
        )
        self.assertEqual(record["verdict"], "fail")
        self.assertIs(record["passed"], False)

        def clean(request):
            return response(
                JAPANESE if "日本" in request["messages"][0]["content"] else KOREAN
            )

        self.assertEqual(mojibake.run(clean, repeats=1)["verdict"], "pass")

        def flaky(request):
            raise TimeoutError

        record = mojibake.run(flaky, repeats=1)
        self.assertEqual(record["verdict"], "inconclusive")
        self.assertIs(record["passed"], False)
        self.assertEqual(record["runs"][0]["error"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
