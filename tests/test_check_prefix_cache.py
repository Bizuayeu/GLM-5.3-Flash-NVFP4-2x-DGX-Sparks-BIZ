import contextlib
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from tools import check_prefix_cache


def usage(cached=None, details=True):
    call = {"prompt_tokens": 100}
    if details:
        call["prompt_tokens_details"] = {"cached_tokens": cached}
    return call


class CheckPrefixCacheMainTests(unittest.TestCase):
    def run_main(self, side_effect, *extra):
        out, err = io.StringIO(), io.StringIO()
        with (
            patch.object(check_prefix_cache, "ask", side_effect=side_effect) as ask,
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = check_prefix_cache.main(["--model", "m", "--lines", "2", *extra])
        return code, out.getvalue(), err.getvalue(), ask

    def test_a_restored_repeat_exits_zero_with_the_exact_opt_out(self):
        code, out, err, ask = self.run_main([usage(0), usage(64)])
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        result = json.loads(out)
        self.assertEqual(result["cached_tokens"], [0, 64])
        self.assertTrue(result["reused_on_repeat"])
        self.assertEqual(ask.call_count, 2)
        for call in ask.call_args_list:
            self.assertIs(call.args[4], True)

    def test_approximate_sends_both_calls_without_the_opt_out(self):
        code, _, _, ask = self.run_main([usage(0), usage(64)], "--approximate")
        self.assertEqual(code, 0)
        for call in ask.call_args_list:
            self.assertIs(call.args[4], False)

    def test_missing_details_exits_one(self):
        code, out, err, _ = self.run_main([usage(details=False), usage(details=False)])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["reports_prompt_tokens_details"])
        self.assertIn("prompt_tokens_details", err)

    def test_zero_or_absent_cached_tokens_exit_one(self):
        for cached in (0, None):
            with self.subTest(cached=cached):
                code, out, err, _ = self.run_main([usage(0), usage(cached)])
                self.assertEqual(code, 1)
                self.assertFalse(json.loads(out)["reused_on_repeat"])
                self.assertIn("Nothing was restored", err)

    def test_an_unreachable_server_exits_two_without_a_result(self):
        for error in (urllib.error.URLError("refused"), TimeoutError()):
            with self.subTest(error=type(error).__name__):
                code, out, err, _ = self.run_main(error)
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn("request failed", err)


if __name__ == "__main__":
    unittest.main()
