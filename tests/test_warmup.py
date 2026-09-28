import base64
import copy
import struct
import unittest
import zlib
from pathlib import Path

from glm53_setup import server_config as config
from glm53_setup import warmup

ROOT = Path(__file__).resolve().parents[1]

LOG_LINE = (
    "(Worker_TP0 pid=1) WARNING [jit_monitor.py:141] Triton kernel JIT compilation "
    "during inference: {name}. This causes a latency spike; consider extending warmup."
)


def fake_count(text):
    return len(text.split())


class WarmupTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(ROOT / "examples/server.example.toml")
        self.profile["generation"]["max_tokens"] = 4096

    def test_ladder_follows_profile_features(self):
        names = [name for name, _ in warmup.rungs(self.profile)]
        self.assertEqual(names, ["text", "tool", "image", "canary"])
        self.profile["runtime"]["vision"] = False
        self.profile["generation"]["warmup_long_tokens"] = 65536
        names = [name for name, _ in warmup.rungs(self.profile)]
        self.assertEqual(names, ["text", "tool", "long", "canary"])
        for name, request in warmup.rungs(self.profile):
            if request is not None and name != "canary":
                self.assertEqual(request["max_tokens"], warmup.ANSWER_TOKENS)
        canary = dict(warmup.rungs(self.profile))["canary"]
        self.assertEqual(canary["temperature"], 0)
        self.assertEqual(canary["reasoning_effort"], "low")
        self.assertEqual(canary["max_tokens"], warmup.CANARY_TOKENS)

    def test_png_is_a_valid_data_url(self):
        url = warmup.png_data_url(width=8, height=8)
        head, payload = url.split(",", 1)
        self.assertEqual(head, "data:image/png;base64")
        png = base64.b64decode(payload)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(png[12:16], b"IHDR")
        self.assertIn(b"IEND", png)
        idat = png.index(b"IDAT")
        length = int.from_bytes(png[idat - 4 : idat], "big")
        raw = zlib.decompress(png[idat + 4 : idat + 4 + length])
        self.assertEqual(len(raw), 8 * (1 + 8 * 3))
        head = base64.b64decode(warmup.png_data_url().split(",", 1)[1])[16:24]
        self.assertEqual(struct.unpack(">II", head), (672, 336))

    def test_long_prompt_targets_the_served_tokenizer_and_respects_the_limit(self):
        text, tokens = warmup.long_prompt(1000, 1200, fake_count)
        self.assertGreaterEqual(tokens, 990)
        self.assertLessEqual(tokens, 1200)

    def test_long_prompt_converges_when_a_line_costs_more_as_it_grows(self):
        # Mimics the real tokenizer: "warmup line 12345." costs more than
        # "warmup line 7." A single short sample under-counts, so the builder
        # must rescale against the measured count.
        def digit_cost(text):
            return sum(3 + len(line.split()[-1]) for line in text.splitlines())

        for target in (1000, 20000, 65536):
            _, tokens = warmup.long_prompt(target, 200000, digit_cost)
            self.assertLessEqual(abs(tokens - target), target * 0.05, target)
        # The limit still wins: asking for exactly the limit must not exceed it.
        _, tokens = warmup.long_prompt(5000, 5000, digit_cost)
        self.assertLessEqual(tokens, 5000)
        with self.assertRaises(ValueError):
            warmup.long_prompt(9000, 5000, digit_cost)
        # A tokenizer that counts more than the sample predicted is trimmed.
        text, tokens = warmup.long_prompt(
            1000, 1000, lambda t: fake_count(t) + (200 if len(t) > 4000 else 0)
        )
        self.assertLessEqual(tokens, 1000)
        with self.assertRaises(ValueError):
            warmup.long_prompt(0, 10, fake_count)
        with self.assertRaises(ValueError):
            warmup.long_prompt(2000, 1000, fake_count)

    def test_run_records_every_rung_and_the_kernels_compiled_meanwhile(self):
        self.profile["runtime"]["vision"] = False
        self.profile["generation"]["warmup_long_tokens"] = 300
        logs = [LOG_LINE.format(name="_unpack")]
        sent = []

        def ask(request):
            sent.append(request)
            if request.get("tools"):
                raise RuntimeError("tool parser down")
            if len(sent) == 3:
                logs.append(LOG_LINE.format(name="mhc_pre_big_fuse_with_norm_tilelang"))
            return {
                "usage": {"prompt_tokens": fake_count(str(request["messages"]))},
                "choices": [{"finish_reason": "stop", "message": {"content": COUNT}}],
            }

        clock = iter(range(0, 100, 1))
        resets = []
        record = warmup.run(
            self.profile,
            ask=ask,
            count_tokens=fake_count,
            logs=lambda: "\n".join(logs),
            clock=lambda: next(clock),
            reset=lambda: resets.append(True),
        )
        self.assertEqual(
            [r["rung"] for r in record["rungs"]], ["text", "tool", "long", "canary"]
        )
        self.assertEqual(
            [r["status"] for r in record["rungs"]], ["ok", "failed", "ok", "ok"]
        )
        self.assertEqual(record["rungs"][1]["error"], "RuntimeError")
        self.assertGreaterEqual(record["rungs"][2]["built_prompt_tokens"], 295)
        self.assertEqual(
            record["compiled_during_warmup"], ["mhc_pre_big_fuse_with_norm_tilelang"]
        )
        self.assertEqual(record["compiled_before_warmup"], ["_unpack"])
        self.assertTrue(record["prefix_cache_reset"])
        self.assertEqual(resets, [True])
        self.assertFalse(record["passed"])
        # Without a reset callable nothing is reset and nothing is claimed.
        plain = warmup.run(
            copy.deepcopy(self.profile),
            ask=ask,
            count_tokens=fake_count,
            logs=lambda: "",
            clock=lambda: 0,
        )
        self.assertFalse(plain["prefix_cache_reset"])


if __name__ == "__main__":
    unittest.main()


def answering(content, finish="stop"):
    def ask(request):
        return {
            "usage": {"prompt_tokens": 18},
            "choices": [{"finish_reason": finish, "message": {"content": content}}],
        }

    return ask


COUNT = " ".join(str(i) for i in range(1, warmup.CANARY_COUNT + 1))


class CanaryTests(unittest.TestCase):
    """Mia #268: a boot that answers /health but serves garbage must not pass."""

    def setUp(self):
        self.profile = config.load(ROOT / "examples/server.example.toml")
        self.profile["generation"]["max_tokens"] = 4096
        self.profile["runtime"]["vision"] = False
        self.profile["mtp"]["enabled"] = True

    def run_ladder(self, ask, counters=None):
        readings = iter(counters or [])
        return warmup.run(
            self.profile,
            ask=ask,
            count_tokens=fake_count,
            logs=lambda: "",
            clock=lambda: 0,
            spec_counters=(lambda: next(readings)) if counters else None,
        )

    def counters(self, drafted, accepted):
        return [
            {"num_draft_tokens": 100.0, "num_accepted_tokens": 70.0},
            {
                "num_draft_tokens": 100.0 + drafted,
                "num_accepted_tokens": 70.0 + accepted,
            },
        ]

    def test_the_canary_asks_for_a_count_long_enough_to_judge_the_drafts(self):
        canary = dict(warmup.rungs(self.profile))["canary"]
        self.assertIn(f"1 to {warmup.CANARY_COUNT}", canary["messages"][0]["content"])

    def test_the_counters_bracket_the_canary_rung_alone(self):
        # Judged from the canary only, so the verdict does not depend on which
        # other rungs a profile runs (2026-09-28: the full AXL ladder drafted 69).
        events = []

        def ask(request):
            events.append("ask")
            return answering(COUNT)(request)

        def counters():
            events.append("read")
            return {"num_draft_tokens": 0.0, "num_accepted_tokens": 0.0}

        warmup.run(
            self.profile,
            ask=ask,
            count_tokens=fake_count,
            logs=lambda: "",
            clock=lambda: 0,
            spec_counters=counters,
        )
        self.assertEqual(events[-3:], ["read", "ask", "read"])
        self.assertEqual(events.count("read"), 2)

    def test_a_healthy_answer_with_accepted_drafts_passes(self):
        for content in (COUNT, " " + COUNT + ".\n", COUNT.replace(" ", ", ")):
            with self.subTest(content=content[:12]):
                record = self.run_ladder(answering(content), self.counters(126, 126))
                self.assertIs(record["canary"]["answer_ok"], True)
                self.assertIs(record["canary"]["acceptance_ok"], True)
                self.assertIs(record["degenerate"], False)
                self.assertTrue(record["passed"])

    def test_garbage_or_a_truncated_answer_is_degenerate(self):
        short = " ".join(str(i) for i in range(1, warmup.CANARY_COUNT))
        for content, finish in (
            (short, "stop"),
            (COUNT, "length"),
            (COUNT.replace("41", "14"), "stop"),
            ("是的", "stop"),
        ):
            with self.subTest(content=content[-12:], finish=finish):
                record = self.run_ladder(
                    answering(content, finish), self.counters(126, 126)
                )
                self.assertIs(record["canary"]["answer_ok"], False)
                self.assertIs(record["degenerate"], True)
                self.assertFalse(record["passed"])

    def test_zero_acceptance_over_enough_drafts_is_degenerate(self):
        record = self.run_ladder(answering(COUNT), self.counters(warmup.MIN_DRAFTS, 0))
        self.assertIs(record["canary"]["acceptance_ok"], False)
        self.assertIs(record["degenerate"], True)

    def test_what_cannot_be_judged_never_trips(self):
        few = self.run_ladder(answering(COUNT), self.counters(warmup.MIN_DRAFTS - 1, 0))
        self.assertIsNone(few["canary"]["acceptance_ok"])
        self.assertIs(few["degenerate"], False)
        unread = self.run_ladder(answering(COUNT))
        self.assertIsNone(unread["canary"]["acceptance_ok"])
        self.profile["mtp"]["enabled"] = False
        off = self.run_ladder(answering(COUNT), self.counters(126, 0))
        self.assertIsNone(off["canary"]["acceptance_ok"])
        self.assertIs(off["degenerate"], False)

        def broken(request):
            raise RuntimeError("transport")

        failed = self.run_ladder(broken, self.counters(0, 0))
        self.assertIsNone(failed["canary"]["answer_ok"])
        self.assertIs(failed["degenerate"], False)
        self.assertFalse(failed["passed"])

        def unreadable():
            raise OSError("metrics down")

        record = warmup.run(
            self.profile,
            ask=answering(COUNT),
            count_tokens=fake_count,
            logs=lambda: "",
            clock=lambda: 0,
            spec_counters=unreadable,
        )
        self.assertIsNone(record["canary"]["acceptance_ok"])
        self.assertIs(record["degenerate"], False)


class SpecCounterTests(unittest.TestCase):
    def test_totals_sum_labelled_series_and_ignore_other_metrics(self):
        metrics = "\n".join(
            [
                "# HELP vllm:spec_decode_num_draft_tokens_total drafted",
                'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 30.0',
                'vllm:spec_decode_num_draft_tokens_total{engine="1",model_name="m"} 12.0',
                'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 23.0',
                'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 9.0',
                "vllm:num_requests_running 0.0",
            ]
        )
        self.assertEqual(
            warmup.spec_counters(metrics),
            {"num_draft_tokens": 42.0, "num_accepted_tokens": 23.0},
        )
