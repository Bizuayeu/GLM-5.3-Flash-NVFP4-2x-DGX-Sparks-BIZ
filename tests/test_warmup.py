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
        self.assertEqual(names, ["text", "tool", "image"])
        self.profile["runtime"]["vision"] = False
        self.profile["generation"]["warmup_long_tokens"] = 65536
        names = [name for name, _ in warmup.rungs(self.profile)]
        self.assertEqual(names, ["text", "tool", "long"])
        for _, request in warmup.rungs(self.profile):
            if request is not None:
                self.assertEqual(request["max_tokens"], warmup.ANSWER_TOKENS)

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
                "choices": [{"finish_reason": "stop"}],
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
        self.assertEqual([r["rung"] for r in record["rungs"]], ["text", "tool", "long"])
        self.assertEqual([r["status"] for r in record["rungs"]], ["ok", "failed", "ok"])
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


class BurstTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(ROOT / "examples/server.example.toml")
        self.profile["runtime"]["vision"] = False
        self.gathered = []

    def ask(self, request):
        return {
            "usage": {"prompt_tokens": fake_count(str(request["messages"]))},
            "choices": [{"finish_reason": "length"}],
        }

    def gather(self, calls):
        # Sequential stand-in for the threads; records each burst's width.
        self.gathered.append(len(calls))
        results = []
        for call in calls:
            try:
                results.append(call())
            except Exception as error:  # noqa: BLE001 - mirrors warmup.threads
                results.append(error)
        return results

    def run_ladder(self, ask=None):
        return warmup.run(
            self.profile,
            ask=ask or self.ask,
            count_tokens=fake_count,
            logs=lambda: "",
            clock=lambda: 0,
            gather=self.gather,
        )

    def test_one_sequence_keeps_the_ladder_unchanged(self):
        self.assertEqual(self.profile["context"]["max_num_seqs"], 1)
        record = self.run_ladder()
        self.assertEqual(record["bursts"], [])
        self.assertEqual(self.gathered, [])
        self.assertEqual([r["rung"] for r in record["rungs"]], ["text", "tool"])
        self.assertTrue(record["passed"])

    def test_two_sequences_add_one_width_two_burst_after_the_rungs(self):
        self.profile["context"]["max_num_seqs"] = 2
        order = []

        def ask(request):
            order.append(request["messages"][0]["content"])
            return self.ask(request)

        record = self.run_ladder(ask)
        self.assertEqual(self.gathered, [2])
        self.assertEqual([b["burst"] for b in record["bursts"]], [2])
        burst = record["bursts"][0]
        self.assertEqual(burst["status"], "ok")
        self.assertEqual(len(burst["members"]), 2)
        # The rungs go first, then the burst's two distinct requests.
        self.assertEqual(order[0], "Reply with the word ready.")
        self.assertEqual(len(set(order[-2:])), 2)
        self.assertTrue(record["passed"])
        self.profile["context"]["max_num_seqs"] = 4
        self.assertEqual(warmup.burst_widths(self.profile), [2, 3, 4])

    def test_a_failed_member_is_recorded_and_does_not_abort(self):
        self.profile["context"]["max_num_seqs"] = 3
        calls = []

        def ask(request):
            calls.append(request)
            if "(2 of 2)" in request["messages"][0]["content"]:
                raise TimeoutError("second sequence timed out")
            return self.ask(request)

        resets = []
        record = warmup.run(
            self.profile,
            ask=ask,
            count_tokens=fake_count,
            logs=lambda: "",
            clock=lambda: 0,
            reset=lambda: resets.append(True),
            gather=self.gather,
        )
        first, second = record["bursts"]
        self.assertEqual(first["status"], "failed")
        self.assertEqual([m["status"] for m in first["members"]], ["ok", "failed"])
        self.assertEqual(first["members"][1]["error"], "TimeoutError")
        # The next width still ran, and the prefix reset still followed.
        self.assertEqual(second["status"], "ok")
        self.assertEqual(self.gathered, [2, 3])
        self.assertEqual(resets, [True])
        self.assertFalse(record["passed"])

    def test_a_gather_that_fails_is_recorded(self):
        self.profile["context"]["max_num_seqs"] = 2

        def broken(calls):
            raise RuntimeError("no threads")

        record = warmup.run(
            self.profile,
            ask=self.ask,
            count_tokens=fake_count,
            logs=lambda: "",
            clock=lambda: 0,
            gather=broken,
        )
        self.assertEqual(record["bursts"][0]["status"], "failed")
        self.assertEqual(record["bursts"][0]["error"], "RuntimeError")
        self.assertFalse(record["passed"])

    def test_threads_returns_each_result_or_its_exception_in_order(self):
        def fail():
            raise ValueError("x")

        results = warmup.threads([lambda: 1, fail, lambda: 3])
        self.assertEqual(results[0], 1)
        self.assertIsInstance(results[1], ValueError)
        self.assertEqual(results[2], 3)


if __name__ == "__main__":
    unittest.main()
