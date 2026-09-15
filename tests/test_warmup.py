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


if __name__ == "__main__":
    unittest.main()
