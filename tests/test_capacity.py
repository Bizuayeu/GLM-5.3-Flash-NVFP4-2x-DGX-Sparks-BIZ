import unittest
from pathlib import Path

from glm53_setup import capacity
from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]

LOGS = """
(EngineCore pid=1) INFO [kv_cache_utils.py:2315] GPU KV cache size: 235,016 tokens, Maximum concurrency for 204,800 tokens per request: 1.15x
(EngineCore pid=1) INFO [kv_cache_utils.py:732] kv cache group sizes [4608, 4, 16384, 16384, 16384]
(EngineCore pid=1) INFO [kv_cache_utils.py:733] kv lcm block sizes 147456
"""

METRICS = (
    "# HELP vllm:cache_config_info x\n"
    'vllm:cache_config_info{block_size="4608",num_gpu_blocks="920",'
    'kv_cache_size_tokens="None",kv_cache_max_concurrency="None",'
    'enable_prefix_caching="True"} 1.0\n'
)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(ROOT / "examples/server.example.toml")

    def test_stock_line_is_decomposed_and_named_for_what_it_is(self):
        report = capacity.summarize(self.profile, LOGS, METRICS)
        self.assertEqual(report["stock"]["kv_cache_size_tokens"], 235016)
        self.assertEqual(report["stock"]["max_model_len"], 204800)
        self.assertEqual(report["group_block_sizes"], [4608, 4, 16384, 16384, 16384])
        self.assertEqual(report["scheduler_block_size"], 147456)
        self.assertEqual(report["num_gpu_blocks"], 920)
        self.assertEqual(report["full_length_requests_that_fit"], 1)
        self.assertEqual(report["blocks_per_max_length_request"], 800)
        self.assertNotIn("kv_cache_size_tokens", report["cache_config_info"])
        self.assertIn("max_concurrency x max_model_len", report["meaning"])
        # Without the worker extension the conversation figure is withheld.
        self.assertIn("withheld", report["cached_conversations"])

    def test_conversation_estimate_uses_group_kinds_and_dense_retention(self):
        layout = {
            "rank": 0,
            "num_blocks": 921,
            "retention_interval": None,
            "groups": [
                {"kind": "MLAAttentionSpec", "block_size": 4608, "layers": []},
                {"kind": "KpoolTailSpec", "block_size": 4, "layers": []},
                {"kind": "MambaSpec", "block_size": 16384, "layers": []},
                {"kind": "MambaSpec", "block_size": 16384, "layers": []},
            ],
        }
        report = capacity.summarize(self.profile, LOGS, METRICS, layout)
        self.assertEqual(report["num_gpu_blocks"], 921)
        rows = {row["tokens"]: row for row in report["cached_conversations"]["rows"]}
        self.assertEqual(rows[16384]["per_group"], [4, 0, 1, 1])
        self.assertEqual(rows[16384]["conversations"], 920 // 6)
        self.assertEqual(rows[204800]["per_group"], [45, 0, 13, 13])
        self.assertEqual(rows[204800]["conversations"], 920 // 71)
        # A retention interval other than dense, or an unknown kind, withholds.
        layout["retention_interval"] = 0
        self.assertIn(
            "withheld",
            capacity.summarize(self.profile, LOGS, METRICS, layout)[
                "cached_conversations"
            ],
        )
        layout["retention_interval"] = None
        layout["groups"].append(
            {"kind": "SinkFullAttentionSpec", "block_size": 64, "layers": []}
        )
        withheld = capacity.summarize(self.profile, LOGS, METRICS, layout)[
            "cached_conversations"
        ]
        self.assertEqual(
            withheld, {"withheld": ["SinkFullAttentionSpec is not modelled"]}
        )

    def test_missing_lines_do_not_invent_numbers(self):
        report = capacity.summarize(self.profile, "", "")
        self.assertNotIn("stock", report)
        self.assertIsNone(report["num_gpu_blocks"])
        self.assertNotIn("blocks_per_max_length_request", report)


if __name__ == "__main__":
    unittest.main()
