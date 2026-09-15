import json
import unittest
from unittest.mock import Mock

from glm53_setup import server, server_config
from glm53_setup.config import ROOT


class APCStartupTests(unittest.TestCase):
    def profile(self):
        value = server_config.load(ROOT / "examples/server.example.toml")
        value["cache"]["prefix_caching"] = True
        value["lpa"]["enabled"] = True
        value["lpa"]["break_even_tokens"] = 1024
        return value

    def test_apc_priority_is_wired_from_the_unified_profile(self):
        value = self.profile()
        server_config.validate(value)
        config = json.loads(server_config.environment(value, 0)["GLM53_APC_LPA_CONFIG"])
        self.assertEqual(config["break_even"], 1024)
        self.assertEqual(config["tail"], value["lpa"]["tail"])
        self.assertEqual(config["projector_sha256"], value["lpa"]["projector_sha256"])
        self.assertEqual(config["skip_mla_queries"], value["lpa"]["skip_mla_queries"])
        value["cache"]["prefix_caching"] = False
        self.assertNotIn("GLM53_APC_LPA_CONFIG", server_config.environment(value, 0))

    def test_combined_client_leaves_n_and_h_to_server_admission(self):
        sender = Mock(return_value={"ok": True})
        result = server.ask(
            self.profile(),
            {
                "messages": [{"role": "user", "content": "prime"}],
                "vllm_xargs": {"glm53_lpa_mode": "off"},
            },
            sender=sender,
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(sender.call_count, 1)
        self.assertEqual(sender.call_args.args[1], "/v1/chat/completions")
        self.assertEqual(
            sender.call_args.args[2]["vllm_xargs"]["glm53_lpa_mode"], "off"
        )

    def test_invalid_threshold_does_not_enable_a_different_policy(self):
        for threshold in (-1, 0.5, True):
            value = self.profile()
            value["lpa"]["break_even_tokens"] = threshold
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                server_config.validate(value)

    def test_no_cache_client_uses_h_zero_for_the_same_threshold(self):
        value = self.profile()
        value["cache"]["prefix_caching"] = False
        value["lpa"]["tail"] = 512
        self.assertEqual(server_config.lpa_request(value, 1536)["mode"], "off")
        self.assertEqual(server_config.lpa_request(value, 1537)["mode"], "predict")


if __name__ == "__main__":
    unittest.main()
