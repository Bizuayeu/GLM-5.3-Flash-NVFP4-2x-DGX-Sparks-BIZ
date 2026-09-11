import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import startup
from glm53_setup import startup_config as config

ROOT = Path(__file__).resolve().parents[1]


class StartupConfigTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(ROOT / "examples/startup.example.toml")

    def test_categories_control_both_ranks_and_context(self):
        self.profile["context"]["max_model_len"] = 8192
        for rank in (0, 1):
            args = config.serve_args(self.profile, rank, "/hf/model")
            self.assertEqual(args[args.index("--max-model-len") + 1], "8192")
            self.assertEqual(args[args.index("--node-rank") + 1], str(rank))
            self.assertEqual("--headless" in args, rank == 1)
            self.assertIn("--no-enable-prefix-caching", args)
            self.assertNotIn("--speculative-config", args)

    def test_display_rename_keeps_running_profile_fingerprint(self):
        legacy = copy.deepcopy(self.profile)
        legacy["llkv"] = legacy.pop("allkv")
        legacy["runtime"]["llkv_image"] = legacy["runtime"].pop("allkv_image")
        original = hashlib.sha256(
            json.dumps(
                {"settings": legacy, "lock": config.load_lock()}, sort_keys=True
            ).encode()
        ).hexdigest()
        self.assertEqual(config.fingerprint(self.profile), original)
        self.profile["context"]["max_model_len"] = 8192
        self.assertNotEqual(config.fingerprint(self.profile), original)

    def test_mtp_and_llkv_select_distinct_startup_paths(self):
        self.profile["mtp"]["enabled"] = True
        args = config.serve_args(self.profile, 0, "/hf/mtp-view")
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(
            spec,
            {"method": "mtp", "num_speculative_tokens": 3, "moe_backend": "triton"},
        )
        self.assertNotIn("--worker-extension-cls", args)
        self.profile["mtp"]["enabled"] = False
        self.profile["allkv"]["enabled"] = True
        args = config.serve_args(self.profile, 0, "/hf/model")
        self.assertIn("--worker-extension-cls", args)
        self.assertEqual(
            config.environment(self.profile, 0)["VLLM_SERVER_DEV_MODE"], "1"
        )
        self.profile["mtp"]["enabled"] = True
        args = config.serve_args(self.profile, 0, "/hf/mtp-view")
        self.assertIn("--speculative-config", args)
        self.assertIn("--worker-extension-cls", args)
        self.assertTrue(config.llkv_request(self.profile, 2048)["allow_mtp"])

    def test_command_mounts_mtp_view_and_projector_without_mutating_cache(self):
        self.profile["allkv"]["enabled"] = True
        self.profile["mtp"]["enabled"] = True
        path = ROOT / "state/startup.toml"
        args = startup.command(
            self.profile, path, 1, "test-container", ROOT / "state/test-hf"
        )
        self.assertIn(
            "/hf/local-views/glm53-mtp-compatible/" + config.load_lock()["revision"],
            args,
        )
        self.assertIn(
            str((path.parent / "projector.pt").resolve()) + ":/llkv/projector.pt:ro",
            args,
        )
        self.assertEqual(args[args.index("--memory") + 1], "112g")
        self.assertNotIn("--rm", args)
        self.assertNotIn("--privileged", args)

    def test_request_resets_llkv_after_generation_failure(self):
        self.profile["allkv"]["enabled"] = True
        calls = []

        def sender(profile, path, body):
            calls.append((path, copy.deepcopy(body)))
            if path == "/tokenize":
                return {"tokens": list(range(1024))}
            if path == "/v1/chat/completions":
                raise RuntimeError("generation failed")
            return {"results": []}

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            startup.ask(
                self.profile,
                {"messages": [{"role": "user", "content": "hello"}]},
                sender,
            )
        self.assertEqual(calls[1][1]["kwargs"]["mode"], "predict")
        self.assertEqual(calls[-1][1]["kwargs"]["mode"], "off")

    def test_request_discards_tokenization_mismatch(self):
        self.profile["allkv"]["enabled"] = True
        responses = [{"tokens": [1, 2]}, {}, {"usage": {"prompt_tokens": 3}}, {}]
        with patch.object(startup, "post", side_effect=responses) as sender:
            with self.assertRaisesRegex(ValueError, "Tokenization differs"):
                startup.ask(
                    self.profile,
                    {"messages": [{"role": "user", "content": "hello"}]},
                    sender,
                )
            self.assertEqual(sender.call_count, 4)

    def test_profile_mismatch_cannot_control_unrelated_container(self):
        info = {"Config": {"Labels": {startup.LABEL: "old"}}}
        with patch.object(startup.service, "run", return_value=json.dumps([info])):
            with self.assertRaises(ValueError):
                startup.inspect_owned("container", "new")

    def test_stop_does_not_depend_on_valid_edited_settings(self):
        with (
            patch.object(startup.os, "name", "posix"),
            patch.object(startup, "read_json", return_value={"name": "owned"}),
            patch.object(startup, "inspect_owned"),
            patch.object(
                startup.settings, "load", side_effect=ValueError("bad TOML")
            ) as load,
            patch.object(startup.service, "run", return_value="stopped") as run,
        ):
            startup.main(["stop", "--rank", "0"])
            load.assert_not_called()
            run.assert_called_once_with("docker", "stop", "owned")

    def test_invalid_or_incompatible_options_fail_closed(self):
        for section, key, value in [
            ("context", "max_model_len", True),
            ("context", "max_num_seqs", 2),
            ("context", "typo", 123),
            ("cache", "gpu_memory_utilization", float("nan")),
            ("mtp", "num_speculative_tokens", 2),
            ("allkv", "cut", 45),
            ("allkv", "tail", 0),
            ("cache", "prefix_caching", True),
            ("runtime", "enforce_eager", False),
        ]:
            with self.subTest(section=section, key=key):
                p = copy.deepcopy(self.profile)
                p["allkv"]["enabled"] = True
                p[section][key] = value
                with self.assertRaises(ValueError):
                    config.validate(p)

    def test_request_uses_template_and_protects_short_prompt(self):
        self.profile["allkv"]["enabled"] = True
        body = config.request_body(
            self.profile, {"messages": [{"role": "user", "content": "hello"}]}
        )
        self.assertEqual(
            body["chat_template_kwargs"],
            {"reasoning_effort": "low", "clear_thinking": True},
        )
        spec = config.llkv_request(self.profile, 100)
        self.assertEqual(spec["mode"], "off")
        self.assertEqual(spec["tail"], 100)
        self.assertEqual(
            config.llkv_request(self.profile, 2048)["predictor_path"],
            "/llkv/projector.pt",
        )
        with self.assertRaises(ValueError):
            config.request_body(self.profile, {"messages": [], "stream": True})


if __name__ == "__main__":
    unittest.main()
