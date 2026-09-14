import copy
import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from glm53_setup import startup
from glm53_setup import startup_config as config

ROOT = Path(__file__).resolve().parents[1]


class StartupConfigTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(ROOT / "examples/startup.example.toml")
        # Independent feature tests start from an explicit all-off baseline.
        self.profile["runtime"]["index_checks"] = "auto"
        self.profile["mtp"]["enabled"] = False
        self.profile["lpa"]["enabled"] = False
        self.profile["cache"]["prefix_caching"] = False
        self.profile["cache"]["fused_unpack"] = False
        self.profile["cache"].pop("prefix_cache_retention_interval", None)

    def test_distributed_profile_wires_combined_paths_on_both_ranks(self):
        profile = config.load(ROOT / "examples/startup.example.toml")
        for rank in (0, 1):
            args = config.serve_args(profile, rank, "/hf/mtp-view")
            env = config.environment(profile, rank)
            spec = json.loads(args[args.index("--speculative-config") + 1])
            self.assertEqual(spec["num_speculative_tokens"], 3)
            self.assertEqual(args[args.index("--max-model-len") + 1], "262144")
            self.assertEqual(
                args[args.index("--kv-cache-memory-bytes") + 1], "3221225472"
            )
            self.assertIn("--enable-prefix-caching", args)
            self.assertEqual(
                args[args.index("--prefix-cache-retention-interval") + 1], "None"
            )
            self.assertEqual(env["GLM53_ASYNC_INDEX_CHECKS"], "1")
            self.assertEqual(env["GLM53_FUSED_UNPACK"], "1")
            # LPA is a batch opt-in: the distributed profile keeps prefix caching
            # instead, because an approximated request publishes nothing.
            self.assertNotIn("GLM53_APC_LPA_CONFIG", env)
            self.assertNotIn("--worker-extension-cls", args)
        self.assertEqual(profile["resources"]["run_seconds"], 0)
        self.assertEqual(profile["resources"]["reserve_gib"], 3)

    def test_index_check_mode_is_explicit_without_disabling_validation(self):
        self.assertNotIn(
            "GLM53_ASYNC_INDEX_CHECKS", config.environment(self.profile, 0)
        )
        self.profile["runtime"]["index_checks"] = "async"
        config.validate(self.profile)
        for rank in (0, 1):
            self.assertEqual(
                config.environment(self.profile, rank)["GLM53_ASYNC_INDEX_CHECKS"], "1"
            )
            self.assertIn(
                "--enforce-eager", config.serve_args(self.profile, rank, "/hf/model")
            )
        self.profile["runtime"]["index_checks"] = "disabled"
        with self.assertRaises(ValueError):
            config.validate(self.profile)
        self.profile["runtime"]["index_checks"] = "sync"
        self.profile["runtime"]["enforce_eager"] = False
        with self.assertRaisesRegex(ValueError, "Graph"):
            config.validate(self.profile)

    def test_toml_validation_stays_at_load_and_command_boundaries(self):
        with patch.object(config, "validate", wraps=config.validate) as validate:
            config.load(ROOT / "examples/startup.example.toml")
            self.assertEqual(validate.call_count, 1)
            validate.reset_mock()
            config.serve_args(self.profile, 0, "/hf/model")
            validate.assert_not_called()
            startup.command(self.profile, ROOT / "state/startup.toml", 0, "test")
            self.assertEqual(validate.call_count, 1)
            validate.reset_mock()
            self.profile["context"]["max_num_seqs"] = 0
            with self.assertRaises(ValueError):
                startup.command(self.profile, ROOT / "state/startup.toml", 0, "test")
            self.assertEqual(validate.call_count, 1)

    def test_throughput_sequences_are_separate_from_lpa_layout(self):
        self.profile["context"]["max_num_seqs"] = 2
        config.validate(self.profile)
        self.profile["lpa"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "LPA requires max_num_seqs=1"):
            config.validate(self.profile)

    def test_graphs_are_explicit_uncompiled_decode_with_checked_indices(self):
        eager_env = config.environment(self.profile, 0)
        self.assertNotIn("GLM53_ASYNC_INDEX_CHECKS", eager_env)
        self.profile["runtime"]["enforce_eager"] = False
        config.validate(self.profile)
        for rank in (0, 1):
            args = config.serve_args(self.profile, rank, "/hf/model")
            self.assertNotIn("--enforce-eager", args)
            self.assertEqual(
                json.loads(args[args.index("--compilation-config") + 1]),
                {
                    "mode": 0,
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [1],
                },
            )
            self.assertEqual(
                config.environment(self.profile, rank)["GLM53_ASYNC_INDEX_CHECKS"], "1"
            )
            self.assertNotIn(
                "GLM53_FUSED_UNPACK", config.environment(self.profile, rank)
            )

    def test_prompt_tokens_details_flag_is_optional_and_explicit(self):
        self.profile["api"]["prompt_tokens_details"] = True
        config.validate(self.profile)
        self.assertIn(
            "--enable-prompt-tokens-details",
            config.serve_args(self.profile, 0, "/hf/model"),
        )
        self.profile["api"]["prompt_tokens_details"] = False
        self.assertNotIn(
            "--enable-prompt-tokens-details",
            config.serve_args(self.profile, 0, "/hf/model"),
        )
        self.profile["api"].pop("prompt_tokens_details")
        config.validate(self.profile)
        self.assertNotIn(
            "--enable-prompt-tokens-details",
            config.serve_args(self.profile, 0, "/hf/model"),
        )

    def test_graph_combination_scope_is_explicit_until_integration(self):
        for section, key, value in (
            ("mtp", "enabled", True),
            ("cache", "prefix_caching", True),
            ("context", "max_num_seqs", 2),
        ):
            p = copy.deepcopy(self.profile)
            p["runtime"]["enforce_eager"] = False
            p[section][key] = value
            with self.assertRaisesRegex(ValueError, "Graph"):
                config.validate(p)

    def test_component_worker_is_an_explicit_independent_diagnostic(self):
        self.profile["validation"]["component_worker"] = True
        config.validate(self.profile)
        args = config.serve_args(self.profile, 0, "/hf/model")
        self.assertEqual(
            args[args.index("--worker-extension-cls") + 1],
            "glm53_setup.runtime.component_worker.ComponentWorker",
        )
        for feature in ("lpa", "mtp"):
            self.profile[feature]["enabled"] = True
            with self.assertRaises(ValueError):
                config.validate(self.profile)
            self.profile[feature]["enabled"] = False

    def test_kernel_profiling_omits_frontend_and_duplicate_summary_materialization(
        self,
    ):
        self.profile["profiling"]["enabled"] = True
        args = config.serve_args(self.profile, 0, "/hf/model")
        options = json.loads(args[args.index("--profiler-config") + 1])
        self.assertTrue(options["ignore_frontend"])
        self.assertFalse(options["torch_profiler_dump_cuda_time_total"])
        self.assertFalse(options["torch_profiler_record_shapes"])

    def test_no_deadline_is_valid_but_negative_or_boolean_is_not(self):
        self.profile["resources"]["run_seconds"] = 0
        config.validate(self.profile)
        for invalid in (-1, True):
            self.profile["resources"]["run_seconds"] = invalid
            with self.assertRaises(ValueError):
                config.validate(self.profile)

    def test_reasoning_levels_follow_the_glm_model_contract(self):
        for effort in ("low", "high", "max"):
            self.profile["generation"]["reasoning_effort"] = effort
            config.validate(self.profile)
        self.profile["generation"]["reasoning_effort"] = "medium"
        with self.assertRaises(ValueError):
            config.validate(self.profile)

    def test_unlimited_run_keeps_memory_protection_and_timed_run_expires(self):
        for seconds, reason, sleeps in [
            (0, "memory-reserve", 1),
            (1800, "run-deadline", 0),
        ]:
            with self.subTest(seconds=seconds):
                self.profile["resources"]["run_seconds"] = seconds
                with (
                    patch("pathlib.Path.open", mock_open()),
                    patch.object(
                        startup,
                        "inspect_owned",
                        return_value={"State": {"Running": True}},
                    ),
                    patch.object(
                        startup,
                        "available_gib",
                        side_effect=[100, self.profile["resources"]["reserve_gib"] - 1],
                    ),
                    patch.object(startup.time, "monotonic", side_effect=[0, 1000000]),
                    patch.object(startup.time, "sleep") as sleep,
                    patch.object(startup, "write_json") as write,
                    patch.object(startup.service, "run") as stop,
                    patch.object(startup.subprocess, "run"),
                ):
                    startup.supervise(self.profile, "owned", Path("record"))
                    write.assert_any_call(
                        Path("record/stop-reason.json"), {"reason": reason}
                    )
                    stop.assert_called_once_with("docker", "stop", "owned")
                    self.assertEqual(sleep.call_count, sleeps)

    def test_categories_control_both_ranks_and_context(self):
        self.profile["context"]["max_model_len"] = 8192
        for rank in (0, 1):
            args = config.serve_args(self.profile, rank, "/hf/model")
            self.assertEqual(args[args.index("--max-model-len") + 1], "8192")
            self.assertEqual(args[args.index("--node-rank") + 1], str(rank))
            self.assertEqual("--headless" in args, rank == 1)
            self.assertIn("--no-enable-prefix-caching", args)
            self.assertNotIn("--speculative-config", args)

    def test_mtp_and_lpa_select_distinct_startup_paths(self):
        self.profile["mtp"]["enabled"] = True
        args = config.serve_args(self.profile, 0, "/hf/mtp-view")
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(
            spec,
            {"method": "mtp", "num_speculative_tokens": 3, "moe_backend": "triton"},
        )
        self.assertNotIn("--worker-extension-cls", args)
        self.profile["mtp"]["enabled"] = False
        self.profile["lpa"]["enabled"] = True
        args = config.serve_args(self.profile, 0, "/hf/model")
        self.assertIn("--worker-extension-cls", args)
        self.assertEqual(
            config.environment(self.profile, 0)["VLLM_SERVER_DEV_MODE"], "1"
        )
        self.profile["mtp"]["enabled"] = True
        args = config.serve_args(self.profile, 0, "/hf/mtp-view")
        self.assertIn("--speculative-config", args)
        self.assertIn("--worker-extension-cls", args)
        self.assertTrue(config.lpa_request(self.profile, 2048)["allow_mtp"])

    def test_command_mounts_mtp_view_and_projector_without_mutating_cache(self):
        self.profile["lpa"]["enabled"] = True
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
            str((path.parent / "projector.pt").resolve()) + ":/lpa/projector.pt:ro",
            args,
        )
        self.assertEqual(args[args.index("--memory") + 1], "112g")
        self.assertNotIn("--rm", args)
        self.assertNotIn("--privileged", args)

    def test_request_resets_lpa_after_generation_failure(self):
        self.profile["lpa"]["enabled"] = True
        calls = []

        def sender(profile, path, body):
            calls.append((path, copy.deepcopy(body)))
            if path == "/tokenize":
                return {"tokens": list(range(2048))}
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
        self.profile["lpa"]["enabled"] = True
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
            ("lpa", "cut", 45),
            ("lpa", "tail", 0),
            ("lpa", "break_even_tokens", -1),
            ("runtime", "enforce_eager", False),
        ]:
            with self.subTest(section=section, key=key):
                p = copy.deepcopy(self.profile)
                p["lpa"]["enabled"] = True
                p[section][key] = value
                with self.assertRaises(ValueError):
                    config.validate(p)

    def test_request_uses_template_and_protects_short_prompt(self):
        self.profile["lpa"]["enabled"] = True
        body = config.request_body(
            self.profile, {"messages": [{"role": "user", "content": "hello"}]}
        )
        self.assertEqual(
            body["chat_template_kwargs"],
            {"reasoning_effort": "low", "clear_thinking": True},
        )
        spec = config.lpa_request(self.profile, 100)
        self.assertEqual(spec["mode"], "off")
        self.assertEqual(spec["tail"], 100)
        self.assertEqual(
            config.lpa_request(self.profile, 2048)["predictor_path"],
            "/lpa/projector.pt",
        )
        with self.assertRaises(ValueError):
            config.request_body(self.profile, {"messages": [], "stream": True})


if __name__ == "__main__":
    unittest.main()
