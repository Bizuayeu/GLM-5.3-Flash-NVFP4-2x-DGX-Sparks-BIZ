import copy
import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from glm53_setup import server
from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]


class ServerConfigTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(ROOT / "examples/server.example.toml")
        # Independent feature tests start from an explicit all-off baseline.
        self.profile["runtime"]["index_checks"] = "auto"
        self.profile["runtime"]["vision"] = False
        self.profile["mtp"]["enabled"] = False
        self.profile["lpa"]["enabled"] = False
        self.profile["cache"]["prefix_caching"] = False
        self.profile["cache"]["fused_unpack"] = False
        self.profile["cache"].pop("prefix_cache_retention_interval", None)

    def test_distributed_profile_wires_combined_paths_on_both_ranks(self):
        profile = config.load(ROOT / "examples/server.example.toml")
        for rank in (0, 1):
            args = config.serve_args(profile, rank, "/hf/mtp-view")
            env = config.environment(profile, rank)
            spec = json.loads(args[args.index("--speculative-config") + 1])
            self.assertEqual(spec["num_speculative_tokens"], 3)
            self.assertEqual(args[args.index("--max-model-len") + 1], "204800")
            self.assertEqual(
                args[args.index("--kv-cache-memory-bytes") + 1], "2684354560"
            )
            self.assertIn("--enable-prefix-caching", args)
            # The distributed profile accepts images; video stays rejected.
            self.assertNotIn("--language-model-only", args)
            limit = args[args.index("--limit-mm-per-prompt") + 1]
            self.assertEqual(json.loads(limit), {"video": 0})
            self.assertEqual(args[args.index("--mm-processor-cache-gb") + 1], "0.1")
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
        self.assertEqual(profile["resources"]["reserve_gib"], 2.5)

    def test_reserve_accepts_fractional_gib(self):
        self.profile["resources"]["reserve_gib"] = 2.5
        config.validate(self.profile)
        for bad in ("2.5", float("nan"), 0.5):
            profile = copy.deepcopy(self.profile)
            profile["resources"]["reserve_gib"] = bad
            with self.assertRaises(ValueError):
                config.validate(profile)

    def test_jit_caches_live_in_the_mounted_runtime_cache(self):
        for rank in (0, 1):
            env = config.environment(self.profile, rank)
            self.assertEqual(env["TRITON_CACHE_DIR"], "/root/.cache/triton")
            self.assertEqual(env["TILELANG_CACHE_DIR"], "/root/.cache/tilelang")
            self.assertEqual(
                env["TORCHINDUCTOR_CACHE_DIR"], "/root/.cache/torchinductor"
            )

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
            config.load(ROOT / "examples/server.example.toml")
            self.assertEqual(validate.call_count, 1)
            validate.reset_mock()
            config.serve_args(self.profile, 0, "/hf/model")
            validate.assert_not_called()
            server.command(self.profile, ROOT / "state/server.toml", 0, "test")
            self.assertEqual(validate.call_count, 1)
            validate.reset_mock()
            self.profile["context"]["max_num_seqs"] = 0
            with self.assertRaises(ValueError):
                server.command(self.profile, ROOT / "state/server.toml", 0, "test")
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

    def test_dev_endpoints_toggle_is_optional_and_explicit(self):
        # Omitted or false: the stock surface; the dev router stays unmounted.
        for rank in (0, 1):
            self.assertNotIn(
                "VLLM_SERVER_DEV_MODE", config.environment(self.profile, rank)
            )
        self.profile["api"]["dev_endpoints"] = True
        config.validate(self.profile)
        for rank in (0, 1):
            self.assertEqual(
                config.environment(self.profile, rank)["VLLM_SERVER_DEV_MODE"], "1"
            )
        distributed = config.load(ROOT / "examples/server.example.toml")
        self.assertIs(distributed["api"]["dev_endpoints"], False)
        for bad in (1, "true", None):
            profile = copy.deepcopy(self.profile)
            profile["api"]["dev_endpoints"] = bad
            with self.assertRaises(ValueError):
                config.validate(profile)

    def test_stall_and_warmup_keys_are_optional_nonnegative(self):
        for section, key in (
            ("resources", "stall_seconds"),
            ("generation", "warmup_long_tokens"),
        ):
            profile = copy.deepcopy(self.profile)
            profile[section].pop(key, None)
            config.validate(profile)
            profile[section][key] = 0
            config.validate(profile)
            for bad in (-1, 1.5, "600", True):
                profile[section][key] = bad
                with self.assertRaises(ValueError):
                    config.validate(profile)
        profile = copy.deepcopy(self.profile)
        profile["generation"]["warmup"] = "yes"
        with self.assertRaises(ValueError):
            config.validate(profile)
        # The long rung must leave room for the answer inside the context.
        profile = copy.deepcopy(self.profile)
        profile["generation"]["warmup_long_tokens"] = (
            profile["context"]["max_model_len"] - profile["generation"]["max_tokens"]
        )
        with self.assertRaises(ValueError):
            config.validate(profile)
        distributed = config.load(ROOT / "examples/server.example.toml")
        self.assertEqual(distributed["resources"]["stall_seconds"], 600)
        self.assertIs(distributed["generation"]["warmup"], True)

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

    def test_vision_is_optional_and_defaults_to_text_only(self):
        self.profile["runtime"].pop("vision", None)
        config.validate(self.profile)
        for rank in (0, 1):
            self.assertIn(
                "--language-model-only",
                config.serve_args(self.profile, rank, "/hf/model"),
            )
        self.profile["runtime"]["vision"] = False
        config.validate(self.profile)
        args = config.serve_args(self.profile, 0, "/hf/model")
        self.assertIn("--language-model-only", args)
        self.assertNotIn("--limit-mm-per-prompt", args)
        self.profile["runtime"]["vision"] = True
        config.validate(self.profile)
        for rank in (0, 1):
            args = config.serve_args(self.profile, rank, "/hf/model")
            self.assertNotIn("--language-model-only", args)
            # Video is disabled: startup profiling would otherwise push a
            # 30,000-token video through the vision tower.
            self.assertEqual(
                json.loads(args[args.index("--limit-mm-per-prompt") + 1]),
                {"video": 0},
            )
        self.profile["runtime"]["vision"] = "true"
        with self.assertRaisesRegex(ValueError, "runtime.vision"):
            config.validate(self.profile)

    def test_vision_processor_cache_is_small_by_default_and_explicit(self):
        def cache_arg(profile):
            args = config.serve_args(profile, 0, "/hf/model")
            if "--mm-processor-cache-gb" not in args:
                return None
            return args[args.index("--mm-processor-cache-gb") + 1]

        self.profile["cache"].pop("mm_processor_cache_gb", None)
        self.profile["runtime"]["vision"] = False
        config.validate(self.profile)
        self.assertIsNone(cache_arg(self.profile))
        # vLLM's 4 GiB default is duplicated in the head's API and engine
        # processes; absent the key, vision uses the small repo default.
        self.profile["runtime"]["vision"] = True
        config.validate(self.profile)
        self.assertEqual(cache_arg(self.profile), "0.1")
        self.profile["cache"]["mm_processor_cache_gb"] = 0.25
        config.validate(self.profile)
        self.assertEqual(cache_arg(self.profile), "0.25")
        self.profile["cache"]["mm_processor_cache_gb"] = 0
        config.validate(self.profile)
        self.assertEqual(cache_arg(self.profile), "0")
        self.profile["runtime"]["vision"] = False
        self.assertIsNone(cache_arg(self.profile))
        for bad in (-1, "0.1", float("nan"), True):
            self.profile["cache"]["mm_processor_cache_gb"] = bad
            with self.assertRaisesRegex(ValueError, "mm_processor_cache_gb"):
                config.validate(self.profile)

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
                self.profile["resources"]["stall_seconds"] = 0
                with (
                    patch("pathlib.Path.open", mock_open()),
                    patch.object(
                        server,
                        "inspect_owned",
                        return_value={"State": {"Running": True}},
                    ),
                    patch.object(
                        server,
                        "available_gib",
                        side_effect=[100, self.profile["resources"]["reserve_gib"] - 1],
                    ),
                    patch.object(server.time, "monotonic", side_effect=[0, 1000000]),
                    patch.object(server.time, "sleep") as sleep,
                    patch.object(server, "write_json") as write,
                    patch.object(server.host, "run") as stop,
                    patch.object(server.subprocess, "run"),
                ):
                    server.supervise(self.profile, "owned", Path("record"), 0)
                    write.assert_any_call(
                        Path("record/stop-reason.json"), {"reason": reason}
                    )
                    stop.assert_called_once_with("docker", "stop", "owned")
                    self.assertEqual(sleep.call_count, sleeps)

    def _supervise(self, samples, monotonic, rank=0, stall=600):
        self.profile["resources"]["run_seconds"] = 0
        self.profile["resources"]["stall_seconds"] = stall
        with (
            patch("pathlib.Path.open", mock_open()),
            patch.object(
                server, "inspect_owned", return_value={"State": {"Running": True}}
            ),
            patch.object(server, "available_gib", return_value=100),
            patch.object(server, "progress_sample", side_effect=samples) as probe,
            patch.object(server.time, "monotonic", side_effect=monotonic),
            patch.object(server.time, "sleep"),
            patch.object(server, "write_json") as write,
            patch.object(server.host, "run") as stop,
            patch.object(server.subprocess, "run"),
        ):
            server.supervise(self.profile, "owned", Path("record"), rank)
        return probe, write, stop

    def test_engine_stall_stops_only_when_every_progress_signal_is_frozen(self):
        running = {
            "vllm:num_requests_running": 1,
            "vllm:kv_cache_usage_perc": 0.5,
            "vllm:prompt_tokens_total": 100,
            "vllm:generation_tokens_total": 40,
        }
        prefilling = {**running, "vllm:kv_cache_usage_perc": 0.6}
        # Sample at 0 s, prefill moves the KV usage at 500 s, then nothing
        # moves from 500 s to 1200 s: the stall clock restarts at the move.
        # monotonic: stall base, one read per move, one per frozen check.
        probe, write, stop = self._supervise(
            [running, prefilling, prefilling, prefilling],
            [0, 0, 500, 900, 1200],
        )
        self.assertEqual(probe.call_count, 4)
        write.assert_any_call(
            Path("record/stop-reason.json"),
            {"reason": "engine-stall", "stall_seconds": 600, "progress": prefilling},
        )
        stop.assert_called_once_with("docker", "stop", "owned")

    def test_idle_engine_unreachable_metrics_and_rank_one_never_stall(self):
        idle = {
            "vllm:num_requests_running": 0,
            "vllm:kv_cache_usage_perc": 0.0,
            "vllm:prompt_tokens_total": 100,
            "vllm:generation_tokens_total": 40,
        }
        frozen = {**idle, "vllm:num_requests_running": 1}
        for name, samples in (
            ("idle", [idle, idle, idle, StopIteration]),
            ("unreachable", [frozen, None, None, StopIteration]),
        ):
            with self.subTest(name=name):
                # Neither an idle engine nor a lost sample counts as a stall,
                # however long the clock runs; the probe runs out first.
                with self.assertRaises(StopIteration):
                    self._supervise(
                        samples, [0, 0, 5000, 5000, 9000, 9000, 20000, 20000]
                    )
        # The worker has no API; /metrics is never asked for.
        with patch.object(server, "progress_sample") as probe:
            with (
                patch("pathlib.Path.open", mock_open()),
                patch.object(
                    server, "inspect_owned", return_value={"State": {"Running": False}}
                ),
                patch.object(server.host, "run"),
                patch.object(server.subprocess, "run"),
                patch.object(server, "write_json"),
            ):
                self.profile["resources"]["stall_seconds"] = 600
                server.supervise(self.profile, "owned", Path("record"), 1)
            probe.assert_not_called()

    def test_unobservable_metrics_never_stop_the_head(self):
        for error in (TimeoutError(), ConnectionResetError(), ValueError("x")):
            with self.subTest(error=type(error).__name__):
                with patch.object(server, "metrics_text", side_effect=error):
                    self.assertIsNone(server.progress_sample(self.profile))
        with patch.object(server, "metrics_text", return_value="# nothing\n"):
            self.assertIsNone(server.progress_sample(self.profile))

    def test_metrics_parser_sums_label_sets_and_ignores_comments(self):
        text = (
            "# HELP vllm:num_requests_running x\n"
            'vllm:num_requests_running{engine="0",model="m"} 1.0\n'
            'vllm:num_requests_running{engine="1",model="m"} 2.0\n'
            'vllm:generation_tokens_total{model="m"} 5\n'
            'vllm:other{model="m"} 9\n'
            "vllm:kv_cache_usage_perc{} nan-ish\n"
        )
        self.assertEqual(
            server.parse_metrics(text, server.PROGRESS_SIGNALS),
            {"vllm:num_requests_running": 3.0, "vllm:generation_tokens_total": 5.0},
        )

    def test_dev_endpoints_gate_reset_and_capacity_layout(self):
        self.profile["lpa"]["enabled"] = False
        self.assertFalse(server.dev_endpoints(self.profile))
        self.profile["api"]["dev_endpoints"] = True
        self.assertTrue(server.dev_endpoints(self.profile))
        with (
            patch.object(server, "container_logs", return_value=""),
            patch.object(server, "metrics_text", return_value=""),
            patch.object(server, "collective_rpc") as rpc,
        ):
            report = server.capacity_report(self.profile, "owned")
            rpc.assert_not_called()  # No LPA extension: no layout RPC.
        self.assertIn("withheld", report["cached_conversations"])

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
        path = ROOT / "state/server.toml"
        args = server.command(
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
            server.ask(
                self.profile,
                {"messages": [{"role": "user", "content": "hello"}]},
                sender,
            )
        self.assertEqual(calls[1][1]["kwargs"]["mode"], "predict")
        self.assertEqual(calls[-1][1]["kwargs"]["mode"], "off")

    def test_request_discards_tokenization_mismatch(self):
        self.profile["lpa"]["enabled"] = True
        responses = [{"tokens": [1, 2]}, {}, {"usage": {"prompt_tokens": 3}}, {}]
        with patch.object(server, "post", side_effect=responses) as sender:
            with self.assertRaisesRegex(ValueError, "Tokenization differs"):
                server.ask(
                    self.profile,
                    {"messages": [{"role": "user", "content": "hello"}]},
                    sender,
                )
            self.assertEqual(sender.call_count, 4)

    def test_image_capability_markers_follow_enabled_features(self):
        # A missing or empty image env fails every required marker closed.
        for image_config in ({}, {"Env": None}):
            checks = server.image_capability_checks(
                self.profile, {"Config": image_config}
            )
            self.assertEqual(checks, {"reference_attention": False})
        self.profile["lpa"]["enabled"] = True
        self.profile["cache"]["prefix_caching"] = True
        self.profile["cache"]["fused_unpack"] = True
        env = ["GLM53_REFERENCE_ATTENTION=1", "GLM53_LPA_API=2", "GLM53_APC_LPA_API=1"]
        checks = server.image_capability_checks(self.profile, {"Config": {"Env": env}})
        self.assertEqual(
            checks,
            {
                "fused_unpack_support": False,
                "lpa_worker": True,
                "apc_lpa_support": True,
                "reference_attention": True,
            },
        )
        # Disabled features are not checked, so their markers may be absent.
        self.assertNotIn("pipeline_support", checks)
        self.assertNotIn("decode_graph_support", checks)

    def test_profile_mismatch_cannot_control_unrelated_container(self):
        info = {"Config": {"Labels": {server.LABEL: "old"}}}
        with patch.object(server.host, "run", return_value=json.dumps([info])):
            with self.assertRaises(ValueError):
                server.inspect_owned("container", "new")

    def test_stop_does_not_depend_on_valid_edited_settings(self):
        with (
            patch.object(server.os, "name", "posix"),
            patch.object(server, "read_json", return_value={"name": "owned"}),
            patch.object(server, "inspect_owned"),
            patch.object(
                server.settings, "load", side_effect=ValueError("bad TOML")
            ) as load,
            patch.object(server.host, "run", return_value="stopped") as run,
        ):
            server.main(["stop", "--rank", "0"])
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
