import contextlib
import copy
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch

from glm53_setup import server
from glm53_setup import server_config as config
from glm53_setup.config import DEFAULT_PROFILE, STATE

ROOT = Path(__file__).resolve().parents[1]


class ServerConfigTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(ROOT / "examples/server.example.toml")
        # Independent feature tests start from an explicit all-off baseline.
        self.profile["runtime"]["index_checks"] = "auto"
        self.profile["runtime"]["vision"] = False
        self.profile["runtime"]["canonical_moe_order"] = False
        self.profile["runtime"]["stable_indexer_topk"] = False
        self.profile["runtime"]["fa2_attention"] = False
        self.profile["runtime"]["inductor_deterministic"] = False
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
            self.assertEqual(args[args.index("--max-model-len") + 1], "262144")
            self.assertEqual(
                args[args.index("--kv-cache-memory-bytes") + 1], "3221225472"
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
        self.assertEqual(profile["resources"]["reserve_gib"], 3.0)

    def test_reserve_accepts_fractional_gib(self):
        self.profile["resources"]["reserve_gib"] = 2.5
        config.validate(self.profile)
        for bad in ("2.5", float("nan"), 0.5):
            profile = copy.deepcopy(self.profile)
            profile["resources"]["reserve_gib"] = bad
            with self.assertRaises(ValueError):
                config.validate(profile)

    def test_optional_per_node_cpu_sets_keep_the_default_command_unchanged(self):
        for rank in (0, 1):
            self.assertNotIn(
                "--cpuset-cpus",
                server.command(self.profile, ROOT / "state/server.toml", rank, "test"),
            )
        self.profile["nodes"][0]["cpuset_cpus"] = "5-9,15-19"
        self.profile["nodes"][1]["cpuset_cpus"] = "1,3-4"
        config.validate(self.profile)
        for rank, requested in ((0, "5-9,15-19"), (1, "1,3-4")):
            args = server.command(
                self.profile, ROOT / "state/server.toml", rank, "test"
            )
            self.assertEqual(args[args.index("--cpuset-cpus") + 1], requested)
        for invalid in ("", "1-", "4-2", "1,1", "1-3,3-4", "1, 2", "10000", 5):
            profile = copy.deepcopy(self.profile)
            profile["nodes"][0]["cpuset_cpus"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config.validate(profile)

    def test_explicit_cpu_set_preflight_and_docker_readback(self):
        self.profile["nodes"][0]["cpuset_cpus"] = "5-6"
        config.validate(self.profile)
        cache = Path.home() / ".cache/huggingface"
        model = server.model_path(self.profile, cache)
        image_id = config.selected_image(self.profile)

        def run(*args):
            if args[:3] == ("docker", "image", "inspect"):
                return json.dumps(
                    [
                        {
                            "Id": image_id,
                            "Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]},
                        }
                    ]
                )
            raise AssertionError(args)

        with (
            patch.object(server, "read_json") as read_json,
            patch.object(server.host, "snapshot_from_state", return_value=model),
            patch.object(server.host, "fabric_checks", return_value={}),
            patch.object(server.host, "run", side_effect=run),
            patch.object(server.host, "running_containers", return_value=[]),
            patch.object(
                server.os, "sched_getaffinity", return_value={5, 6}, create=True
            ),
        ):
            read_json.return_value = {
                "text_config": {"num_hidden_layers": server.MODEL_LAYERS}
            }
            result = server.preflight(
                self.profile, ROOT / "state/server.toml", 0, check_memory=False
            )
            self.assertIs(result["checks"]["cpu_set_available"], True)
            with patch.object(
                server.os, "sched_getaffinity", return_value={5}, create=True
            ):
                denied = server.preflight(
                    self.profile, ROOT / "state/server.toml", 0, check_memory=False
                )
            self.assertIs(denied["checks"]["cpu_set_available"], False)
            self.assertIs(denied["passed"], False)
        with patch.object(server, "inspect_owned") as inspect:
            for actual in ("5-6", "5,6"):
                inspect.return_value = {"HostConfig": {"CpusetCpus": actual}}
                server.verify_cpu_set(self.profile, 0, "test")
            for actual in ("0-19", "", None):
                inspect.return_value = {"HostConfig": {"CpusetCpus": actual}}
                with (
                    self.subTest(actual=actual),
                    self.assertRaisesRegex(ValueError, "CPU set"),
                ):
                    server.verify_cpu_set(self.profile, 0, "test")

    def test_failed_cpu_set_readback_stops_the_new_container(self):
        self.profile["nodes"][0]["cpuset_cpus"] = "5-6"
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            args = SimpleNamespace(rank=0, run_id=None, config=Path("server.toml"))
            with (
                patch.object(server, "RECORDS", temporary_root / "records"),
                patch.object(server, "STATE", temporary_root / "state"),
                patch.object(
                    server, "state_path", return_value=temporary_root / "rank.json"
                ),
                patch.object(server, "command", return_value=["docker", "run"]),
                patch.object(
                    server, "verify_cpu_set", side_effect=OSError("inspect failed")
                ),
                patch.object(
                    server.host, "run", side_effect=["created", "stopped"]
                ) as run,
            ):
                with self.assertRaisesRegex(OSError, "inspect failed"):
                    server.start_rank(None, args, self.profile, {"passed": True})
            self.assertEqual(run.call_args_list[-1].args[:2], ("docker", "stop"))
            self.assertFalse((temporary_root / "rank.json").exists())

    def test_the_supervision_banner_names_every_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            args = SimpleNamespace(rank=0, run_id=None, config=Path("server.toml"))
            out = io.StringIO()
            with (
                patch.object(server, "RECORDS", temporary_root / "records"),
                patch.object(server, "STATE", temporary_root / "state"),
                patch.object(
                    server, "state_path", return_value=temporary_root / "rank.json"
                ),
                patch.object(server, "command", return_value=["docker", "run"]),
                patch.object(server, "verify_cpu_set"),
                patch.object(server.host, "run", return_value="created"),
                patch.object(server, "supervise") as supervise,
                contextlib.redirect_stdout(out),
            ):
                server.start_rank(None, args, self.profile, {"passed": True})
            supervise.assert_called_once()
        banner = out.getvalue().splitlines()[-1]
        self.assertTrue(banner.startswith("Supervising in foreground"), banner)
        # supervise's stop reasons: memory-reserve, run-deadline, engine-stall.
        for stop in ("Ctrl+C", "low memory", "deadline", "engine stall"):
            self.assertIn(stop, banner)

    def test_jit_caches_live_in_the_mounted_runtime_cache(self):
        for rank in (0, 1):
            env = config.environment(self.profile, rank)
            self.assertEqual(env["TRITON_CACHE_DIR"], "/root/.cache/triton")
            self.assertEqual(env["TILELANG_CACHE_DIR"], "/root/.cache/tilelang")
            self.assertEqual(
                env["TORCHINDUCTOR_CACHE_DIR"], "/root/.cache/torchinductor"
            )

    def test_autotuned_kernel_choices_are_kept_with_the_triton_cache(self):
        # Without this every launch tunes again, and one KDA kernel's pick decides
        # which of two numerical states the launch computes in.
        for rank in (0, 1):
            env = config.environment(self.profile, rank)
            self.assertEqual(env["TRITON_CACHE_AUTOTUNING"], "1")
            self.assertTrue(env["TRITON_CACHE_DIR"].startswith("/root/.cache/"))

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
        self.profile["runtime"]["decode_graphs"] = True
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
        self.profile["runtime"]["decode_graphs"] = True
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

    def test_nccl_channels_is_optional_and_pins_both_bounds(self):
        # The template pins 8, measured against NCCL's own 64 on the reference pair.
        distributed = config.load(ROOT / "examples/server.example.toml")
        self.assertEqual(distributed["runtime"]["nccl_channels"], 8)
        for rank in (0, 1):
            env = config.environment(distributed, rank)
            self.assertEqual(env["NCCL_MIN_NCHANNELS"], "8")
            self.assertEqual(env["NCCL_MAX_NCHANNELS"], "8")
        # Omitted: NCCL chooses, so profiles written before 1.3.1 keep their fingerprint.
        self.profile["runtime"].pop("nccl_channels")
        config.validate(self.profile)
        for rank in (0, 1):
            env = config.environment(self.profile, rank)
            self.assertNotIn("NCCL_MIN_NCHANNELS", env)
            self.assertNotIn("NCCL_MAX_NCHANNELS", env)
        for bad in (0, -1, 1.5, "8", True, None):
            profile = copy.deepcopy(self.profile)
            profile["runtime"]["nccl_channels"] = bad
            with self.assertRaises(ValueError):
                config.validate(profile)

    def test_canonical_moe_order_is_on_in_the_template_and_optional(self):
        distributed = config.load(ROOT / "examples/server.example.toml")
        self.assertIs(distributed["runtime"]["canonical_moe_order"], True)
        self.assertEqual(
            config.environment(distributed, 0)["GLM53_CANONICAL_MOE_ORDER"], "1"
        )
        image = {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
        self.assertIs(
            config.image_capability_checks(distributed, image)["moe_order_support"],
            False,
        )
        # Marker 1 is shared by the image whose sort mis-sized its buffer, so a new
        # launch needs 2. A switch prepares and, if it fails, restarts the running
        # pair with the checkout that switches away from it: that path keeps 1.
        for marker, recovery, expected in (
            ("GLM53_MOE_ORDER_API=2", False, True),
            ("GLM53_MOE_ORDER_API=2", True, True),
            ("GLM53_MOE_ORDER_API=1", False, False),
            ("GLM53_MOE_ORDER_API=1", True, True),
        ):
            env = ["GLM53_REFERENCE_ATTENTION=1", marker]
            self.assertIs(
                config.image_capability_checks(
                    distributed, {"Config": {"Env": env}}, recovery=recovery
                )["moe_order_support"],
                expected,
            )
        self.assertEqual(
            config.capability_warnings(
                distributed,
                {"Config": {"Env": ["GLM53_MOE_ORDER_API=1"]}},
                recovery=True,
            ),
            ["moe_order_marker_1_accepted_for_recovery"],
        )
        for env, recovery in (
            (["GLM53_MOE_ORDER_API=2"], True),
            (["GLM53_MOE_ORDER_API=1"], False),
        ):
            self.assertEqual(
                config.capability_warnings(
                    distributed, {"Config": {"Env": env}}, recovery=recovery
                ),
                [],
            )
        # Off is an explicit comparison arm and needs no support from the image.
        self.profile["runtime"]["canonical_moe_order"] = False
        config.validate(self.profile)
        self.assertEqual(
            config.environment(self.profile, 1)["GLM53_CANONICAL_MOE_ORDER"], "0"
        )
        self.assertNotIn(
            "moe_order_support",
            config.image_capability_checks(
                self.profile, {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
            ),
        )
        # Omitted: the image decides, so profiles written before this key keep
        # their fingerprint and an older image is not refused.
        self.profile["runtime"].pop("canonical_moe_order")
        config.validate(self.profile)
        self.assertNotIn(
            "GLM53_CANONICAL_MOE_ORDER", config.environment(self.profile, 0)
        )
        for bad in (1, 0, "true", None):
            profile = copy.deepcopy(self.profile)
            profile["runtime"]["canonical_moe_order"] = bad
            with self.assertRaises(ValueError):
                config.validate(profile)

    def test_stable_indexer_topk_is_on_in_the_template_and_optional(self):
        distributed = config.load(ROOT / "examples/server.example.toml")
        self.assertIs(distributed["runtime"]["stable_indexer_topk"], True)
        self.assertEqual(
            config.environment(distributed, 0)["GLM53_STABLE_INDEXER_TOPK"], "1"
        )
        image = {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
        self.assertIs(
            config.image_capability_checks(distributed, image)["indexer_topk_support"],
            False,
        )
        image["Config"]["Env"].append("GLM53_INDEXER_TOPK_API=1")
        self.assertIs(
            config.image_capability_checks(distributed, image)["indexer_topk_support"],
            True,
        )
        # Off is the comparison arm: the kernels alone, which any image has.
        self.profile["runtime"]["stable_indexer_topk"] = False
        config.validate(self.profile)
        self.assertEqual(
            config.environment(self.profile, 1)["GLM53_STABLE_INDEXER_TOPK"], "0"
        )
        self.assertNotIn(
            "indexer_topk_support",
            config.image_capability_checks(
                self.profile, {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
            ),
        )
        # Omitted: the image decides, and an earlier profile keeps its fingerprint.
        self.profile["runtime"].pop("stable_indexer_topk")
        config.validate(self.profile)
        self.assertNotIn(
            "GLM53_STABLE_INDEXER_TOPK", config.environment(self.profile, 0)
        )
        for bad in (1, 0, "true", None):
            profile = copy.deepcopy(self.profile)
            profile["runtime"]["stable_indexer_topk"] = bad
            with self.assertRaises(ValueError):
                config.validate(profile)

    def test_inductor_deterministic_is_optional_and_sets_the_torch_switch(self):
        # 2026-09-24: the replicated indexer's compiled key norm picked its
        # config by timing on each rank, per launch; two picks differ in bits
        # and one rank's key forked completions (launch state 2). Inductor's
        # deterministic mode selects that config without timing, on every rank.
        # On in both templates from 1.12.0; an earlier profile without the key
        # keeps its fingerprint and the timed choice.
        for name in ("server.example.toml", "server.axl.example.toml"):
            template = config.load(ROOT / "examples" / name)
            self.assertIs(template["runtime"]["inductor_deterministic"], True, name)
            self.assertEqual(
                config.environment(template, 1)["TORCHINDUCTOR_DETERMINISTIC"], "1"
            )
        self.profile["runtime"]["inductor_deterministic"] = True
        config.validate(self.profile)
        for rank in (0, 1):
            env = config.environment(self.profile, rank)
            self.assertEqual(env["TORCHINDUCTOR_DETERMINISTIC"], "1")
            # Graphs compiled without the mode are restored from the cache with
            # their timed candidates and cache paths (Det1, 2026-09-24): the mode
            # compiles into a cache of its own, and the other stays as it was.
            self.assertEqual(
                env["TORCHINDUCTOR_CACHE_DIR"],
                "/root/.cache/torchinductor-deterministic",
            )
        # torch reads only "1", so the comparison arm is the variable left unset.
        self.profile["runtime"]["inductor_deterministic"] = False
        config.validate(self.profile)
        self.assertNotIn(
            "TORCHINDUCTOR_DETERMINISTIC", config.environment(self.profile, 0)
        )
        self.assertEqual(
            config.environment(self.profile, 0)["TORCHINDUCTOR_CACHE_DIR"],
            "/root/.cache/torchinductor",
        )
        # No image support is needed: the pinned torch reads the variable itself.
        self.assertNotIn(
            "inductor_deterministic",
            json.dumps(
                config.image_capability_checks(
                    self.profile, {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
                )
            ),
        )
        # Adding the key moves the fingerprint; an earlier profile keeps its own.
        with_key = config.fingerprint(self.profile)
        self.profile["runtime"].pop("inductor_deterministic")
        self.assertNotEqual(config.fingerprint(self.profile), with_key)
        for bad in (1, 0, "true", None):
            profile = copy.deepcopy(self.profile)
            profile["runtime"]["inductor_deterministic"] = bad
            with self.assertRaises(ValueError):
                config.validate(profile)

    def test_prefix_page_dedup_is_off_unless_set_and_needs_the_image_marker(self):
        distributed = config.load(ROOT / "examples/server.example.toml")
        self.assertNotIn("prefix_page_dedup", distributed["runtime"])
        self.assertNotIn("GLM53_PREFIX_PAGE_DEDUP", config.environment(distributed, 0))
        self.assertNotIn(
            "prefix_dedup_support",
            config.image_capability_checks(
                distributed, {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
            ),
        )
        self.profile["runtime"]["prefix_page_dedup"] = True
        config.validate(self.profile)
        self.assertEqual(
            config.environment(self.profile, 1)["GLM53_PREFIX_PAGE_DEDUP"], "1"
        )
        image = {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
        self.assertIs(
            config.image_capability_checks(self.profile, image)["prefix_dedup_support"],
            False,
        )
        image["Config"]["Env"].append("GLM53_PREFIX_DEDUP_API=1")
        self.assertIs(
            config.image_capability_checks(self.profile, image)["prefix_dedup_support"],
            True,
        )
        self.profile["runtime"]["prefix_page_dedup"] = False
        config.validate(self.profile)
        self.assertEqual(
            config.environment(self.profile, 1)["GLM53_PREFIX_PAGE_DEDUP"], "0"
        )
        for bad in (1, 0, "true", None):
            profile = copy.deepcopy(self.profile)
            profile["runtime"]["prefix_page_dedup"] = bad
            with self.assertRaises(ValueError):
                config.validate(profile)

    def test_retired_mla_decode_cpb_is_gone_from_the_examples(self):
        for name in ("server.example.toml", "server.axl.example.toml"):
            example = config.load(ROOT / "examples" / name)
            self.assertNotIn("mla_decode_cpb", example["runtime"])
            self.assertNotIn("GLM53_MLA_DECODE_CPB", config.environment(example, 0))

    def test_a_profile_that_still_carries_mla_decode_cpb_is_refused(self):
        # Retired in 1.16.0 (it never ran in serving), removed in 1.18.0. The sentence
        # comes before the schema's "Unknown/missing settings", whatever the value.
        for value in (True, False, 1, 0, "true", None):
            profile = copy.deepcopy(self.profile)
            profile["runtime"]["mla_decode_cpb"] = value
            with self.assertRaisesRegex(
                ValueError,
                r"^runtime\.mla_decode_cpb was retired in 1\.16\.0 and removed in "
                r"1\.18\.0; delete the key from the profile$",
            ):
                config.validate(profile)

    def test_no_capability_row_or_warning_names_mla_decode_cpb(self):
        image = {"Config": {"Env": ["GLM53_MOE_ORDER_API=1"]}}
        self.profile["runtime"]["canonical_moe_order"] = True
        for recovery in (False, True):
            self.assertNotIn(
                "mla_decode_cpb_support",
                config.image_capability_checks(self.profile, image, recovery=recovery),
            )
            warnings = config.capability_warnings(
                self.profile, image, recovery=recovery
            )
            self.assertNotIn("mla_decode_cpb_retired", warnings)
        self.assertEqual(warnings, ["moe_order_marker_1_accepted_for_recovery"])

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
        # One sequence with MTP and prefix caching was qualified on the MTP fixture
        # (records/20260918-stage1-graph); batching was not.
        p = copy.deepcopy(self.profile)
        p["runtime"]["decode_graphs"] = True
        p["mtp"]["enabled"] = True
        p["cache"]["prefix_caching"] = True
        config.validate(p)
        p["context"]["max_num_seqs"] = 2
        with self.assertRaisesRegex(ValueError, "Graph"):
            config.validate(p)

    def test_speculative_depths_one_to_five_are_accepted(self):
        # k=2, 4 and 5 are launchable for the depth sweep; the measured choice
        # stays in the template (RELEASE 1.6.0 Stage 2).
        for depth in (1, 2, 3, 4, 5):
            p = copy.deepcopy(self.profile)
            p["mtp"]["enabled"] = True
            p["mtp"]["num_speculative_tokens"] = depth
            config.validate(p)
            args = config.serve_args(p, 0, "/hf/model")
            spec = json.loads(args[args.index("--speculative-config") + 1])
            self.assertEqual(spec["num_speculative_tokens"], depth)
        for depth in (0, 6, True, 2.5):
            p = copy.deepcopy(self.profile)
            p["mtp"]["num_speculative_tokens"] = depth
            with self.assertRaises(ValueError):
                config.validate(p)

    def test_decode_graphs_is_the_positive_switch(self):
        # One-stop true/false in the profile; enforce_eager stays readable as the
        # legacy spelling and may not contradict it.
        del self.profile["runtime"]["decode_graphs"]
        config.validate(self.profile)
        self.assertIn(
            "--enforce-eager", config.serve_args(self.profile, 0, "/hf/model")
        )
        self.profile["runtime"]["enforce_eager"] = False  # the earlier spelling
        config.validate(self.profile)
        self.assertNotIn(
            "--enforce-eager", config.serve_args(self.profile, 0, "/hf/model")
        )
        self.profile["runtime"]["decode_graphs"] = True
        config.validate(self.profile)
        args = config.serve_args(self.profile, 0, "/hf/model")
        self.assertNotIn("--enforce-eager", args)
        self.assertIn("--compilation-config", args)
        self.profile["runtime"]["enforce_eager"] = True
        with self.assertRaisesRegex(ValueError, "decode_graphs"):
            config.validate(self.profile)
        self.profile["runtime"]["decode_graphs"] = "yes"
        del self.profile["runtime"]["enforce_eager"]
        with self.assertRaisesRegex(ValueError, "decode_graphs"):
            config.validate(self.profile)

    def test_graph_capture_size_follows_the_speculative_depth(self):
        # The pinned runtime rounds decode capture sizes up to a multiple of
        # num_speculative_tokens + 1 and rejects [1] outright with MTP on.
        self.profile["runtime"]["decode_graphs"] = True
        self.profile["mtp"]["enabled"] = True
        self.profile["mtp"]["num_speculative_tokens"] = 3
        config.validate(self.profile)
        args = config.serve_args(self.profile, 0, "/hf/model")
        self.assertEqual(
            json.loads(args[args.index("--compilation-config") + 1])[
                "cudagraph_capture_sizes"
            ],
            [4],
        )
        self.profile["mtp"]["enabled"] = False
        args = config.serve_args(self.profile, 0, "/hf/model")
        self.assertEqual(
            json.loads(args[args.index("--compilation-config") + 1])[
                "cudagraph_capture_sizes"
            ],
            [1],
        )

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

    def test_memory_observations_are_recorded_but_only_reserve_stops(self):
        self.profile["resources"]["run_seconds"] = 0
        self.profile["resources"]["stall_seconds"] = 0
        reserve = self.profile["resources"]["reserve_gib"]
        opened = mock_open()
        with (
            patch("pathlib.Path.open", opened),
            patch.object(
                server,
                "inspect_owned",
                return_value={"Id": "abc", "State": {"Running": True}},
            ),
            patch.object(server, "available_gib", side_effect=[100, reserve - 1]),
            patch.object(
                server.host,
                "memory_sample",
                return_value={"mem_free_gib": 0.01, "free_2mib_gib": 0.0},
            ),
            patch.object(
                server.host,
                "container_memory_sample",
                return_value={"container_cgroup_gib": 7.1, "container_rss_gib": 3.7},
            ) as container,
            patch.object(server.time, "sleep") as sleep,
            patch.object(server, "write_json") as write,
            patch.object(server.host, "run"),
            patch.object(server.subprocess, "run"),
        ):
            server.supervise(self.profile, "owned", Path("record"), 0)
        lines = [json.loads(call.args[0]) for call in opened().write.call_args_list]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(line["mem_free_gib"], 0.01)
            self.assertEqual(line["free_2mib_gib"], 0.0)
            # Flat cgroup/RSS beside a falling MemAvailable is the device side.
            self.assertEqual(line["container_cgroup_gib"], 7.1)
            self.assertEqual(line["container_rss_gib"], 3.7)
        container.assert_called_with("abc")
        # A tiny MemFree alone never stops the rank; MemAvailable still does.
        self.assertEqual(sleep.call_count, 1)
        write.assert_any_call(
            Path("record/stop-reason.json"), {"reason": "memory-reserve"}
        )

    def test_preflight_refuses_foreign_gpu_containers_with_or_without_memory(self):
        cache = Path.home() / ".cache/huggingface"
        profile = self.profile
        model = server.model_path(profile, cache)
        image_id = config.selected_image(profile)
        gpu = {"DeviceRequests": [{"Capabilities": [["gpu"]]}]}
        owned = {
            "Name": "/glm53-startup-r0-old",
            "Config": {"Labels": {server.LABEL: "old-fingerprint"}},
            "HostConfig": gpu,
        }
        foreign = [
            owned,
            {"Name": "/other-gpu", "Config": {"Labels": {}}, "HostConfig": gpu},
        ]

        def run(*args):
            if args[:3] == ("docker", "image", "inspect"):
                env = ["GLM53_REFERENCE_ATTENTION=1"]
                return json.dumps([{"Id": image_id, "Config": {"Env": env}}])
            raise AssertionError(args)

        with (
            patch.object(server, "read_json") as read_json,
            patch.object(server.host, "snapshot_from_state", return_value=model),
            patch.object(server.host, "fabric_checks", return_value={}),
            patch.object(server.host, "run", side_effect=run),
            patch.object(server.host, "running_containers", return_value=[owned]),
        ):
            read_json.return_value = {
                "text_config": {"num_hidden_layers": server.MODEL_LAYERS}
            }
            result = server.preflight(
                profile, ROOT / "state/server.toml", 0, check_memory=False
            )
        self.assertIs(result["checks"]["exclusive_gpu"], True)
        self.assertEqual(result["foreign_gpu_containers"], [])

        for check_memory in (True, False):
            with self.subTest(check_memory=check_memory):
                with (
                    patch.object(server, "read_json") as read_json,
                    patch.object(
                        server.host, "snapshot_from_state", return_value=model
                    ),
                    patch.object(server.host, "fabric_checks", return_value={}),
                    patch.object(server.host, "run", side_effect=run),
                    patch.object(
                        server.host, "running_containers", return_value=foreign
                    ),
                    patch.object(server, "available_gib", return_value=100),
                ):
                    read_json.return_value = {
                        "text_config": {"num_hidden_layers": server.MODEL_LAYERS}
                    }
                    result = server.preflight(
                        profile,
                        ROOT / "state/server.toml",
                        0,
                        check_memory=check_memory,
                    )
                self.assertIs(result["checks"]["exclusive_gpu"], False)
                self.assertEqual(result["foreign_gpu_containers"], ["other-gpu"])
                self.assertIs(result["passed"], False)

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

    def test_dev_mode_is_what_turns_on_the_server_dev_mode(self):
        self.assertFalse(config.dev_mode(self.profile))
        self.assertNotIn("VLLM_SERVER_DEV_MODE", config.environment(self.profile, 0))
        for section, key in (
            ("api", "dev_endpoints"),
            ("lpa", "enabled"),
            ("validation", "component_worker"),
            ("validation", "expert_worker"),
        ):
            with self.subTest(key=key):
                profile = copy.deepcopy(self.profile)
                profile[section][key] = True
                self.assertTrue(config.dev_mode(profile))
                self.assertEqual(
                    config.environment(profile, 0)["VLLM_SERVER_DEV_MODE"], "1"
                )

    def test_dev_endpoints_gate_reset_and_capacity_layout(self):
        self.profile["lpa"]["enabled"] = False
        self.assertFalse(config.dev_mode(self.profile))
        self.profile["api"]["dev_endpoints"] = True
        self.assertTrue(config.dev_mode(self.profile))
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

    def test_the_speculative_examples_are_what_the_launcher_passes(self):
        # docs/README.md names the two files as the MTP configuration examples.
        for depth in (1, 3):
            text = (ROOT / f"examples/speculative.mtp{depth}.json").read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                json.loads(text, object_pairs_hook=list),
                list(config.speculative_config(depth).items()),
            )
        self.profile["mtp"]["enabled"] = True
        args = config.serve_args(self.profile, 0, "/hf/mtp-view")
        self.assertEqual(
            args[args.index("--speculative-config") + 1],
            json.dumps(
                config.speculative_config(self.profile["mtp"]["num_speculative_tokens"])
            ),
        )

    def test_speculative_config_carries_each_depth_in_a_fresh_dict(self):
        for depth in (1, 2, 3, 4, 5):
            spec = config.speculative_config(depth)
            # Key order is what serve_args dumps and records store.
            self.assertEqual(
                list(spec), ["method", "num_speculative_tokens", "moe_backend"]
            )
            self.assertEqual(spec["num_speculative_tokens"], depth)
        first = config.speculative_config(3)
        first["num_speculative_tokens"] = 9
        first["extra"] = True
        self.assertEqual(
            config.speculative_config(3),
            {"method": "mtp", "num_speculative_tokens": 3, "moe_backend": "triton"},
        )

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

    def derived(self, root, real=False):
        overlay = root / "kda-quant.py"
        overlay.write_text("x = 1  # kda-quant-overlay\n", encoding="utf-8")
        # The hosts are Linux; validation takes POSIX paths, the checks read a real file.
        return {
            "path": "/srv/weights-g",
            "requant_target": "g",
            "overlays": [
                {
                    "target": "kda.py",
                    "source": str(overlay) if real else "/srv/kda-quant.py",
                    "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
                    "base_sha256": "a" * 64,
                    "marker": "kda-quant-overlay",
                }
            ],
        }

    def test_derived_checkpoint_is_optional_and_strictly_shaped(self):
        before = config.fingerprint(self.profile)
        config.validate(self.profile)
        with tempfile.TemporaryDirectory() as tmp:
            good = self.derived(Path(tmp).resolve())
            self.profile["runtime"]["derived_checkpoint"] = good
            config.validate(self.profile)
            self.assertNotEqual(config.fingerprint(self.profile), before)
            for key, bad in (
                ("path", "relative/dir"),
                ("path", 3),
                ("requant_target", ""),
                ("overlays", []),
                ("overlays", [{**good["overlays"][0], "target": "../kda.py"}]),
                ("overlays", [{**good["overlays"][0], "sha256": "xyz"}]),
                ("overlays", [{**good["overlays"][0], "extra": 1}]),
                ("overlays", [good["overlays"][0], good["overlays"][0]]),
            ):
                profile = copy.deepcopy(self.profile)
                profile["runtime"]["derived_checkpoint"][key] = bad
                with self.assertRaises(ValueError, msg=(key, bad)):
                    config.validate(profile)
            profile = copy.deepcopy(self.profile)
            profile["runtime"]["derived_checkpoint"]["unknown"] = 1
            with self.assertRaises(ValueError):
                config.validate(profile)
        self.profile["runtime"].pop("derived_checkpoint")
        self.assertEqual(config.fingerprint(self.profile), before)

    def test_derived_checkpoint_enabled_false_serves_the_pinned_snapshot(self):
        # The table can stay in the profile with its switch off (one-stop
        # true/false); only a boolean is accepted.
        self.profile["mtp"]["enabled"] = True
        with tempfile.TemporaryDirectory() as tmp:
            derived = self.derived(Path(tmp).resolve())
            derived["enabled"] = False
            self.profile["runtime"]["derived_checkpoint"] = derived
            config.validate(self.profile)
            self.assertIsNone(config.derived_checkpoint(self.profile))
            args = server.command(
                self.profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
            )
            self.assertNotIn("/derived", args)
            derived["enabled"] = True
            self.assertEqual(config.derived_checkpoint(self.profile), derived)
            derived["enabled"] = "no"
            with self.assertRaisesRegex(ValueError, "enabled"):
                config.validate(self.profile)

    def test_command_serves_a_derived_checkpoint_with_its_overlays(self):
        self.profile["mtp"]["enabled"] = True
        with tempfile.TemporaryDirectory() as tmp:
            derived = self.derived(Path(tmp).resolve())
            self.profile["runtime"]["derived_checkpoint"] = derived
            args = server.command(
                self.profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
            )
        self.assertIn(derived["path"] + ":/derived:ro", args)
        self.assertIn("/derived", args)
        self.assertIn(
            derived["overlays"][0]["source"]
            + ":"
            + server.VLLM_MODEL_DIR
            + "/kda.py:ro",
            args,
        )
        self.assertFalse([a for a in args if "glm53-mtp-compatible" in a])

    def test_derived_checks_fail_closed_on_every_mismatch(self):
        metadata = {
            "quantization_config": {
                "quant_algo": "MIXED_PRECISION",
                "producer": {"requant_target": "g"},
                "quantized_layers": {
                    "model.language_model.layers.0.self_attn.o_proj": {}
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            derived = self.derived(Path(tmp).resolve(), real=True)
            self.profile["runtime"]["derived_checkpoint"] = derived
            self.profile["mtp"]["enabled"] = True

            def checks(metadata=metadata, base="a" * 64):
                with patch.object(
                    server.host, "run", return_value=base + "  /x/kda.py\n"
                ) as run:
                    result = server.derived_checks(self.profile, metadata)
                self.assertIn("--network", run.call_args.args)
                return result

            self.assertEqual(
                checks(),
                {
                    "derived_checkpoint": True,
                    "derived_mtp_draft_unquantized": True,
                    "derived_overlays": True,
                },
            )
            self.assertIs(checks(base="b" * 64)["derived_overlays"], False)
            wrong = copy.deepcopy(metadata)
            wrong["quantization_config"]["producer"]["requant_target"] = "h"
            self.assertIs(checks(wrong)["derived_checkpoint"], False)
            wrong = copy.deepcopy(metadata)
            wrong["quantization_config"]["quant_algo"] = "NVFP4"
            self.assertIs(checks(wrong)["derived_checkpoint"], False)
            wrong = copy.deepcopy(metadata)
            wrong["quantization_config"]["quantized_layers"][
                f"model.language_model.layers.{server.MODEL_LAYERS}.mlp.experts"
            ] = {}
            self.assertIs(checks(wrong)["derived_mtp_draft_unquantized"], False)
            Path(derived["overlays"][0]["source"]).write_text("x = 2\n")
            self.assertIs(checks()["derived_overlays"], False)
        self.profile["runtime"].pop("derived_checkpoint")
        self.assertEqual(server.derived_checks(self.profile, metadata), {})

    def test_preflight_runs_the_derived_checks_beside_the_pinned_snapshot(self):
        cache = Path.home() / ".cache/huggingface"
        self.profile["mtp"]["enabled"] = True
        snapshot = server.model_path(
            {**self.profile, "mtp": {**self.profile["mtp"], "enabled": False}}, cache
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.profile["runtime"]["derived_checkpoint"] = self.derived(
                Path(tmp).resolve(), real=True
            )
            image_id = config.selected_image(self.profile)

            def run(*args):
                if args[:3] == ("docker", "image", "inspect"):
                    env = ["GLM53_REFERENCE_ATTENTION=1"]
                    return json.dumps([{"Id": image_id, "Config": {"Env": env}}])
                if args[:2] == ("docker", "run"):
                    return "a" * 64 + "  kda.py\n"
                raise AssertionError(args)

            with (
                patch.object(server, "read_json") as read_json,
                patch.object(server.host, "snapshot_from_state", return_value=snapshot),
                patch.object(server.host, "fabric_checks", return_value={}),
                patch.object(server.host, "run", side_effect=run),
                patch.object(server.host, "running_containers", return_value=[]),
            ):
                read_json.return_value = {
                    "text_config": {"num_hidden_layers": server.MODEL_LAYERS},
                    "quantization_config": {
                        "quant_algo": "MIXED_PRECISION",
                        "producer": {"requant_target": "g"},
                        "quantized_layers": {},
                    },
                }
                result = server.preflight(
                    self.profile, ROOT / "state/server.toml", 0, check_memory=False
                )
        for key in (
            "derived_checkpoint",
            "derived_mtp_draft_unquantized",
            "derived_overlays",
            "full_model",
        ):
            self.assertIs(result["checks"][key], True, key)
        self.assertNotIn("mtp_view", result["checks"])

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
            checks = config.image_capability_checks(
                self.profile, {"Config": image_config}
            )
            self.assertEqual(checks, {"reference_attention": False})
        self.profile["lpa"]["enabled"] = True
        self.profile["cache"]["prefix_caching"] = True
        self.profile["cache"]["fused_unpack"] = True
        env = ["GLM53_REFERENCE_ATTENTION=1", "GLM53_LPA_API=2", "GLM53_APC_LPA_API=1"]
        checks = config.image_capability_checks(self.profile, {"Config": {"Env": env}})
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
            ("mtp", "num_speculative_tokens", 0),
            ("mtp", "num_speculative_tokens", 6),
            ("mtp", "num_speculative_tokens", 2.5),
            ("lpa", "cut", 45),
            ("lpa", "tail", 0),
            ("lpa", "break_even_tokens", -1),
            ("runtime", "decode_graphs", True),
        ]:
            with self.subTest(section=section, key=key):
                p = copy.deepcopy(self.profile)
                p["lpa"]["enabled"] = True
                p[section][key] = value
                with self.assertRaises(ValueError):
                    config.validate(p)

    def test_lpa_with_mtp_launches_only_the_depths_its_workers_accept(self):
        # The LPA and APC/LPA workers refuse the others on every request.
        self.profile["lpa"]["enabled"] = True
        self.profile["mtp"]["enabled"] = True
        for depth in (1, 2, 3, 4, 5):
            with self.subTest(depth=depth):
                self.profile["mtp"]["num_speculative_tokens"] = depth
                if depth <= 3:
                    config.validate(self.profile)
                    continue
                with self.assertRaises(ValueError):
                    config.validate(self.profile)

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


class ReferenceImageMarkerTests(unittest.TestCase):
    """The markers preflight requires are the ones the reference build bakes."""

    # The reference build's ENV lines no capability check requires, and why.
    UNCHECKED_MARKERS = {
        "GLM53_KPOOL_SEED_STRIDE=1": "build-patch record, no reader (CHANGELOG 1.13.0)",
        "GLM53_SLOT_MAPPING_GUARD=1": "build-patch record, no reader (CHANGELOG 1.7.0)",
        "GLM53_CANONICAL_CANDIDATES=1": "switch default read by candidate_order",
        "GLM53_CANONICAL_MOE_ORDER=1": "switch default read by moe_token_order",
        "GLM53_STABLE_INDEXER_TOPK=1": "switch default read by stable_topk",
    }

    def dockerfile_env(self):
        dockerfile = (ROOT / "docker/Dockerfile.reference").read_text(encoding="utf-8")
        return re.findall(r"^ENV (GLM53_\w+=\S+)$", dockerfile, re.MULTILINE)

    def enabled_profile(self):
        profile = config.load(ROOT / "examples/server.example.toml")
        # Every feature a check is conditional on, turned on.
        profile["runtime"].update(
            pipeline_parallel_size=2,
            expert_parallel=True,
            decode_graphs=True,
            canonical_moe_order=True,
            stable_indexer_topk=True,
            prefix_page_dedup=True,
            fa2_attention=True,
        )
        profile["validation"]["component_worker"] = True
        profile["cache"].update(fused_unpack=True, prefix_caching=True)
        profile["lpa"]["enabled"] = True
        return profile

    def test_every_reference_marker_is_required_or_named_unchecked(self):
        env, profile = self.dockerfile_env(), self.enabled_profile()
        # A marker is required when the checks fail without it (new launch).
        unchecked = {
            marker
            for marker in env
            if all(
                config.image_capability_checks(
                    profile, {"Config": {"Env": [m for m in env if m != marker]}}
                ).values()
            )
        }
        self.assertEqual(unchecked, set(self.UNCHECKED_MARKERS))

    def test_the_reference_dockerfile_satisfies_every_capability_check(self):
        env, profile = self.dockerfile_env(), self.enabled_profile()
        # A new launch: the stricter requirement.
        checks = config.image_capability_checks(profile, {"Config": {"Env": env}})
        # A check added without turning its feature on above fails this list.
        self.assertEqual(
            list(checks),
            [
                "pipeline_support",
                "expert_parallel_support",
                "component_worker",
                "fused_unpack_support",
                "decode_graph_support",
                "async_index_check_support",
                "lpa_worker",
                "apc_lpa_support",
                "reference_attention",
                "moe_order_support",
                "indexer_topk_support",
                "prefix_dedup_support",
                "fa2_attention_support",
            ],
        )
        self.assertEqual([key for key, ok in checks.items() if not ok], [])


class HostFactOwnerTests(unittest.TestCase):
    """Names and paths the launcher, the coordinator and the tools must agree on."""

    def test_container_name_is_the_startup_name_per_rank_and_run(self):
        # A coordinator that spelled it differently could not confirm readiness.
        self.assertEqual(server.container_name(1, "abc"), "glm53-startup-r1-abc")

    def test_rank_state_lives_in_the_state_directory(self):
        self.assertEqual(server.state_path(1), STATE / "startup-rank1.json")

    def test_hf_cache_is_read_from_the_home_directory_at_call_time(self):
        with patch.object(server.Path, "home", return_value=Path("/elsewhere")):
            self.assertEqual(server.hf_cache(), Path("/elsewhere/.cache/huggingface"))

    def test_every_command_defaults_to_the_operator_profile(self):
        self.assertEqual(DEFAULT_PROFILE, STATE / "server.toml")
        self.assertEqual(server.parser().get_default("config"), DEFAULT_PROFILE)


class HeadClientTests(unittest.TestCase):
    """The one spelling of how a host-local client reaches the running head."""

    PROFILE = {"api": {"port": 8123}}

    def test_the_head_is_reached_on_the_loopback_at_the_profile_port(self):
        self.assertEqual(server.api_origin(self.PROFILE), "http://127.0.0.1:8123")

    def test_a_reset_that_is_not_acknowledged_fails(self):
        with patch.object(server, "post", return_value={"success": True}) as post:
            server.reset_prefix_cache(self.PROFILE)
        post.assert_called_once_with(self.PROFILE, "/reset_prefix_cache", {})
        for answer in ({}, {"success": False}):
            with (
                self.subTest(answer=answer),
                patch.object(server, "post", return_value=answer),
                self.assertRaisesRegex(ValueError, "not acknowledged"),
            ):
                server.reset_prefix_cache(self.PROFILE)

    def test_metrics_are_decoded_as_each_caller_asks(self):
        @contextlib.contextmanager
        def response(origin, path, timeout):
            self.assertEqual((origin, path), ("http://127.0.0.1:8123", "/metrics"))
            seen.append(timeout)
            yield io.BytesIO(b"a \xff")

        seen = []
        with patch.object(server.model_http, "open_response", response):
            self.assertEqual(server.metrics_text(self.PROFILE), "a " + chr(0xFFFD))
            with self.assertRaises(UnicodeDecodeError):
                server.metrics_text(self.PROFILE, timeout=10, errors="strict")
        self.assertEqual(seen, [2, 10])


class RecordedHeadRunTests(unittest.TestCase):
    """warmup, mojibake and agreement: one run on the head, one record each."""

    RUNNERS = {
        "warmup": ("warmup_report", server.warmup_running),
        "mojibake": ("mojibake", server.mojibake_running),
        "agreement": ("agreement", server.agreement_running),
    }

    def setUp(self):
        self.profile = config.load(ROOT / "examples/server.example.toml")
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(server, "RECORDS", self.tmp))
        self.locked = []

        @contextlib.contextmanager
        def lock():
            self.locked.append(True)
            yield
            self.locked.append(False)

        self.enterContext(patch.object(server, "request_lock", lock))

    def head(self, running=True):
        info = {"State": {"Running": running}}
        return patch.object(server, "running_head", return_value=({"name": "c"}, info))

    def job(self, action):
        """Replace what the action runs; the replacement records the lock state."""
        name, _ = self.RUNNERS[action]

        def run(*args):
            self.assertEqual(self.locked, [True])
            return {"passed": True}

        if name == "warmup_report":
            return patch.object(server, name, side_effect=run)
        return patch.object(getattr(server, name), "run", side_effect=run)

    def test_each_writes_its_result_into_a_new_stamped_record(self):
        for action, (_, running) in self.RUNNERS.items():
            with self.subTest(action=action), self.head(), self.job(action):
                result = running(self.profile)
                record = Path(result["record"])
                self.assertEqual(record.parent, self.tmp)
                self.assertRegex(record.name, rf"^\d{{8}}T\d{{12}}Z-{action}-r0$")
                saved = json.loads((record / "result.json").read_text("utf-8"))
                self.assertEqual(saved, result)
                self.assertEqual(list(saved), ["passed", "record"])
            self.locked.clear()

    def test_a_stopped_head_is_refused_before_the_lock_and_the_record(self):
        for action, (_, running) in self.RUNNERS.items():
            with self.subTest(action=action), self.head(running=False):
                with self.assertRaisesRegex(
                    ValueError, f"^{action} requires the running rank 0$"
                ):
                    running(self.profile)
        self.assertEqual(self.locked, [])
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_agreement_compares_with_a_reference_before_naming_the_record(self):
        reference = self.tmp / "reference.json"
        reference.write_text(json.dumps({"texts": []}), encoding="utf-8")
        with (
            self.head(),
            self.job("agreement"),
            patch.object(
                server.agreement, "compare_records", return_value={"same": True}
            ) as compare,
        ):
            result = server.agreement_running(self.profile, reference)
        self.assertEqual(compare.call_args.args[0], {"texts": []})
        self.assertEqual(list(result), ["passed", "reference", "comparison", "record"])
        self.assertEqual(result["reference"], str(reference))

    def test_agreement_refuses_native_lpa_before_reading_the_head(self):
        self.profile["lpa"]["enabled"] = True
        self.profile["cache"]["prefix_caching"] = False
        with (
            patch.object(server, "running_head") as head,
            self.assertRaisesRegex(ValueError, "native LPA"),
        ):
            server.agreement_running(self.profile)
        head.assert_not_called()
        self.assertEqual(self.locked, [])


class RequestLockTests(unittest.TestCase):
    def test_the_lock_is_an_exclusive_nonblocking_flock_on_the_state_file(self):
        calls = []
        fake = SimpleNamespace(
            LOCK_EX=2, LOCK_NB=4, flock=lambda f, flags: calls.append((f.name, flags))
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(server, "STATE", Path(tmp)),
            patch.dict(sys.modules, {"fcntl": fake}),
        ):
            with server.request_lock():
                pass
            self.assertEqual(calls, [(str(Path(tmp) / "startup-request.lock"), 6)])

    @unittest.skipUnless(os.name == "posix", "flock exists on the model host only")
    def test_a_second_client_is_refused_while_the_first_holds_it(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(server, "STATE", Path(tmp)),
        ):
            with server.request_lock():
                with self.assertRaises(BlockingIOError):
                    with server.request_lock():
                        pass
            with server.request_lock():
                pass


if __name__ == "__main__":
    unittest.main()
