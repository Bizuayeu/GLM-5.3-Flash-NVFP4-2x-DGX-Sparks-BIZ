"""The engine settings each fixture runner launches under.

These values -- the seed, the KV budget, the memory share, the block size,
the backends -- are what makes one measurement comparable to the next, and
what AGENTS.md means by preserving existing generation settings. Until now no
test read any of them: they were literals inside a main() that only a GPU
could reach.

Each runner exposes ``engine_kwargs(args)``, and each is required to produce
those settings without importing torch or vLLM. That second part is checked
in a subprocess with both modules blocked, because a test that merely calls
the function proves nothing on a machine that has them installed.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKED = """
import json, sys
for name in ("torch", "vllm"):
    sys.modules[name] = None
from glm53_setup.validation import {module} as runner
args = runner.parser().parse_args({argv})
print(json.dumps(runner.engine_kwargs(args, *{extra}), default=str))
"""

# Every runner that builds an engine. The extra argument is a vLLM constant
# these two need; it is a parameter precisely so the settings stay readable
# without the GPU stack, and 0 stands in for CompilationMode.NONE here.
ENGINE_RUNNERS = {
    "run_fixture": (),
    "run_indexer_fixture": (),
    "run_repeat_trace": (),
    "run_apc_lpa_fixture": (),
    "run_lpa": (0,),
    "run_graph_fixture": (0,),
}


def kwargs_without_gpu(module, argv, extra=()):
    """Build one runner's engine kwargs with torch and vLLM unimportable."""
    script = BLOCKED.format(module=module, argv=repr(argv), extra=repr(list(extra)))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(
            f"{module}.engine_kwargs needs a GPU import:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def argv(**overrides):
    args = ["--fixture", "/fixture", "--output", "/out"]
    for flag, value in overrides.items():
        args += ["--" + flag, str(value)]
    return args


class EveryEngineRunnerTests(unittest.TestCase):
    """What must hold for all of them, whatever else each one measures."""

    def settings(self, module):
        return kwargs_without_gpu(module, argv(), ENGINE_RUNNERS[module])

    def test_none_of_them_needs_a_gpu_stack_to_state_its_settings(self):
        for module in ENGINE_RUNNERS:
            with self.subTest(runner=module):
                self.assertIn("model", self.settings(module))

    def test_the_seed_is_fixed_so_a_repeated_run_is_comparable(self):
        for module in ENGINE_RUNNERS:
            with self.subTest(runner=module):
                self.assertEqual(self.settings(module)["seed"], 42)

    def test_warmup_and_autotune_stay_off_so_timings_mean_something(self):
        for module in ENGINE_RUNNERS:
            with self.subTest(runner=module):
                config = self.settings(module)["kernel_config"]
                for key in (
                    "enable_flashinfer_autotune",
                    "enable_cutedsl_warmup",
                    "enable_jit_warmup",
                ):
                    self.assertFalse(config[key], f"{module}.{key}")

    def test_the_kv_budget_is_stated_rather_than_profiled(self):
        # A fixed byte budget makes vLLM skip startup memory profiling, so the
        # number here is the whole of what the KV cache gets.
        for module in ENGINE_RUNNERS:
            with self.subTest(runner=module):
                self.assertIsInstance(
                    self.settings(module)["kv_cache_memory_bytes"], int
                )

    def test_a_fixture_runner_never_asks_for_more_than_one_gpu(self):
        for module in ENGINE_RUNNERS:
            with self.subTest(runner=module):
                self.assertEqual(self.settings(module)["tensor_parallel_size"], 1)


class FixtureRunnerTests(unittest.TestCase):
    def argv(self, **overrides):
        return argv(**overrides)

    def test_the_engine_settings_are_reachable_without_a_gpu_stack(self):
        kwargs = kwargs_without_gpu("run_fixture", self.argv())
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["tensor_parallel_size"], 1)
        self.assertEqual(kwargs["block_size"], 256)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
        self.assertEqual(kwargs["kv_cache_memory_bytes"], 512 * 1024**2)
        self.assertEqual(kwargs["gpu_memory_utilization"], 0.20)
        self.assertEqual(kwargs["max_num_seqs"], 2)
        self.assertTrue(kwargs["enforce_eager"])
        self.assertFalse(kwargs["enable_prefix_caching"])
        self.assertTrue(kwargs["enable_chunked_prefill"])
        self.assertTrue(kwargs["language_model_only"])
        self.assertEqual(
            kwargs["worker_extension_cls"],
            "glm53_setup.validation.run_fixture.FixtureWorkerExtension",
        )

    def test_the_command_line_selects_the_context_chunk_and_backend(self):
        default = kwargs_without_gpu("run_fixture", self.argv())
        self.assertEqual(default["max_model_len"], 2048)
        self.assertEqual(default["max_num_batched_tokens"], 512)
        self.assertEqual(default["kernel_config"]["moe_backend"], "auto")
        wide = kwargs_without_gpu(
            "run_fixture", self.argv(context=16384, chunk=128, backend="marlin")
        )
        self.assertEqual(wide["max_model_len"], 16384)
        self.assertEqual(wide["max_num_batched_tokens"], 128)
        self.assertEqual(wide["kernel_config"]["moe_backend"], "marlin")
        self.assertEqual(wide["kernel_config"]["linear_backend"], "marlin")

    def test_the_warmup_and_autotune_stay_off_so_timings_are_comparable(self):
        config = kwargs_without_gpu("run_fixture", self.argv())["kernel_config"]
        for key in (
            "enable_flashinfer_autotune",
            "enable_cutedsl_warmup",
            "enable_jit_warmup",
        ):
            with self.subTest(key=key):
                self.assertFalse(config[key])


COMPLETE = {"status": "complete", "all_tensor_bytes_verified": True}
FOUR_LAYER = {"_test_fixture_only": True, "text_config": {"num_hidden_layers": 4}}


class FixtureGateTests(unittest.TestCase):
    """Only a byte-verified four-layer fixture may be run."""

    def test_a_byte_verified_four_layer_fixture_is_accepted(self):
        from glm53_setup.validation import run_fixture

        run_fixture.check_fixture(FOUR_LAYER, COMPLETE)

    def test_every_missing_guarantee_is_refused(self):
        from glm53_setup.validation import run_fixture

        cases = {
            "not marked as a fixture": (
                {"text_config": {"num_hidden_layers": 4}},
                COMPLETE,
            ),
            "the full model": (
                {**FOUR_LAYER, "text_config": {"num_hidden_layers": 45}},
                COMPLETE,
            ),
            "an incomplete download": (FOUR_LAYER, {**COMPLETE, "status": "partial"}),
            "unverified bytes": (
                FOUR_LAYER,
                {**COMPLETE, "all_tensor_bytes_verified": False},
            ),
            "no verification recorded": (FOUR_LAYER, {"status": "complete"}),
        }
        for name, (config, status) in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    run_fixture.check_fixture(config, status)


class FixtureFilesTests(unittest.TestCase):
    def test_the_gate_reads_the_two_files_the_fixture_builder_writes(self):
        from glm53_setup.validation import run_fixture

        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            status = fixture / "fixture-status.json"
            (fixture / "config.json").write_text(
                json.dumps(FOUR_LAYER), encoding="utf-8"
            )
            status.write_text(json.dumps(COMPLETE), encoding="utf-8")
            run_fixture.read_fixture(fixture)
            status.write_text(json.dumps({"status": "partial"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_fixture.read_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
