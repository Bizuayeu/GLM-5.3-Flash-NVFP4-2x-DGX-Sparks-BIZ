import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from glm53_setup.runtime import patch_adaptive_depth as patch

# The lines of vLLM 385dce36 that the patch touches, arranged so each excerpt still runs.
SCHEDULER = """from typing import Any


class Scheduler:
    def __init__(self, vllm_config, requests):
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = vllm_config.num_lookahead_tokens
        self.dynamic_sd_lookup = None
        self.requests = requests

    def schedule(self, num_scheduled_tokens):
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]

        return num_spec_tokens_to_schedule

    def update_from_output(self, request, scheduled_spec_token_ids, generated):
        if scheduled_spec_token_ids:
            if generated:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = max(len(generated) - 1, 0)
                num_rejected = num_draft_tokens - num_accepted
                return num_rejected
"""

RUNNER = """from typing import NamedTuple


def use_workspace_lane(lane):
    return lane


class GPUModelRunner:
    def execute_model(self, scheduler_output):
        self.execute_model_state = ExecuteModelState(
            routed_experts=routed_experts,
            cudagraph_stats=cudagraph_stats,
        )

        if not self.is_last_pp_rank:
            return None

    def sample_tokens(self):
        routed_experts = self.execute_model_state.routed_experts
        cudagraph_stats = self.execute_model_state.cudagraph_stats
        self.execute_model_state = None
        if self.speculator is not None:
            with use_workspace_lane(self._draft_workspace_lane):
                draft_tokens = self.speculator.propose(
                    input_batch,
                )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens

        if self.num_speculative_steps > 0:
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )


class ExecuteModelState(NamedTuple):
    routed_experts: object
    cudagraph_stats: CUDAGraphStat | None
"""

SPECULATOR = """def record(*arguments, advance_draft_positions):
    return arguments


class AutoRegressiveSpeculator:
    def __init__(self, num_speculative_steps):
        self.num_speculative_steps = num_speculative_steps
        self.max_model_len = 100
        self.draft_tokens = [[7] * num_speculative_steps]
        self.forwards = []

    @property
    def advance_draft_positions(self) -> bool:
        return True

    def propose(self, max_seq_len, num_reqs=1):
        self.draft_max_seq_len = min(
            max_seq_len + self.num_speculative_steps, self.max_model_len
        )
        self.forwards.append(0)

        if self.num_speculative_steps == 1:
            # Early exit.
            return self.draft_tokens[:num_reqs, :1]

        self._multi_step_decode()
        return self.draft_tokens[:num_reqs]

    def _multi_step_decode(self):
        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            self._generate_draft(step)

    def _generate_fused_drafts(self, attn_metadata):
        attn_groups = (
            []
        )

        for step in range(1, self.num_speculative_steps):
            self._generate_draft(step)
            if (
                step < self.num_speculative_steps - 1
                and attn_metadata is not None
            ):
                self.forwards.append("metadata")

    def _generate_draft(self, step):
        self.forwards.append(step)
        self.kernel_arguments = record(
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
        )
"""


def load(name, source, class_name):
    namespace = {}
    exec(compile(patch.patch_text(name, source), name, "exec"), namespace)
    return namespace[class_name]


def scheduler(depth=5, requests=None):
    config = SimpleNamespace(num_speculative_tokens=depth, num_lookahead_tokens=depth)
    return load(patch.SCHEDULER, SCHEDULER, "Scheduler")(config, requests or {})


class SchedulerPatchTests(unittest.TestCase):
    def test_without_the_variable_the_step_depth_is_the_configured_depth(self):
        with mock.patch.dict(os.environ, clear=True):
            patched = scheduler(requests={"a": SimpleNamespace()})
            request = patched.requests["a"]
            for _ in range(40):
                patched.update_from_output(request, [-1] * 5, [11])
            self.assertEqual(patched.schedule({"a": 6}), 5)
            self.assertEqual(vars(request), {})

    def test_with_the_variable_rejections_lower_the_depth_of_that_request(self):
        with mock.patch.dict(os.environ, {"GLM53_ADAPTIVE_DEPTH": "1"}, clear=True):
            patched = scheduler(
                requests={"a": SimpleNamespace(), "b": SimpleNamespace()}
            )
            self.assertEqual(patched.schedule({"a": 6}), 5)
            for _ in range(10):
                # Five drafts checked, none accepted: one generated token.
                patched.update_from_output(patched.requests["a"], [-1] * 5, [11])
            self.assertEqual(patched.schedule({"a": 6}), 1)
            # A request without evidence still asks for the ceiling; the step takes the deepest.
            self.assertEqual(patched.schedule({"a": 2, "b": 6}), 5)
            self.assertEqual(patched.schedule({}), 5)

    def test_a_single_draft_depth_has_nothing_to_adapt(self):
        with mock.patch.dict(os.environ, {"GLM53_ADAPTIVE_DEPTH": "1"}, clear=True):
            self.assertFalse(scheduler(depth=1)._glm53_adaptive_depth)

    def test_a_malformed_switch_fails_at_start(self):
        with mock.patch.dict(os.environ, {"GLM53_ADAPTIVE_DEPTH": "yes"}, clear=True):
            with self.assertRaises(ValueError):
                scheduler()


class SpeculatorPatchTests(unittest.TestCase):
    def setUp(self):
        self.cls = load(patch.SPECULATOR, SPECULATOR, "AutoRegressiveSpeculator")

    def test_unset_depth_runs_every_configured_draft_forward(self):
        speculator = self.cls(5)
        speculator.propose(max_seq_len=10)
        self.assertEqual(speculator.forwards, [0, 1, 2, 3, 4])
        self.assertEqual(speculator.draft_max_seq_len, 15)
        self.assertEqual(speculator.kernel_arguments, (100, 5))
        speculator.glm53_active_steps = None
        self.assertEqual(speculator.glm53_steps, 5)

    def test_a_lower_depth_stops_both_draft_loops_and_the_kernel_bound(self):
        speculator = self.cls(5)
        speculator.glm53_active_steps = 3
        drafts = speculator.propose(max_seq_len=10)
        self.assertEqual(speculator.forwards, [0, 1, 2])
        self.assertEqual(speculator.draft_max_seq_len, 13)
        # The final-step test inside the input-update kernel follows the depth.
        self.assertEqual(speculator.kernel_arguments, (100, 3))
        self.assertEqual(len(drafts[0]), 5)
        speculator.forwards.clear()
        speculator._generate_fused_drafts(attn_metadata=object())
        self.assertEqual(speculator.forwards, [1, "metadata", 2])

    def test_depth_one_returns_after_the_first_forward_with_the_full_width(self):
        speculator = self.cls(5)
        speculator.glm53_active_steps = 1
        drafts = speculator.propose(max_seq_len=10)
        self.assertEqual(speculator.forwards, [0])
        self.assertEqual(len(drafts[0]), 5)


class RunnerPatchTests(unittest.TestCase):
    def setUp(self):
        self.patched = patch.patch_text(patch.RUNNER, RUNNER)

    def test_the_depth_travels_from_the_scheduler_output_to_the_draft_call(self):
        self.assertIn(
            "glm53_draft_depth=scheduler_output.num_spec_tokens_to_schedule,",
            self.patched,
        )
        self.assertIn("    glm53_draft_depth: int\n", self.patched)
        read = self.patched.index("glm53_draft_depth = self.execute_model_state.")
        cleared = self.patched.index("self.execute_model_state = None")
        armed = self.patched.index("self.speculator.glm53_active_steps = glm53_")
        proposed = self.patched.index("self.speculator.propose(")
        disarmed = self.patched.index("self.speculator.glm53_active_steps = None")
        self.assertLess(read, cleared)
        self.assertLess(armed, proposed)
        self.assertLess(proposed, disarmed)

    def test_a_depth_outside_the_configured_range_means_the_configured_depth(self):
        self.assertIn(
            "        if not 0 < glm53_draft_depth < self.num_speculative_steps:\n"
            "            glm53_draft_depth = self.num_speculative_steps\n",
            self.patched,
        )

    def test_the_synchronous_scheduler_gets_one_placeholder_per_drafted_position(self):
        self.assertIn(
            "self.req_states.draft_tokens[input_batch.idx_mapping][\n"
            "                    :, :glm53_draft_depth\n",
            self.patched,
        )
        # The write into the persistent buffer keeps its full width.
        self.assertIn(
            "self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens",
            self.patched,
        )


class PatchTests(unittest.TestCase):
    sources = {
        patch.SCHEDULER: SCHEDULER,
        patch.RUNNER: RUNNER,
        patch.SPECULATOR: SPECULATOR,
    }

    def test_refuses_drifted_or_already_patched_source(self):
        drift = {
            patch.SCHEDULER: "num_rejected = ",
            patch.RUNNER: "cudagraph_stats: ",
            patch.SPECULATOR: "# Early exit.",
        }
        for name, source in self.sources.items():
            with self.assertRaises(ValueError):
                patch.patch_text(name, source.replace(drift[name], "drifted"))
            with self.assertRaises(ValueError):
                patch.patch_text(name, patch.patch_text(name, source))

    def test_prepare_checks_every_hash_before_returning_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digests = {}
            for name, source in self.sources.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(source.encode())
                digests[name] = hashlib.sha256(source.encode()).hexdigest()
            with mock.patch.object(patch, "SOURCES", digests):
                patched = patch.prepare(root)
                self.assertEqual(set(patched), set(self.sources))
                for data in patched.values():
                    self.assertTrue(data.startswith(b"# Modified by GLM setup"))
                (root / patch.SPECULATOR).write_bytes(b"drifted\n")
                with self.assertRaises(ValueError):
                    patch.prepare(root)


if __name__ == "__main__":
    unittest.main()
