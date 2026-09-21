"""Source-pinned patch: the V2 MTP speculator drafts the depth the scheduler asks for.

vLLM 385dce36 already carries a per-step draft depth from the scheduler to the workers
(``SchedulerOutput.num_spec_tokens_to_schedule``): the async scheduler sizes the next step's
placeholders from it and the verifier takes each request's draft count from what was scheduled.
Nothing under ``v1/worker/gpu/`` reads it, so ``AutoRegressiveSpeculator`` always runs
``num_speculative_tokens`` draft forwards (upstream issue #51510; PR #57053, open on 2026-09-21,
adds the reader together with CUDA-graph support this deployment does not use). Three edits:

- ``scheduler.py``: with ``GLM53_ADAPTIVE_DEPTH=1`` the step's depth comes from
  ``adaptive_depth.batch_depth`` and every counted verification feeds ``adaptive_depth.observe``.
  Without the variable the scheduler emits the configured depth, as before. With
  ``GLM53_DEPTH_TRACE=<file>`` every counted verification is also appended to that file
  (``adaptive_depth.trace``), with the policy on or off.
- ``model_runner.py``: the depth travels with ``ExecuteModelState`` to the draft call, and the
  synchronous scheduler gets that many placeholders back.
- ``autoregressive/speculator.py``: the draft loops stop at that depth. The draft buffer keeps its
  full width; columns past the depth keep earlier token ids, which the verifier never reads
  because it reads as many columns as the scheduler scheduled.

A fourth file and two more runner edits carry a diagnostic, ``GLM53_DRAFT_OBSERVE=<file>``
(``draft_observe.py``): ``spec_decode/speculator.py`` hands the greedy draft to
``draft_observe.draft`` (same ``compute_logits`` + ``argmax``, plus the draft's confidence kept on
the GPU) and the runner summarises the verifier's logits around the rejection sampler and writes
one line per step. Unset, each site is one cached boolean test.

Every rank honours the depth in the scheduler output, so ranks cannot disagree on the number of
draft forwards. Eager execution only: the speculator's CUDA graphs are captured for the configured
depth. To be dropped when the pin moves past the merge of #57053.
"""

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

from .patch_nope_reference import replace_once

SCHEDULER = "v1/core/sched/scheduler.py"
RUNNER = "v1/worker/gpu/model_runner.py"
SPECULATOR = "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
BASE_SPECULATOR = "v1/worker/gpu/spec_decode/speculator.py"
SOURCES = {
    SCHEDULER: "0f7e248a92e7eb7255264956b5bb1947beca9710bad8e886585b703b045080f2",
    RUNNER: "5d6aa0b3dd567b0a3e5d6a32723ffeac5f331ee37b4e35804a7e011120212f48",
    SPECULATOR: "97e6bfe9a50f4dd6455702e1fd0bfb4ae98edcd45ba812f9e0d3250916b28e3f",
    BASE_SPECULATOR: "6443d873f066c4f78b090d5afacea4a675b0b3757a4b1f4e9573cda3cc11f480",
}
HEADER = (
    "# Modified by GLM setup: the MTP draft depth follows the scheduler's per-step depth.\n"
    "# Original vLLM Apache-2.0 notices below remain applicable.\n"
)

SCHEDULER_EDITS = (
    (
        "from typing import Any\n",
        "from typing import Any\n\n"
        "from glm53_setup.runtime import adaptive_depth as _glm53_adaptive_depth\n",
    ),
    (
        "        self.num_lookahead_tokens = vllm_config.num_lookahead_tokens\n",
        "        self.num_lookahead_tokens = vllm_config.num_lookahead_tokens\n"
        "        self._glm53_adaptive_depth = (\n"
        "            self.num_spec_tokens > 1 and _glm53_adaptive_depth.enabled()\n"
        "        )\n"
        "        self._glm53_depth_trace = _glm53_adaptive_depth.trace_enabled()\n",
    ),
    (
        "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
        "                len(num_scheduled_tokens)\n"
        "            ]\n",
        "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
        "                len(num_scheduled_tokens)\n"
        "            ]\n"
        "        if self._glm53_adaptive_depth and num_scheduled_tokens:\n"
        "            num_spec_tokens_to_schedule = min(\n"
        "                num_spec_tokens_to_schedule,\n"
        "                _glm53_adaptive_depth.batch_depth(\n"
        "                    (self.requests[req_id] for req_id in num_scheduled_tokens),\n"
        "                    self.num_spec_tokens,\n"
        "                ),\n"
        "            )\n",
    ),
    (
        "                num_rejected = num_draft_tokens - num_accepted\n",
        "                num_rejected = num_draft_tokens - num_accepted\n"
        "                if self._glm53_adaptive_depth:\n"
        "                    _glm53_adaptive_depth.observe(\n"
        "                        request, self.num_spec_tokens, num_draft_tokens, num_accepted\n"
        "                    )\n"
        "                if self._glm53_depth_trace:\n"
        "                    _glm53_adaptive_depth.trace(\n"
        "                        request,\n"
        "                        request.num_output_tokens,\n"
        "                        num_draft_tokens,\n"
        "                        num_accepted,\n"
        "                        scheduler_output.num_spec_tokens_to_schedule,\n"
        "                    )\n",
    ),
)

RUNNER_EDITS = (
    (
        "import torch\n",
        "import torch\n\n"
        "from glm53_setup.runtime import draft_observe as _glm53_draft_observe\n",
    ),
    (
        "            assert self.speculator is not None\n"
        "            sampler_output = self.rejection_sampler(\n"
        "                logits,\n"
        "                input_batch,\n"
        "                # Draft logits are needed for probabilistic rejection sampling.\n"
        "                self.speculator.draft_logits,\n"
        "            )\n",
        "            assert self.speculator is not None\n"
        "            glm53_observed = (\n"
        "                _glm53_draft_observe.target(logits, input_batch)\n"
        "                if _glm53_draft_observe.active() and shard_metadata is None\n"
        "                else None\n"
        "            )\n"
        "            sampler_output = self.rejection_sampler(\n"
        "                logits,\n"
        "                input_batch,\n"
        "                # Draft logits are needed for probabilistic rejection sampling.\n"
        "                self.speculator.draft_logits,\n"
        "            )\n"
        "            if glm53_observed is not None:\n"
        "                _glm53_draft_observe.write(\n"
        "                    self.speculator,\n"
        "                    input_batch,\n"
        "                    glm53_observed,\n"
        "                    sampler_output.num_sampled,\n"
        "                )\n",
    ),
    (
        "            routed_experts=routed_experts,\n"
        "            cudagraph_stats=cudagraph_stats,\n"
        "        )\n\n"
        "        if not self.is_last_pp_rank:\n",
        "            routed_experts=routed_experts,\n"
        "            cudagraph_stats=cudagraph_stats,\n"
        "            glm53_draft_depth=scheduler_output.num_spec_tokens_to_schedule,\n"
        "        )\n\n"
        "        if not self.is_last_pp_rank:\n",
    ),
    (
        "        cudagraph_stats = self.execute_model_state.cudagraph_stats\n",
        "        cudagraph_stats = self.execute_model_state.cudagraph_stats\n"
        "        glm53_draft_depth = self.execute_model_state.glm53_draft_depth\n"
        "        if not 0 < glm53_draft_depth < self.num_speculative_steps:\n"
        "            glm53_draft_depth = self.num_speculative_steps\n",
    ),
    (
        "            with use_workspace_lane(self._draft_workspace_lane):\n"
        "                draft_tokens = self.speculator.propose(\n",
        "            self.speculator.glm53_active_steps = glm53_draft_depth\n"
        "            with use_workspace_lane(self._draft_workspace_lane):\n"
        "                draft_tokens = self.speculator.propose(\n",
    ),
    (
        "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n",
        "            self.speculator.glm53_active_steps = None\n"
        "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n",
    ),
    (
        "                input_batch,\n"
        "                self.req_states.draft_tokens[input_batch.idx_mapping],\n"
        "            )\n",
        "                input_batch,\n"
        "                self.req_states.draft_tokens[input_batch.idx_mapping][\n"
        "                    :, :glm53_draft_depth\n"
        "                ],\n"
        "            )\n",
    ),
    (
        "    cudagraph_stats: CUDAGraphStat | None\n",
        "    cudagraph_stats: CUDAGraphStat | None\n    glm53_draft_depth: int\n",
    ),
)

ACTIVE = "self.glm53_steps"
SPECULATOR_EDITS = (
    (
        "    @property\n    def advance_draft_positions(self) -> bool:\n",
        "    @property\n"
        "    def glm53_steps(self) -> int:\n"
        "        # Draft forwards for this propose() call; the runner sets it around the call.\n"
        '        active = getattr(self, "glm53_active_steps", None)\n'
        "        return active or self.num_speculative_steps\n\n"
        "    @property\n    def advance_draft_positions(self) -> bool:\n",
    ),
    (
        "            max_seq_len + self.num_speculative_steps, self.max_model_len\n",
        f"            max_seq_len + {ACTIVE}, self.max_model_len\n",
    ),
    (
        "        if self.num_speculative_steps == 1:\n"
        "            # Early exit.\n"
        "            return self.draft_tokens[:num_reqs, :1]\n",
        f"        if {ACTIVE} == 1:\n"
        "            # Early exit. Full width: the runner assigns into a full-width buffer.\n"
        "            return self.draft_tokens[:num_reqs]\n",
    ),
    (
        "        slot_mappings_by_layer = None\n"
        "        for step in range(1, self.num_speculative_steps):\n",
        "        slot_mappings_by_layer = None\n"
        f"        for step in range(1, {ACTIVE}):\n",
    ),
    (
        "        )\n\n        for step in range(1, self.num_speculative_steps):\n",
        f"        )\n\n        for step in range(1, {ACTIVE}):\n",
    ),
    (
        "                step < self.num_speculative_steps - 1\n",
        f"                step < {ACTIVE} - 1\n",
    ),
    (
        "            self.max_model_len,\n"
        "            self.num_speculative_steps,\n"
        "            advance_draft_positions=self.advance_draft_positions,\n",
        "            self.max_model_len,\n"
        f"            {ACTIVE},\n"
        "            advance_draft_positions=self.advance_draft_positions,\n",
    ),
)
BASE_SPECULATOR_EDITS = (
    (
        "import torch.nn as nn\n",
        "import torch.nn as nn\n\n"
        "from glm53_setup.runtime import draft_observe as _glm53_draft_observe\n",
    ),
    (
        "        return self._greedy_sample_draft(hidden_states)\n",
        "        if _glm53_draft_observe.active():\n"
        "            return _glm53_draft_observe.draft(\n"
        "                self, hidden_states, idx_mapping, draft_step\n"
        "            )\n"
        "        return self._greedy_sample_draft(hidden_states)\n",
    ),
)
EDITS = {
    SCHEDULER: SCHEDULER_EDITS,
    RUNNER: RUNNER_EDITS,
    SPECULATOR: SPECULATOR_EDITS,
    BASE_SPECULATOR: BASE_SPECULATOR_EDITS,
}


def patch_text(name, text):
    if "glm53_" in text:
        raise ValueError("Adaptive-depth patch already applied: " + name)
    for old, new in EDITS[name]:
        text = replace_once(text, old, new)
    text = HEADER + text
    compile(text, name, "exec")
    return text


def prepare(package):
    patched = {}
    for name, digest in SOURCES.items():
        original = (package / name).read_bytes()
        if hashlib.sha256(original).hexdigest() != digest:
            raise ValueError("Adaptive-depth source hash mismatch: " + name)
        patched[name] = patch_text(name, original.decode("utf-8")).encode("utf-8")
    return patched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    package = args.package or Path(sysconfig.get_paths()["purelib"]) / "vllm"
    patched = prepare(package)
    record = {
        "files": {
            name: {
                "source_sha256": SOURCES[name],
                "patched_sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in patched.items()
        },
        "check_only": args.check,
    }
    if not args.check:
        for name, data in patched.items():
            (package / name).write_bytes(data)
        (package.parent / "glm53-adaptive-depth-patch.json").write_text(
            json.dumps(record, indent=2)
        )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
