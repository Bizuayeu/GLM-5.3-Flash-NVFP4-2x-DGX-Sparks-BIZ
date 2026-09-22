"""Source-pinned PP fixture patch; retain deferred mHC state at stage boundaries."""

import ast

from . import pinned_patch
from .pinned_patch import replace_once

MODEL = "models/glm5next/nvidia/model.py"
SOURCE_SHA256 = "26c73584381edb16f28b213d7976018d58ddfe8165657e105453d89b4d7e0fe2"
MISMATCH = "Pipeline model source hash mismatch"
RECORD = "glm53-pipeline-patch.json"
HEADER = "# Modified by GLM setup: experimental deferred-state-preserving PP.\n# Original vLLM Apache-2.0 notices below remain applicable.\n"


def append_method(text, class_name, method):
    node = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    if any(
        isinstance(n, ast.FunctionDef) and n.name == "make_empty_intermediate_tensors"
        for n in node.body
    ):
        raise ValueError("Pipeline method already exists")
    lines = text.splitlines(keepends=True)
    lines.insert(node.end_lineno, "\n" + method + "\n")
    return "".join(lines)


def patch_text(text):
    if "from glm53_setup.runtime.pipeline_state import validate_pipeline" in text:
        raise ValueError("Pipeline patch already applied")
    text = replace_once(
        text,
        "        config = vllm_config.model_config.hf_config\n        self.config = config\n",
        "        config = vllm_config.model_config.hf_config\n        self.config = config\n"
        "        from glm53_setup.runtime.pipeline_state import validate_pipeline\n"
        "        validate_pipeline(vllm_config)\n",
    )
    text = replace_once(
        text,
        "            # post/comb (deferred mHC hc_post state) are not propagated across\n"
        "            # PP ranks; the receiving rank's first mHC layer uses standalone pre.\n"
        "            post = None\n            comb = None\n",
        '            post = intermediate_tensors["post"]\n'
        '            comb = intermediate_tensors["comb"]\n',
    )
    text = replace_once(
        text,
        "            # PP is gated off for GLM-5.3-Flash (no make_empty_intermediate_tensors),\n"
        "            # so this branch is not exercised. post/comb are the deferred\n"
        "            # hc_post state of this rank's last mHC layer; a future PP path\n"
        "            # would need to propagate them, but for now they are dropped (the\n"
        "            # receiving rank's first layer would fall back to standalone pre).\n",
        "            # Preserve the deferred post state for the next stage's fused pre.\n",
    )
    text = replace_once(
        text,
        '                {"hidden_states": hidden_states, "residual": residual}\n',
        '                {"hidden_states": hidden_states, "residual": residual,\n'
        '                 "post": post, "comb": comb}\n',
    )
    text = replace_once(
        text,
        "        # Glm5NextForCausalLM does not implement make_empty_intermediate_tensors,\n"
        "        # so pipeline parallelism is gated off (consistent with the text-only\n"
        "        # model) and we intentionally do not alias it here.\n",
        "        # Pipeline buffer creation delegates through the language model.\n",
    )
    text = append_method(
        text,
        "Glm5NextModel",
        """    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        from glm53_setup.runtime.pipeline_state import allocate_intermediate
        return IntermediateTensors(allocate_intermediate(self.config, batch_size, dtype, device))""",
    )
    for name, target in (
        ("Glm5NextForCausalLM", "model"),
        ("Glm5NextForConditionalGeneration", "language_model"),
    ):
        text = append_method(
            text,
            name,
            f"    def make_empty_intermediate_tensors(self, batch_size, dtype, device):\n"
            f"        return self.{target}.make_empty_intermediate_tensors(batch_size, dtype, device)",
        )
    text = HEADER + text
    compile(text, MODEL, "exec")
    return text


def prepare(package):
    return pinned_patch.prepare(package, MODEL, SOURCE_SHA256, MISMATCH, patch_text)


def main(argv=None):
    pinned_patch.main(
        argv,
        doc=__doc__,
        target=MODEL,
        sha256=SOURCE_SHA256,
        prepare=prepare,
        record=RECORD,
    )


if __name__ == "__main__":
    main()
