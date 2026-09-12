"""Source-pinned PP fixture patch; retain deferred mHC state at stage boundaries."""

import argparse
import ast
import hashlib
import json
import sysconfig
from pathlib import Path

from .patch_nope_reference import replace_once

MODEL = "models/glm5next/nvidia/model.py"
SOURCE_SHA256 = "26c73584381edb16f28b213d7976018d58ddfe8165657e105453d89b4d7e0fe2"


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


def prepare(package):
    original = (package / MODEL).read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pipeline model source hash mismatch")
    text = original.decode("utf-8")
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
    text = (
        "# Modified by GLM setup: experimental deferred-state-preserving PP.\n# Original vLLM Apache-2.0 notices below remain applicable.\n"
        + text
    )
    compile(text, MODEL, "exec")
    return text.encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    package = args.package or Path(sysconfig.get_paths()["purelib"]) / "vllm"
    patched = prepare(package)
    record = {
        "source_sha256": SOURCE_SHA256,
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "check_only": args.check,
    }
    if not args.check:
        (package / MODEL).write_bytes(patched)
        (package.parent / "glm53-pipeline-patch.json").write_text(
            json.dumps(record, indent=2)
        )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
