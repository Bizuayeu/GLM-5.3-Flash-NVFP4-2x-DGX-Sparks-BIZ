"""Source-pinned patch: canonical token order inside each expert for Marlin MoE."""

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

from .patch_nope_reference import replace_once

TARGET = "model_executor/layers/fused_moe/experts/marlin_moe.py"
SOURCE_SHA256 = "93916b721548e6194dc70cf6cec224b4b1456a0cc0980235e97beba7bf36541d"
ANCHOR = (
    "        expert_map,\n"
    "        ignore_invalid_experts=True,\n"
    "    )\n"
    "\n"
    "    assert activation is not None\n"
)
# Expert parallelism marks foreign experts in expert_ids; that layout is left alone.
INSERT = (
    "        expert_map,\n"
    "        ignore_invalid_experts=True,\n"
    "    )\n"
    "    from glm53_setup.runtime.moe_token_order import (\n"
    "        canonical_expert_order,\n"
    "        moe_order_enabled,\n"
    "    )\n"
    "\n"
    "    if moe_order_enabled() and expert_map is None:\n"
    "        sorted_token_ids = canonical_expert_order(\n"
    "            sorted_token_ids, expert_ids, num_tokens_post_padded\n"
    "        )\n"
    "\n"
    "    assert activation is not None\n"
)
HEADER = (
    "# Modified by GLM setup: canonical token order inside each expert.\n"
    "# Original vLLM Apache-2.0 notices below remain applicable.\n"
)


def patch_text(text):
    if "canonical_expert_order" in text:
        raise ValueError("Marlin MoE order patch already applied")
    patched = HEADER + replace_once(text, ANCHOR, INSERT)
    compile(patched, TARGET, "exec")
    return patched


def prepare(package):
    original = (package / TARGET).read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("Marlin MoE source hash mismatch")
    return patch_text(original.decode("utf-8")).encode("utf-8")


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
        (package / TARGET).write_bytes(patched)
        (package.parent / "glm53-moe-order-patch.json").write_text(
            json.dumps(record, indent=2)
        )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
