"""Source-pinned patch: canonical token order inside each expert for Marlin MoE."""

from . import pinned_patch
from .pinned_patch import replace_once

TARGET = "model_executor/layers/fused_moe/experts/marlin_moe.py"
SOURCE_SHA256 = "93916b721548e6194dc70cf6cec224b4b1456a0cc0980235e97beba7bf36541d"
MISMATCH = "Marlin MoE source hash mismatch"
RECORD = "glm53-moe-order-patch.json"
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
    "            sorted_token_ids, expert_ids, num_tokens_post_padded, block_size_m\n"
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
    return pinned_patch.prepare(package, TARGET, SOURCE_SHA256, MISMATCH, patch_text)


def main(argv=None):
    pinned_patch.main(
        argv,
        doc=__doc__,
        target=TARGET,
        sha256=SOURCE_SHA256,
        prepare=prepare,
        record=RECORD,
    )


if __name__ == "__main__":
    main()
