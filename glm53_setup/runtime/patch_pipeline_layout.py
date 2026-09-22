"""Resolve a common supported layout for heterogeneous PP stages."""

from . import pinned_patch
from .pinned_patch import replace_once

TARGET = "v1/attention/backends/utils.py"
SOURCE_SHA256 = "63f58dd2b29543045109f40897ddc5f57e299bb4259f0e37a64b024cb1f20ddb"
MISMATCH = "Pipeline cache-layout source hash mismatch"
RECORD = "glm53-pipeline-layout-patch.json"
ASSERTION = """    assert all(names == supported_layouts[0] for names in supported_layouts[1:]), (
        f"Workers disagree on supported KV cache layouts: {supported_layouts}."
    )
"""
RECONCILED = (
    """    if vllm_config.parallel_config.pipeline_parallel_size == 2:
        from glm53_setup.runtime.pipeline_state import common_layout_names
        common = common_layout_names(supported_layouts)
        supported_layouts = [common for _ in supported_layouts]
"""
    + ASSERTION
)
HEADER = (
    "# Modified by GLM setup: common supported layouts for two PP stages.\n"
    "# Original vLLM Apache-2.0 notices below remain applicable.\n"
)


def patch_text(text):
    if "from glm53_setup.runtime.pipeline_state import common_layout_names" in text:
        raise ValueError("Pipeline cache-layout patch already applied")
    text = HEADER + replace_once(text, ASSERTION, RECONCILED)
    compile(text, TARGET, "exec")
    return text


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
