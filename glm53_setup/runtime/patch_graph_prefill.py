"""Pinned runner patch: LPA prefill is eager, including short final chunks."""

from . import pinned_patch
from .pinned_patch import replace_once

RUNNER = "v1/worker/gpu/model_runner.py"
SOURCE_SHA256 = "5d6aa0b3dd567b0a3e5d6a32723ffeac5f331ee37b4e35804a7e011120212f48"
MISMATCH = "Graph prefill runner source hash mismatch"
RECORD = "glm53-graph-prefill-patch.json"
HEADER = "# Modified by GLM setup: opt-in LPA eager-prefill routing.\n# Original vLLM Apache-2.0 notices below remain applicable.\n"


def patch_text(text):
    if "_glm53_force_eager_prefill" in text:
        raise ValueError("Graph prefill patch already applied")
    text = replace_once(
        text,
        "import torch\n",
        "import torch\nfrom glm53_setup.runtime.graph_policy import force_eager_prefill as _glm53_force_eager_prefill\n",
    )
    text = replace_once(
        text,
        "need_eager=is_profile or skip_compiled,",
        "need_eager=is_profile or skip_compiled or _glm53_force_eager_prefill(batch_req_state),",
    )
    text = HEADER + text
    compile(text, RUNNER, "exec")
    return text


def prepare(package):
    return pinned_patch.prepare(package, RUNNER, SOURCE_SHA256, MISMATCH, patch_text)


def main(argv=None):
    pinned_patch.main(
        argv,
        doc=__doc__,
        target=RUNNER,
        sha256=SOURCE_SHA256,
        prepare=prepare,
        record=RECORD,
    )


if __name__ == "__main__":
    main()
