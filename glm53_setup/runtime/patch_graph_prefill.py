"""Pinned runner patch: LPA prefill is eager, including short final chunks."""

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

from .patch_nope_reference import replace_once

RUNNER = "v1/worker/gpu/model_runner.py"
SOURCE_SHA256 = "5d6aa0b3dd567b0a3e5d6a32723ffeac5f331ee37b4e35804a7e011120212f48"


def prepare(package):
    original = (package / RUNNER).read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("Graph prefill runner source hash mismatch")
    text = replace_once(
        original.decode("utf-8"),
        "import torch\n",
        "import torch\nfrom glm53_setup.runtime.graph_policy import force_eager_prefill as _glm53_force_eager_prefill\n",
    )
    text = replace_once(
        text,
        "need_eager=is_profile or skip_compiled,",
        "need_eager=is_profile or skip_compiled or _glm53_force_eager_prefill(batch_req_state),",
    )
    text = (
        "# Modified by GLM setup: opt-in LPA eager-prefill routing.\n# Original vLLM Apache-2.0 notices below remain applicable.\n"
        + text
    )
    compile(text, RUNNER, "exec")
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
        (package / RUNNER).write_bytes(patched)
        (package.parent / "glm53-graph-prefill-patch.json").write_text(
            json.dumps(record, indent=2)
        )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
