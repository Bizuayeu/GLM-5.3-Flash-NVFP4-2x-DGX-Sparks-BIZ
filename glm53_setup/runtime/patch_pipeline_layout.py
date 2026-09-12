"""Resolve a common supported layout for heterogeneous PP stages."""

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

from .patch_nope_reference import replace_once

TARGET = "v1/attention/backends/utils.py"
SOURCE_SHA256 = "63f58dd2b29543045109f40897ddc5f57e299bb4259f0e37a64b024cb1f20ddb"


def prepare(package):
    source = (package / TARGET).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pipeline cache-layout source hash mismatch")
    original = """    assert all(names == supported_layouts[0] for names in supported_layouts[1:]), (
        f"Workers disagree on supported KV cache layouts: {supported_layouts}."
    )
"""
    replacement = (
        """    if vllm_config.parallel_config.pipeline_parallel_size == 2:
        from glm53_setup.runtime.pipeline_state import common_layout_names
        common = common_layout_names(supported_layouts)
        supported_layouts = [common for _ in supported_layouts]
"""
        + original
    )
    text = replace_once(source.decode("utf-8"), original, replacement)
    text = (
        "# Modified by GLM setup: common supported layouts for two PP stages.\n"
        "# Original vLLM Apache-2.0 notices below remain applicable.\n" + text
    )
    compile(text, TARGET, "exec")
    return text.encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    package = args.package or Path(sysconfig.get_paths()["purelib"]) / "vllm"
    patched = prepare(package)
    report = {
        "source_sha256": SOURCE_SHA256,
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "check_only": args.check,
    }
    if not args.check:
        (package / TARGET).write_bytes(patched)
        (package.parent / "glm53-pipeline-layout-patch.json").write_text(
            json.dumps(report, indent=2)
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
