"""Source-pinned patch: the kpool indexer's top-k goes through the stable selection."""

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

from .patch_nope_reference import replace_once

TARGET = "model_executor/layers/sparse_attn_indexer_kpool.py"
SOURCE_SHA256 = "631ae1cce792c18f2922af7763264c486317cb6f4318823b1482e55bec5d7f59"
IMPORT_ANCHOR = "RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024\n"
IMPORT = (
    "from glm53_setup.runtime.stable_topk import decode_topk as glm53_decode_topk\n"
    "from glm53_setup.runtime.stable_topk import prefill_topk as glm53_prefill_topk\n"
)
# Same arguments in the same order; the XPU branches are left alone.
CALLS = (
    ("torch.ops._C.persistent_topk(\n", "glm53_decode_topk(\n"),
    ("torch.ops._C.top_k_per_row_prefill(\n", "glm53_prefill_topk(\n"),
)
HEADER = (
    "# Modified by GLM setup: indexer top-k with ties settled by the lower index.\n"
    "# Original vLLM Apache-2.0 notices below remain applicable.\n"
)


def patch_text(text):
    if "glm53_decode_topk" in text:
        raise ValueError("Indexer top-k patch already applied")
    patched = replace_once(text, IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT)
    for old, new in CALLS:
        patched = replace_once(patched, old, new)
    patched = HEADER + patched
    compile(patched, TARGET, "exec")
    return patched


def prepare(package):
    original = (package / TARGET).read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("kpool indexer source hash mismatch")
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
        (package.parent / "glm53-indexer-topk-patch.json").write_text(
            json.dumps(record, indent=2)
        )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
