"""Source-pinned patch: the slot-mapping kernel reads a block table only inside its row.

vLLM 385dce36 computes ``positions // kernel_block_size`` for every KV cache group and loads
``block_table[row + that]`` without asking whether the row is that wide. The kpool indexer's raw-tail
group (``KpoolTailSpec``: one circular block, block size 4, a row of 32 entries) is read past its row
from position 128 on, one byte further per token of context. The read is harmless while the bytes
behind the table are mapped; when they are not, the step ends in a CUDA illegal memory access. On the
reference pair that happened at about 250K tokens with the requantized checkpoint's allocation
layout and not with the stock one; compute-sanitizer reports the same read on a stock four-layer
fixture. Upstream: vllm-project/vllm issue #53982, fix proposed in PR #54296 (open on 2026-09-21);
this is the same guard on the pinned source, to be dropped when the pin moves past the merge. A
position the row cannot index now reads nothing and gets no slot.
"""

from . import pinned_patch
from .pinned_patch import replace_once

TARGET = "v1/worker/gpu/block_table.py"
SOURCE_SHA256 = "61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a"
MISMATCH = "block table source hash mismatch"
RECORD = "glm53-slot-mapping-patch.json"
READ = (
    "        block_numbers = tl.load(\n"
    "            block_table_ptr + req_state_idx * block_table_stride + block_indices,\n"
    "            mask=is_local,\n"
)
BOUNDED_READ = (
    "        in_range = block_indices < block_table_stride\n"
    "        block_numbers = tl.load(\n"
    "            block_table_ptr + req_state_idx * block_table_stride + block_indices,\n"
    "            mask=is_local & in_range,\n"
)
SLOTS = "        slot_ids = block_numbers * kernel_block_size + block_offsets\n"
BOUNDED_SLOTS = SLOTS + "        slot_ids = tl.where(in_range, slot_ids, PAD_ID)\n"
HEADER = (
    "# Modified by GLM setup: the slot-mapping kernel reads a block table only inside its row.\n"
    "# Original vLLM Apache-2.0 notices below remain applicable.\n"
)


def patch_text(text):
    if "in_range = block_indices" in text:
        raise ValueError("Slot-mapping patch already applied")
    patched = replace_once(text, READ, BOUNDED_READ)
    patched = replace_once(patched, SLOTS, BOUNDED_SLOTS)
    patched = HEADER + patched
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
