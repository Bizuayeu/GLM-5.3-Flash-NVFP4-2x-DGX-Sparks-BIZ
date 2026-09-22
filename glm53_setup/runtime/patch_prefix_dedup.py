"""Source-pinned patch: the block pool asks before registering a page under a hash it already holds.

vLLM 385dce36's ``BlockPool._insert_block_hash`` checks only whether *this* block is registered under
the hash and otherwise inserts it, so a recomputed page joins the cache beside its identical copy. This
patch adds one question before the insert, answered by ``glm53_setup.runtime.prefix_dedup`` from the
``GLM53_PREFIX_PAGE_DEDUP`` environment (off unless the profile sets ``runtime.prefix_page_dedup``).
Off, the pool behaves as pinned.
"""

from . import pinned_patch
from .pinned_patch import replace_once

TARGET = "v1/core/block_pool.py"
SOURCE_SHA256 = "0a4ba84de44cf7d569a26c1f3c0ba40a0e382fcd96f4de7bcc2616830c986b46"
MISMATCH = "block pool source hash mismatch"
RECORD = "glm53-prefix-dedup-patch.json"
IMPORT_ANCHOR = "logger = init_logger(__name__)\n"
IMPORT = (
    "from glm53_setup.runtime.prefix_dedup import (\n"
    "    skip_duplicate_page as glm53_skip_duplicate_page,\n"
    ")\n"
)
CONTAIN_CHECK = (
    "        if self.cached_block_hash_to_block.contain(\n"
    "            block_hash_with_group_id, block.block_id\n"
    "        ):\n"
    "            return\n"
    "\n"
)
DEDUP_CHECK = CONTAIN_CHECK + (
    "        if glm53_skip_duplicate_page(\n"
    "            self.cached_block_hash_to_block, block_hash_with_group_id\n"
    "        ):\n"
    "            return\n"
    "\n"
)
HEADER = (
    "# Modified by GLM setup: a page whose hash is already cached is not registered again.\n"
    "# Original vLLM Apache-2.0 notices below remain applicable.\n"
)


def patch_text(text):
    if "glm53_skip_duplicate_page" in text:
        raise ValueError("Prefix dedup patch already applied")
    patched = replace_once(text, IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT)
    patched = replace_once(patched, CONTAIN_CHECK, DEDUP_CHECK)
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
