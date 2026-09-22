"""One cached page per content: a block whose hash is already cached is not registered again.

The pinned vLLM's block pool registers every full block under its hash even when another block with
the same hash is already cached (``BlockHashToBlockMap``, NOTE #1: no de-duplication, so that block
tables stay append-only). Under a draft the prefix lookup drops the last matching block and recomputes
it, so every re-sent history adds one block per KV cache group under a hash that already has one; the
copies then sit in the LRU queue like any cached block and push other histories out first (on the
four-layer fixture, three blocks of about 9K tokens per re-send).

With ``GLM53_PREFIX_PAGE_DEDUP=1`` the pool skips the registration of such a block. The block itself is
untouched (its id and its content stay with the request that computed it), it simply carries no hash,
so when its request finishes it returns to the front of the free queue and is reused before any cached
block is evicted. Lookups keep hitting the copy that was cached first. Nothing numeric changes.
"""

import os


def dedup_enabled():
    value = os.environ.get("GLM53_PREFIX_PAGE_DEDUP", "0")
    if value not in ("0", "1"):
        raise ValueError("GLM53_PREFIX_PAGE_DEDUP must be 0 or 1")
    return value == "1"


def skip_duplicate_page(cached_block_hash_to_block, block_hash_with_group_id):
    """True when another block is already cached under this hash and dedup is on."""
    if not dedup_enabled():
        return False
    return (
        cached_block_hash_to_block.get_one_block(block_hash_with_group_id) is not None
    )
