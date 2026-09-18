"""One token order inside each expert before the Marlin MoE kernel.

The pinned vLLM's ``moe_align_block_size`` hands out slots with an atomic add from
many CUDA threads, so the order of the tokens inside an expert follows thread
scheduling and changes from call to call for identical input. The Marlin MoE result
depends slightly on a row's position, later routers amplify that, and identical
requests stop repeating. Upstream tracks it as vLLM issue #52525.
"""

import os


def moe_order_enabled():
    value = os.environ.get("GLM53_CANONICAL_MOE_ORDER", "1")
    if value not in ("0", "1"):
        raise ValueError("GLM53_CANONICAL_MOE_ORDER must be 0 or 1")
    return value == "1"


def canonical_expert_order(sorted_ids, expert_ids, padded, block_size):
    """Order the written slots by (expert, token id) without a host synchronisation.

    ``sorted_ids`` and ``expert_ids`` are worst-case buffers whose lengths need not
    match (the served model allocates more expert blocks than fit in the slots), so
    the block size comes from the caller; only the first ``padded`` slots (a device
    scalar) are written. The align kernel emits experts in
    ascending order, which is what keeps every token inside its expert's blocks here.
    Padding slots hold the number of real entries, which never exceeds the buffer
    length, so they sort last inside their expert without reaching the next one.
    Unwritten slots get the largest key and stay where they were. Selection, block
    ownership, shape and dtype are unchanged; the input is not mutated.
    """
    import torch

    owner = expert_ids.to(torch.int64).repeat_interleave(block_size)[
        : sorted_ids.numel()
    ]
    slots = torch.arange(sorted_ids.numel(), device=sorted_ids.device)
    key = owner * (sorted_ids.numel() + 1) + sorted_ids.to(torch.int64)
    key = torch.where(slots < padded, key, torch.iinfo(torch.int64).max)
    return sorted_ids[torch.argsort(key, stable=True)]
