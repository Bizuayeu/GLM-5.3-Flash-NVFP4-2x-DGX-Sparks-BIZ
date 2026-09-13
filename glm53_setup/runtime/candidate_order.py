"""Canonical logical candidate order at the shared sparse-MLA boundary."""

import os


def candidate_order_enabled():
    value = os.environ.get("GLM53_CANONICAL_CANDIDATES", "1")
    if value not in ("0", "1"):
        raise ValueError("GLM53_CANONICAL_CANDIDATES must be 0 or 1")
    return value == "1"


def canonical_logical_candidates(indices):
    """Return ascending logical IDs with -1 padding last; preserve multiplicity.

    Call before physical cache mapping. Do not mutate the shared indexer buffer
    or change selection, scores, row ownership, shape, dtype, or device.
    """
    import torch

    if (
        not isinstance(indices, torch.Tensor)
        or indices.ndim != 2
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected an int32/int64 logical candidate matrix")
    if not indices.numel():
        return indices.clone()
    valid = (indices >= -1).all()
    if indices.device.type == "cuda":
        torch._assert_async(valid, "Only -1 is a valid candidate padding ID")
    elif not bool(valid):
        raise ValueError("Only -1 is a valid candidate padding ID")
    # Flipping the sign bit maps nonnegative IDs to ascending negative keys
    # and -1 to the largest key. Unlike a max-ID sentinel, it has no collision
    # with a valid maximum integer and requires no widening or arithmetic wrap.
    keys = torch.bitwise_xor(indices, torch.iinfo(indices.dtype).min)
    return indices.gather(1, torch.argsort(keys, dim=1))
