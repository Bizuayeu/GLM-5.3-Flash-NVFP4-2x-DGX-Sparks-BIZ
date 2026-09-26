"""Experimental shared-K/V NoPE attention with explicit TF32x3 products.

Scores and value products use Triton's TF32x3 mode; softmax and accumulation are
FP32. This is not an IEEE-equivalence claim or a serving backend selection.
"""

import hashlib
import re

import triton
import triton.language as tl

from .fused_nope import checked_cache


@triton.jit
def _attention_dot(
    Query,
    Cache,
    Indices,
    Output,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(1).to(tl.int64)
    heads = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 512)
    query = tl.load(
        Query + token * Q0 + heads[:, None] * Q1 + columns[None, :],
        heads[:, None] < HEADS,
        other=0,
    ).to(tl.float32)
    accumulated = tl.full((16, 512), 0, tl.float32)
    maximum = tl.full((16,), float("-inf"), tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    cache_scales = Cache.to(tl.pointer_type(tl.float32))
    for start in range(0, tl.cdiv(WIDTH, BLOCK_K)):
        offsets = start * BLOCK_K + tl.arange(0, BLOCK_K)
        indices = tl.load(
            Indices + token * I0 + offsets * I1, offsets < WIDTH, other=-1
        )
        valid = (offsets < WIDTH) & (indices >= 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            safe = tl.maximum(indices, 0).to(tl.int64)
            bits = tl.load(
                Cache + safe[:, None] * 656 + columns[None, :],
                valid[:, None],
                other=0,
            )
            values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scales = tl.load(
                cache_scales + safe[:, None] * 164 + 128 + columns[None, :] // 128,
                valid[:, None],
                other=0,
            )
            kv = values * scales
            scores = tl.dot(query, tl.trans(kv), input_precision="tf32x3") * SCALE
            scores = tl.where(valid[None, :], scores, float("-inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum[:, None])
            accumulated = accumulated * alpha[:, None] + tl.dot(
                probabilities, kv, input_precision="tf32x3"
            )
            denominator = denominator * alpha + tl.sum(probabilities, 1)
            maximum = next_maximum
    result = accumulated / tl.maximum(denominator[:, None], 1.0)
    tl.store(
        Output + (token * HEADS + heads[:, None]) * 512 + columns[None, :],
        result,
        heads[:, None] < HEADS,
    )


def dot_nope_tf32x3(
    query, packed_cache, physical_indices, scale, *, tile=16, diagnostics=None
):
    import torch

    if type(tile) is not int or tile not in (16, 32):
        raise ValueError("TF32x3 candidate tile must be 16 or 32")
    cache = checked_cache(query, packed_cache, physical_indices)
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if query.shape[0] and query.shape[1]:
        kernel = _attention_dot[(triton.cdiv(query.shape[1], 16), query.shape[0])](
            query,
            cache,
            physical_indices,
            output,
            query.stride(0),
            query.stride(1),
            physical_indices.stride(0),
            physical_indices.stride(1),
            query.shape[1],
            physical_indices.shape[1],
            scale,
            BLOCK_K=tile,
            num_warps=8,
            num_stages=1,
            enable_fp_fusion=False,
        )
        if diagnostics is not None:
            ptx = kernel.asm["ptx"]
            diagnostics.update(
                precision="tf32x3",
                registers=kernel.n_regs,
                spills=kernel.n_spills,
                shared_bytes=kernel.metadata.shared,
                tf32_mma=bool(re.search(r"mma\.[^\n;]*\.tf32", ptx)),
                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
            )
    return output
