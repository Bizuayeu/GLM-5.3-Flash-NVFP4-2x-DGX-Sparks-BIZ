"""One-kernel FP8 MLA latent unpack; preserves FP32 scale multiplication."""

import triton
import triton.language as tl


# elements is a run-time argument: as a constexpr every distinct size compiled
# and kept its own kernel, which leaks for a caller whose row count varies.
@triton.jit
def _unpack(packed, output, elements, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < elements
    row, column = offset // 512, offset % 512
    bits = tl.load(packed + row * 656 + column, valid, other=0)
    value = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale_ptr = packed.to(tl.pointer_type(tl.float32))
    scale = tl.load(scale_ptr + row * 164 + 128 + column // 128, valid, other=0.0)
    tl.store(output + offset, value * scale, valid)


def unpack_latent_cuda(packed):
    import torch

    if packed.dtype != torch.uint8 or packed.ndim < 1 or packed.shape[-1] != 656:
        raise ValueError("Expected 656-byte MLA records")
    if not packed.is_cuda or not packed.is_contiguous() or packed.storage_offset() % 4:
        raise ValueError("Fused unpack requires aligned contiguous CUDA records")
    output = torch.empty(
        (*packed.shape[:-1], 512), device=packed.device, dtype=torch.float32
    )
    if output.numel():
        # Two records per block bounds registers in this memory-bound pointwise
        # kernel. This is an initial launch shape, not a hardware optimum claim.
        _unpack[(triton.cdiv(output.numel(), 1024),)](
            packed, output, output.numel(), BLOCK=1024, num_warps=4
        )
    return output
