"""Zero-padding for tensor-parallel sizes that do not divide GLM-5.3-Flash's geometry.

The pinned vLLM shards 64 attention and KDA heads, a 2,048-wide MoE intermediate and a
154,880-token vocabulary evenly across ranks, so TP=3 stops at config time. With
``GLM53_TP_PAD_MULTIPLE=m`` the patched config reports ``ceil(64 / m) * m`` heads (66
for m=3) and the smallest MoE width of at least 2,048 whose per-rank share (width / m)
is a multiple of 64 (2,112, 704 per rank), and the vocabulary pads to ``lcm(64, m)``
(192). The patched loaders zero-extend a checkpoint tensor before they take a rank's
shard, so padded heads, rows and columns are exactly zero; Stage 0 of the TP=3 plan
measured that as exact for KDA, MLA, shared experts and the vocabulary, and within the
sharding's own error for the Marlin NVFP4 MoE (records/20260929-tp3-stage0). The knob is
independent of the tensor-parallel size: TP=1 with m=3 reads the padded model on one
GPU. Unset (or 1) every patched path is the pinned code: the config is unchanged and the
loaders run the pinned narrow. ``patch_tp_padding`` installs the call sites. The
mechanism (pad before the narrow, and a config view the MTP draft also reads) follows
FlyCockpit's MIT recipe for TP=3 on three DGX Sparks; no code is taken from it.
"""

# cc-defer: FlashInferMLASparseMetadataBuilder (v1/attention/backends/mla/
# flashinfer_mla_sparse.py) takes its reorder-batch threshold from a table keyed
# by heads per rank, {8, 16, 32: 128, 64: 256, 128: 1024}, default 1024. 22 heads
# (TP=3) and 66 (TP=1 with m=3) fall to 1024, so steps of up to 1024 query tokens
# count as decodes (TP=2's 32 heads: 128). Left as pinned; decide in Stage 4 from
# the TP=3 decode/prefill measurements, or earlier if a fixture difference is
# traced to that split.

import math
import os

ENV = "GLM53_TP_PAD_MULTIPLE"
# The pinned checkpoint's config.json.
HEADS = 64
MOE_WIDTH = 2048
VOCAB = 154880
# Marlin's N granularity (marlin_utils N % 64); a padded MoE width keeps each
# rank's share a multiple of it. vLLM pads every vocabulary to 64 by default.
ALIGN = 64


def pad_multiple(environ=None):
    """The multiple the heads, MoE width and vocabulary pad to; 1 when unset."""
    environ = os.environ if environ is None else environ
    value = environ.get(ENV)
    if value is None:
        return 1
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"{ENV} must be a positive integer, got {value!r}")
    return int(value)


def required_multiple(tp_size, heads=HEADS, moe_width=MOE_WIDTH, vocab=VOCAB):
    """The multiple a tensor-parallel size needs: 1 when it divides everything."""
    divides = (
        heads % tp_size == 0
        and moe_width % (ALIGN * tp_size) == 0
        and math.ceil(vocab / ALIGN) * ALIGN % tp_size == 0
    )
    return 1 if divides else tp_size


def padded_heads(heads, multiple):
    return math.ceil(heads / multiple) * multiple


def padded_moe_width(width, multiple):
    """The smallest width >= ``width`` divisible by ``multiple`` in 64-wide shares."""
    if multiple == 1:
        return width
    step = ALIGN * multiple
    return math.ceil(width / step) * step


def vocab_padding_size(padding_size, multiple=None):
    """``VocabParallelEmbedding``'s padding, widened so every rank gets whole rows."""
    multiple = pad_multiple() if multiple is None else multiple
    return padding_size if multiple == 1 else math.lcm(padding_size, multiple)


def pad_text_config(config, kwargs, multiple=None):
    """Pad a ``Glm5NextTextConfig`` in its ``__init__``, before ``super().__init__``.

    ``kwargs`` still holds the checkpoint's ``linear_attn_config`` dict; it is replaced
    by a padded copy because a later to_dict/from_dict round trip reads the dict before
    ``linear_num_heads``. ``head_dim`` and ``vocab_size`` stay as the checkpoint has them.
    """
    multiple = pad_multiple() if multiple is None else multiple
    if multiple == 1:
        return
    for name in ("num_attention_heads", "num_key_value_heads", "linear_num_heads"):
        setattr(config, name, padded_heads(getattr(config, name), multiple))
    config.moe_intermediate_size = padded_moe_width(
        config.moe_intermediate_size, multiple
    )
    linear = kwargs.get("linear_attn_config")
    if linear and "num_heads" in linear:
        kwargs["linear_attn_config"] = dict(
            linear, num_heads=padded_heads(linear["num_heads"], multiple)
        )


def pad_then_narrow(tensor, dim, start, length, multiple=None):
    """A rank's shard of ``tensor``, zero-extended along ``dim`` where it runs past the end.

    With the multiple at 1 this is ``tensor.narrow(dim, start, length)``, the pinned call.
    """
    multiple = pad_multiple() if multiple is None else multiple
    if multiple == 1:
        return tensor.narrow(dim, start, length)
    size = tensor.size(dim)
    missing = start + length - size
    if missing <= 0:
        return tensor.narrow(dim, start, length)
    if missing >= length:
        raise ValueError(
            "A shard of padding only; the multiple does not fit this shape"
        )
    # new_zeros and copy_, not cat or pad: both also hold for float8 scales.
    shape = list(tensor.shape)
    shape[dim] = start + length
    padded = tensor.new_zeros(shape)
    padded.narrow(dim, 0, size).copy_(tensor)
    return padded.narrow(dim, start, length)
