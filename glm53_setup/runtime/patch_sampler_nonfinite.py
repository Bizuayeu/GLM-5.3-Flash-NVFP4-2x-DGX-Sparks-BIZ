"""Source-pinned patch: the sampler kernels never emit a token id past the vocabulary.

Three Triton kernels of vLLM 385dce36 take an argmax over a ``BLOCK_SIZE``-wide tile whose lanes
past ``vocab_size`` load as ``-inf``, and store ``block_idx * BLOCK_SIZE + idx``: the Gumbel
sampler (``_gumbel_sample_kernel``), the greedy branch of the rejection sampler's local stats
(``_compute_local_logits_stats_kernel``, taken at temperature 0 under MTP) and its resampling
(``_resample_kernel``). When every in-vocabulary lane of the last tile is non-finite, the argmax
can settle on a lane past the vocabulary, and an id the embedding reads as zeros reaches the
output if that tile wins. A row with any finite logit takes its token from a tile holding one,
so the clamp leaves its output unchanged.
Upstream: vllm-project/vllm pull request #50843 by alexbi29 ("Bound tile-local argmax to
vocab_size in samplers", open, head ``3737f51f``), Apache-2.0 as vLLM; this is its change to the
two source files, with its comments, on the pinned source (its two tests are not in the image).
"""

# cc-defer: carries an upstream pull request that is still open; drop it (and the
# Dockerfile RUN) when the vLLM pin moves past the commit that merges #50843, or
# redo it from the merged diff if upstream changes it before merging.

from .pinned_patch import main_files, prepare_files, replace_once

GUMBEL = "v1/worker/gpu/sample/gumbel.py"
REJECTION = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
# Both as pinned (git show 385dce36:vllm/<target>).
SOURCES = {
    GUMBEL: "3ec1df510bdad13e8b0a457b5b67c98affdf6e57369e56cb471246cc8f40dd34",
    REJECTION: "20ca2e5ac34e9ef93dca388bed72a00d4ff67d569ec6ec2393b21a92d74a1fae",
}
RECORD = "glm53-sampler-nonfinite-patch.json"
HEADER = (
    "# Modified by GLM setup: tile-local argmax bounded to the vocabulary\n"
    "# (vllm-project/vllm #50843). Original vLLM Apache-2.0 notices below remain applicable.\n"
)

UNBOUNDED = "token_id = block_idx * BLOCK_SIZE + idx\n"
BOUNDED = "token_id = tl.minimum(block_idx * BLOCK_SIZE + idx, vocab_size - 1)\n"

GUMBEL_EDIT = (
    "    " + UNBOUNDED,
    "    # `idx` is the argmax over a BLOCK_SIZE-wide tile whose out-of-vocab tail\n"
    "    # lanes are loaded as -inf. If the in-vocab lanes are all non-finite, the\n"
    "    # reduction can settle on a tail lane and yield token_id >= vocab_size.\n"
    "    # Clamp to provide an addressable in-vocabulary fallback. This does not\n"
    "    # define a semantically correct token or fix the source of non-finite logits.\n"
    "    " + BOUNDED,
)
# The two sites differ in indent; each anchor takes the line before it as well.
GREEDY = "        value, idx = tl.max(target_logits, axis=0, return_indices=True)\n"
RESAMPLE = "        USE_FP64=USE_FP64,\n    )\n"
REJECTION_EDITS = (
    (
        GREEDY + "        " + UNBOUNDED,
        GREEDY
        + "        # Out-of-vocab tail lanes are loaded as -inf. If the in-vocab lanes are\n"
        "        # all non-finite, the reduction can settle on a tail lane and yield\n"
        "        # token_id >= vocab_size. Clamp to provide an addressable in-vocabulary\n"
        "        # fallback; this neither chooses a semantically correct token nor fixes\n"
        "        # the source of non-finite logits.\n"
        "        " + BOUNDED,
    ),
    (
        RESAMPLE + "    " + UNBOUNDED,
        RESAMPLE + "    # Preserve the bounded-fallback contract from\n"
        "    # _compute_local_logits_stats_kernel for the resampling path.\n"
        "    " + BOUNDED,
    ),
)


def patch_edits(text, target, edits):
    if BOUNDED in text:
        raise ValueError("sampler vocab bound patch already applied")
    patched = text
    for old, new in edits:
        patched = replace_once(patched, old, new)
    patched = HEADER + patched
    compile(patched, target, "exec")
    return patched


def patch_gumbel(text):
    return patch_edits(text, GUMBEL, (GUMBEL_EDIT,))


def patch_rejection(text):
    return patch_edits(text, REJECTION, REJECTION_EDITS)


PATCHES = {GUMBEL: patch_gumbel, REJECTION: patch_rejection}


def prepare(package):
    """Check both files against their pinned hashes before patching either."""
    return prepare_files(
        package, SOURCES, PATCHES, "sampler vocab bound source hash mismatch: "
    )


def main(argv=None):
    main_files(argv, doc=__doc__, sources=SOURCES, prepare=prepare, record=RECORD)


if __name__ == "__main__":
    main()
