"""Stop drafting once the draft is unsure: ``GLM53_DRAFT_GATE=<tau>`` (``mtp.draft_gate``).

On the reference pair the draft's own top-1 probability predicts whether its token holds (AUC
0.89 at every depth, 2026-09-21). After each draft depth the patched speculator asks ``stop``:
when the top-1 probability of the depth just drafted is at or under ``tau`` for every request of the
batch, drafting ends there. That draft is kept, its forward is already paid. The first depth is
always drafted. With several requests the batch drafts as deep as its most confident request.

The probability comes from ``draft_observe.draft`` (full draft logits, so not with
``use_local_argmax_reduction``; then the gate never stops). Reading it on the host is one GPU
synchronisation per drafted depth; the step is GPU-bound, so that costs a kernel-launch gap.

Tensor parallel ranks must leave the draft loop at the same depth or the next collective hangs.
The gathered logits should be identical on every rank, but the decision does not rest on that:
rank 0's flag is broadcast over the tensor-parallel group before anyone reads it. The variable
must therefore be set on every rank (the launcher does).

Fewer drafts shrink the next verification only under synchronous scheduling, where the scheduler
schedules the placeholders the runner returns. The asynchronous scheduler fixes their number
before the step runs, so ``stop`` refuses to work with it.
"""

import os

_tau = False  # False: not read yet; None: off


def tau():
    global _tau
    if _tau is False:
        raw = os.environ.get("GLM53_DRAFT_GATE")
        _tau = None if raw in (None, "") else float(raw)
        if _tau is not None and not 0.0 <= _tau <= 1.0:
            raise ValueError("GLM53_DRAFT_GATE must be a probability")
    return _tau


def active():
    return tau() is not None


def stop(speculator, num_reqs, drafted):
    """After ``drafted`` depths: has every request's last draft fallen under ``tau``?"""
    threshold = tau()
    if threshold is None:
        return False
    if drafted == 1:
        if speculator.vllm_config.scheduler_config.async_scheduling:
            raise ValueError("GLM53_DRAFT_GATE needs --no-async-scheduling")
        speculator.glm53_drafted = None
    buffers = getattr(speculator, "glm53_draft_observe", None)
    if buffers is None:
        return False
    rows = speculator.idx_mapping[:num_reqs].long()
    # At or under: a float32 softmax saturates at exactly 1.0, and tau = 1 must always stop.
    flag = (buffers[0][rows, drafted - 1, 0].max() <= threshold).int()
    flag = _broadcast(flag)
    if not flag.item():
        return False
    speculator.glm53_drafted = drafted
    return True


def _broadcast(flag):
    from vllm.distributed import get_tp_group

    return get_tp_group().broadcast(flag, src=0)


def drafted(speculator, depth):
    """Draft columns the step really filled: the runner hands the scheduler that many."""
    return min(depth, getattr(speculator, "glm53_drafted", None) or depth)
