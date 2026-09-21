"""Diagnostic record of why MTP drafts are rejected: draft confidence beside the target's verdict.

With ``GLM53_DRAFT_OBSERVE=<file>`` (rank 0 only, ``mtp.draft_observe``) the patched V2 runner
appends one JSON line per request and verification step:

    r   request id
    p   output tokens before the step (absolute position of the step's first row + 1 - prompt)
    a   accepted drafts
    d   draft token ids that were verified, one per depth
    dp  the draft's probability of its own top-1 at each depth
    dm  the draft's top-1 minus top-2 logit at each depth (raw dtype; 0 is an exact tie)
    d5  the draft's top-5 ids at each depth. The draft is their argmax: d5[i][0], or on an exact
        tie (dm 0) another entry; otherwise the draft-side row is from another step (unpaired)
    t   the target's argmax id at each of the depth + 1 verified positions
    tl  the target's log-probability of the draft token, per depth
    tm  the target's top-1 minus top-2 logit at each of the depth + 1 positions (raw dtype)

Draft side: the greedy draft already computes full logits on every rank unless
``use_local_argmax_reduction`` is set, so ``draft`` runs the same ``compute_logits`` + ``argmax``
and adds a top-5 and a log-sum-exp over one row per request, kept on the GPU in a buffer indexed
by request slot and draft depth. With the local reduction it records nothing and drafts as usual:
rank 0 alone taking the full-logits path would call a different collective than rank 1. Nothing
here contains a collective, so a rank that observes and one that does not run the same ones.

Target side: ``target`` takes the verifier's full logits (all depth + 1 positions are at hand in
the pinned runner) before the rejection sampler touches them; ``write`` copies the small results to
the host after it. That copy is a GPU synchronisation per step, which the asynchronous scheduler
otherwise avoids: expect a few milliseconds per step. A diagnostic arm, not a serving setting.
"""

import atexit
import json
import os
import time

TOP = 5
FLUSH_SECONDS = 1.0
_active = None
_lines = None


def active():
    global _active
    if _active is None:
        _active = bool(os.environ.get("GLM53_DRAFT_OBSERVE"))
    return _active


def draft_side():
    """The draft's confidence is computed for the record and for the gate (``draft_gate.py``)."""
    from . import draft_gate

    return active() or draft_gate.active()


def _close():
    if _lines is not None:
        _lines["file"].close()


def _append(rows):
    global _lines
    if _lines is None:
        path = os.environ["GLM53_DRAFT_OBSERVE"]
        _lines = {"file": open(path, "a", encoding="utf-8"), "flushed": 0.0}
        atexit.register(_close)
    for row in rows:
        _lines["file"].write(json.dumps(row, separators=(",", ":")) + "\n")
    now = time.monotonic()
    if now - _lines["flushed"] >= FLUSH_SECONDS:
        _lines["file"].flush()
        _lines["flushed"] = now


def draft(speculator, hidden_states, idx_mapping, draft_step):
    """The greedy draft token, as ``_greedy_sample_draft`` computes it, and its confidence."""
    import torch

    if speculator.use_local_argmax_reduction:
        return speculator._greedy_sample_draft(hidden_states)
    logits = speculator.model.compute_logits(hidden_states)
    tokens = logits.argmax(dim=-1)
    top_logits, top_ids = logits.topk(TOP, dim=-1)
    probability = (top_logits[:, 0].float() - logits.float().logsumexp(dim=-1)).exp()
    margin = (top_logits[:, 0] - top_logits[:, 1]).float()
    buffers = getattr(speculator, "glm53_draft_observe", None)
    if buffers is None:
        shape = (speculator.max_num_reqs, speculator.num_speculative_steps)
        buffers = (
            torch.zeros(*shape, 2, dtype=torch.float32, device=logits.device),
            torch.zeros(*shape, TOP, dtype=torch.int64, device=logits.device),
        )
        speculator.glm53_draft_observe = buffers
    rows = idx_mapping.long()
    columns = draft_step.long().expand_as(rows)
    buffers[0][rows, columns] = torch.stack((probability, margin), dim=-1)
    buffers[1][rows, columns] = top_ids
    return tokens


def target(logits, input_batch):
    """GPU-side summary of the verifier's logits; no synchronisation."""
    import torch

    top_logits, top_ids = logits.topk(2, dim=-1)
    inputs = input_batch.input_ids[input_batch.logits_indices]
    # Row r of a request verifies the input of its row r + 1, the draft of depth r + 1.
    following = torch.roll(inputs, -1).long()
    logprob = logits.float().log_softmax(dim=-1).gather(1, following[:, None])[:, 0]
    return {
        "ids": torch.stack(
            (
                inputs.long(),
                top_ids[:, 0],
                input_batch.positions[input_batch.logits_indices].long(),
            )
        ),
        "floats": torch.stack((logprob, (top_logits[:, 0] - top_logits[:, 1]).float())),
    }


def write(speculator, input_batch, pending, num_sampled):
    buffers = getattr(speculator, "glm53_draft_observe", None)
    if buffers is None:
        return
    slots = input_batch.idx_mapping.long()
    ids = pending["ids"].cpu().tolist()
    floats = pending["floats"].cpu().tolist()
    _append(
        rows_of(
            input_batch.req_ids,
            input_batch.cu_num_logits_np.tolist(),
            input_batch.prefill_len_np.tolist(),
            num_sampled.cpu().tolist(),
            ids,
            floats,
            buffers[0][slots].cpu().tolist(),
            buffers[1][slots].cpu().tolist(),
        )
    )


def rows_of(
    req_ids, cu_num_logits, prompt_lens, num_sampled, ids, floats, confidence, top
):
    """One record per request that verified drafts in this step (plain lists in, dicts out)."""
    inputs, argmax, positions = ids
    logprob, margin = floats
    rows = []
    for index, request in enumerate(req_ids):
        low, high = cu_num_logits[index], cu_num_logits[index + 1]
        depth = high - low - 1
        if depth <= 0:
            continue
        rows.append(
            {
                "r": request,
                "p": positions[low] + 1 - prompt_lens[index],
                "a": num_sampled[index] - 1,
                "d": inputs[low + 1 : high],
                "dp": [round(pair[0], 6) for pair in confidence[index][:depth]],
                "dm": [pair[1] for pair in confidence[index][:depth]],
                "d5": top[index][:depth],
                "t": argmax[low:high],
                "tl": [round(value, 5) for value in logprob[low : high - 1]],
                "tm": margin[low:high],
            }
        )
    return rows
