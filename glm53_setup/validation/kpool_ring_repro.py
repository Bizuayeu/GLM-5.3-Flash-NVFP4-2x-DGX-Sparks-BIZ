"""GPU kernel repro of the kpool raw-tail ring (vLLM #58454); no model weights required.

Runs inside the reference image, whose vLLM carries ``patch_kpool_seed`` and
``patch_kpool_ring``. A draft at position 7 completes pool 1 and is rejected;
the three drafts behind it were already stashed. With a one-pool ring (4 slots)
drafts 8 and 9 overwrite the slots of positions 4 and 5 that the redo of 7
reads, so pool 1 differs from the prefill writer's result on the true keys (the
no-speculation reference); with the ring the patched spec gives MTP depth 3 (8
slots) it matches. A control run whose pool-completing token is verified must
match with both rings. The scenario is adapted from vLLM's
``test_rejected_draft_redo_needs_ring_slots`` (Apache-2.0, pull request #58454).
"""

import argparse
import json
from pathlib import Path

HEAD_DIM = 128
POOL = 4
SPEC = 3
PAGE = 64
BLOCKS = 2


def pool_bytes(cache, pool):
    """Page layout: [PAGE * HEAD_DIM bytes of K rows | PAGE * 4 bytes of scales]."""
    import torch

    flat = cache[0].reshape(-1)
    k_bytes = flat[pool * HEAD_DIM : (pool + 1) * HEAD_DIM]
    scale = PAGE * HEAD_DIM
    s_bytes = flat[scale + 4 * pool : scale + 4 * (pool + 1)]
    return torch.cat([k_bytes, s_bytes])


def run(ops, torch, ring, seed):
    dev = "cuda"
    torch.manual_seed(seed)
    n_tok = 3 * POOL
    k = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    score = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    ape = torch.randn(POOL, HEAD_DIM, dtype=torch.float32, device=dev)
    drafts = torch.randn(SPEC, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    draft_scores = torch.randn(SPEC, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    reference = torch.zeros(BLOCKS, PAGE, HEAD_DIM + 4, dtype=torch.uint8, device=dev)
    ops.kpool_compress_and_write_cache(
        reference,
        k.view(3, POOL, HEAD_DIM),
        score.view(3, POOL, HEAD_DIM),
        ape,
        torch.arange(3, dtype=torch.int64, device=dev),
        pool_size=POOL,
        head_dim=HEAD_DIM,
        round_scale=True,
    )
    kv = torch.zeros_like(reference)
    tail = torch.zeros(BLOCKS, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=dev)

    def step(positions, keys, scores):
        pos = torch.tensor([positions], dtype=torch.int32, device=dev)
        slots = [(p // POOL) if p % POOL == POOL - 1 else -1 for p in positions]
        ops.kpool_decode_update_and_maybe_write_cache_batched(
            kv,
            tail,
            pos % ring,
            keys.view(1, -1, HEAD_DIM),
            scores.view(1, -1, HEAD_DIM),
            ape,
            torch.tensor([slots], dtype=torch.int32, device=dev),
            pos,
            POOL,
            HEAD_DIM,
            round_scale=True,
        )

    # Control: verified token 7 completes pool 1 before drafts 8..10 are stashed.
    for t in range(7):
        step([t], k[t], score[t])
    step(
        [7, 8, 9, 10],
        torch.cat([k[7:8], drafts]),
        torch.cat([score[7:8], draft_scores]),
    )
    step([8, 9, 10, 11], k[8:12], score[8:12])  # all drafts rejected
    control = all(
        torch.equal(pool_bytes(kv, p), pool_bytes(reference, p)) for p in (1, 2)
    )
    # Draft 7 completes pool 1 and is rejected; the redo of 7 reads slots 4..6.
    kv.zero_()
    tail.zero_()
    for t in range(6):
        step([t], k[t], score[t])
    step(
        [6, 7, 8, 9], torch.cat([k[6:7], drafts]), torch.cat([score[6:7], draft_scores])
    )
    step([7, 8, 9, 10], k[7:11], score[7:11])
    rejected = torch.equal(pool_bytes(kv, 1), pool_bytes(reference, 1))
    return {
        "ring": ring,
        "control_matches": control,
        "rejected_draft_matches": rejected,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    import torch
    from vllm.models.glm5next.nvidia.ops import kpool_compress as ops

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; the repro runs the Triton kernels")
    rows = [run(ops, torch, ring, args.seed) for ring in (POOL, 2 * POOL)]
    one_pool, spec_ring = rows
    report = {
        "scope": "kpool decode kernel only; not model output",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seed": args.seed,
        "rows": rows,
        # Expected on the patched image: the one-pool ring differs, the MTP-3 ring matches.
        "reproduced": one_pool["control_matches"]
        and spec_ring["control_matches"]
        and not one_pool["rejected_draft_matches"]
        and spec_ring["rejected_draft_matches"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
