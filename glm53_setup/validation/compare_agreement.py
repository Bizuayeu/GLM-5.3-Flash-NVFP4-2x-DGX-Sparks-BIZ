"""Compare two ``run_agreement_fixture`` records: top-k agreement, full-vocabulary
KL and sparse-MLA candidate sets. The first record is the reference."""

import argparse
import json
from pathlib import Path

from .indexer_overlap import compare_candidates


def kl_rows(reference, candidate):
    """KL(reference || candidate) per row from float32 log-probabilities, in nats."""
    import numpy as np

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Log-probability tables differ in shape")
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        raise ValueError("Non-finite log-probability")
    return (np.exp(reference) * (reference - candidate)).sum(axis=1)


def distribution_shift(reference, candidate, block=128):
    """Mean, high quantile and maximum KL plus full-vocabulary argmax agreement."""
    import numpy as np

    if reference.shape != candidate.shape or not reference.shape[0]:
        raise ValueError("Log-probability tables differ in shape or are empty")
    kl, same = [], 0
    for start in range(0, reference.shape[0], block):  # bounds float64 copies
        left, right = reference[start : start + block], candidate[start : start + block]
        kl.extend(kl_rows(left, right).tolist())
        same += int((left.argmax(axis=1) == right.argmax(axis=1)).sum())
    kl = np.asarray(kl)
    return {
        "positions": int(kl.size),
        "kl_mean": float(kl.mean()),
        "kl_p99": float(np.quantile(kl, 0.99)),
        "kl_max": float(kl.max()),
        "kl_max_position": int(kl.argmax()),
        "argmax_agreement": same / kl.size,
    }


def candidate_shift(reference_rows, candidate_rows):
    """Jaccard of the selected sets at every captured layer and query position."""
    left = {(row["layer"], row["query_position"]): row for row in reference_rows}
    right = {(row["layer"], row["query_position"]): row for row in candidate_rows}
    if left.keys() != right.keys() or not left:
        raise ValueError("Candidate captures cover different queries")
    scores = [compare_candidates(left[k], right[k])["jaccard"] for k in sorted(left)]
    worst = min(range(len(scores)), key=scores.__getitem__)
    return {
        "queries": len(scores),
        "identical_sets": sum(score == 1.0 for score in scores) / len(scores),
        "jaccard_mean": sum(scores) / len(scores),
        "jaccard_min": scores[worst],
        "jaccard_min_query": list(sorted(left)[worst]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import numpy as np

    from ..agreement import compare_records

    def first_repeat(root):
        record = json.loads((root / "result.json").read_text(encoding="utf-8"))
        record["texts"] = [row for row in record["texts"] if row["repeat"] == 0]
        return record

    reference, candidate = first_repeat(args.reference), first_repeat(args.candidate)
    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "top_k": compare_records(reference, candidate),
        "full_vocabulary": {},
        "candidates": candidate_shift(
            reference["candidates"][0]["rows"], candidate["candidates"][0]["rows"]
        ),
    }
    for row in reference["texts"]:
        name = f"logprobs-{row['name']}.npy"
        report["full_vocabulary"][row["name"]] = distribution_shift(
            np.load(args.reference / name, mmap_mode="r"),
            np.load(args.candidate / name, mmap_mode="r"),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
