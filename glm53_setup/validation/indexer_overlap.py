"""Compare captured logical Top-K sets without treating padding as overlap."""

import argparse
import json
from pathlib import Path


def compare_candidates(source, target):
    for key in ("request_id", "query_position", "coordinate_space"):
        if source.get(key) != target.get(key) or key not in source:
            raise ValueError(
                "Compare the same request/query in the same coordinate space"
            )
    if source["coordinate_space"] != "logical_tokens":
        raise ValueError("Physical cache slots/pool IDs must be normalized separately")
    position = source["query_position"]
    if type(position) is not int or position < 0:
        raise ValueError("Invalid query position")

    def valid_set(values):
        if any(type(i) is not int or i < -1 or i > position for i in values):
            raise ValueError("Invalid or non-causal logical candidate")
        return {i for i in values if i >= 0}

    first, second = valid_set(source["indices"]), valid_set(target["indices"])
    union, shared = first | second, first & second
    result = {
        "request_id": source["request_id"],
        "query_position": position,
        "source_count": len(first),
        "target_count": len(second),
        "jaccard": len(shared) / len(union) if union else None,
        "target_recall": len(shared) / len(second) if second else None,
        "trivial_full_coverage": len(first) == len(second) == position + 1,
    }
    if "candidate_pool" in source:
        pool = valid_set(source["candidate_pool"])
        if not first <= pool:
            raise ValueError(
                "Expanded candidate pool must include the native selection"
            )
        result["candidate_pool_size"] = len(pool)
        result["target_recall_in_pool"] = (
            len(second & pool) / len(second) if second else None
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture", type=Path, help="JSON containing aligned source/target pairs"
    )
    args = parser.parse_args(argv)
    pairs = json.loads(args.capture.read_text(encoding="utf-8"))["pairs"]
    print(
        json.dumps(
            [compare_candidates(pair["source"], pair["target"]) for pair in pairs],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
