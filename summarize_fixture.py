"""Assess only named generation records, never command/config JSON files."""

import argparse
import json
import math
from pathlib import Path

LABELS = [
    "a-first",
    "b",
    "a-after-b",
    "a-b-batch",
    "forced-1",
    "forced-7",
    "forced-15",
    "long",
]


def assess_outputs(outputs):
    if set(outputs) not in (set(LABELS), set(LABELS) | {"long-forced"}):
        raise ValueError("Generation record set is incomplete or unexpected")
    first, other, replay = (outputs[k][0] for k in ["a-first", "b", "a-after-b"])
    batch = outputs["a-b-batch"]
    complete = all(
        len(rows) == (2 if label == "a-b-batch" else 1)
        and all(
            len(row["token_ids"]) == len(row["logprobs"]) == 16
            and all(
                str(token) in lp
                for token, lp in zip(row["token_ids"], row["logprobs"], strict=True)
            )
            for row in rows
        )
        for label, rows in outputs.items()
    )
    finite = all(
        math.isfinite(v)
        for rows in outputs.values()
        for row in rows
        for lp in row["logprobs"]
        for v in lp.values()
    )
    checks = {
        "all_generations_complete": complete,
        "all_logprobs_finite": finite,
        "a_after_b_tokens_equal": first["token_ids"] == replay["token_ids"],
        "a_after_b_logprobs_equal": first["logprobs"] == replay["logprobs"],
        "batch_a_tokens_equal": first["token_ids"] == batch[0]["token_ids"],
        "batch_b_tokens_equal": other["token_ids"] == batch[1]["token_ids"],
    }
    forced = []
    for position in [1, 7, 15]:
        result = outputs[f"forced-{position}"][0]
        token = first["token_ids"][position]
        expected = first["logprobs"][position][str(token)]
        actual = result["logprobs"][0].get(str(token))
        delta = None if actual is None else abs(actual - expected)
        # Fixture-only provisional bound: two BF16 ulps at reference magnitude.
        tolerance = 2**-6 * max(1.0, abs(expected))
        forced.append(
            {
                "position": position,
                "next_token_equal": result["token_ids"][0] == token,
                "selected_logprob_delta": delta,
                "tolerance": tolerance,
                "within_tolerance": delta is not None and delta <= tolerance,
            }
        )
    checks["forced_prefill_matches_decode"] = all(
        x["next_token_equal"] and x["within_tolerance"] for x in forced
    )
    boundary = None
    if "long-forced" in outputs:
        source, result = outputs["long"][0], outputs["long-forced"][0]
        token = source["token_ids"][1]
        expected = source["logprobs"][1][str(token)]
        actual = result["logprobs"][0].get(str(token))
        delta = None if actual is None else abs(actual - expected)
        tolerance = 2**-6 * max(1.0, abs(expected))
        boundary = {
            "next_token_equal": result["token_ids"][0] == token,
            "selected_logprob_delta": delta,
            "tolerance": tolerance,
        }
        checks["boundary_prefill_matches_decode"] = (
            boundary["next_token_equal"] and delta is not None and delta <= tolerance
        )
    return {
        "checks": checks,
        "forced_prefill": forced,
        "boundary_prefill": boundary,
        "passed": all(checks.values()),
        "full_model_inference_validated": False,
    }


def assess_directory(directory):
    outputs = {}
    for label in LABELS:
        path = directory / (label + ".json")
        if label == "long" and not path.exists():
            path = directory / "long-1300.json"
        outputs[label] = json.loads(path.read_text())
    if (directory / "long-forced.json").exists():
        outputs["long-forced"] = json.loads(
            (directory / "long-forced.json").read_text()
        )
    return assess_outputs(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = assess_directory(args.directory)
    (args.directory / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)
