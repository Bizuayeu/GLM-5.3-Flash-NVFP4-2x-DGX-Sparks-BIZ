"""Reject incomplete vLLM benchmark results even when its CLI exits successfully."""

import argparse
import json
import math
from pathlib import Path


def assess_result(data, requests, output_tokens):
    if requests <= 0 or output_tokens <= 0:
        raise ValueError("Expected request and output counts must be positive")
    metrics = ["mean_ttft_ms", "mean_tpot_ms", "mean_e2el_ms", "output_throughput"]
    checks = {
        "all_requests_completed": data.get("completed") == requests,
        "expected_output_tokens": data.get("total_output_tokens")
        == requests * output_tokens,
        "positive_input_tokens": isinstance(data.get("total_input_tokens"), int)
        and data["total_input_tokens"] > 0,
        "finite_positive_metrics": all(
            isinstance(data.get(k), (int, float))
            and math.isfinite(data[k])
            and data[k] > 0
            for k in metrics
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_requests": requests,
        "expected_output_tokens_per_request": output_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    args = parser.parse_args()
    result = assess_result(
        json.loads(args.result.read_text(encoding="utf-8")),
        args.requests,
        args.output_tokens,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
