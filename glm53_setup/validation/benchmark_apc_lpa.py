"""Measure APC/LPA break-even with an identical restored prefix in every arm."""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from .. import server, server_config
from ..io import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached-prefix-tokens", type=int, required=True)
    parser.add_argument(
        "--cold-only",
        action="store_true",
        help="Measure H=0 only, for an additional short-suffix sweep",
    )
    parser.add_argument(
        "--eligible-tokens", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096]
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    profile = server_config.load(args.config)
    if (
        not server_config.apc_lpa_enabled(profile)
        or profile["lpa"]["break_even_tokens"] != 0
        or profile["lpa"]["cut"] != 32
        or not profile["lpa"]["skip_mla_queries"]
        or profile["context"]["max_num_seqs"] != 1
        or profile["profiling"]["enabled"]
        or args.cached_prefix_tokens < 1
        or args.repeats < 3
        or not args.eligible_tokens
        or min(args.eligible_tokens) < 0
    ):
        raise ValueError(
            "Calibration requires serial APC/LPA, B=0, and at least three repeats"
        )
    raw = args.corpus.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.corpus_sha256:
        raise ValueError("Calibration corpus digest mismatch")
    documents = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    text = "\n\n".join(row["text"] for row in documents if row["split"] == "validation")
    if not text:
        raise ValueError("Calibration requires the validation split")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "starting",
        "profile": profile,
        "corpus_sha256": args.corpus_sha256,
        "cases": [],
    }

    def save():
        write_json(args.output / "result.json", report)

    def post(path, body):
        return server.post(profile, path, body)

    def reset():
        if post("/reset_prefix_cache", {}) != {"success": True}:
            raise ValueError(
                "Cache reset did not complete; do not compare these conditions"
            )

    def generate(ids, mode):
        body = {
            "model": profile["api"]["served_model_name"],
            "prompt": ids,
            "temperature": 0,
            "seed": 42,
            "max_tokens": 1,
            "ignore_eos": True,
            "logprobs": 5,
            "return_token_ids": True,
            "vllm_xargs": {"glm53_lpa_mode": mode},
        }
        began = time.perf_counter()
        response = post("/v1/completions", body)
        elapsed = time.perf_counter() - began
        workers = post(
            "/collective_rpc", {"method": "apc_lpa_report", "kwargs": {}, "timeout": 60}
        )["results"]
        choice = response["choices"][0]
        if (
            response["usage"]["prompt_tokens"] != len(ids)
            or response["usage"]["completion_tokens"] != 1
            or len(choice["token_ids"]) != 1
            or not all(
                math.isfinite(value) for value in choice["logprobs"]["token_logprobs"]
            )
            or len(workers) != 2
        ):
            raise ValueError("Invalid full-model calibration response")
        return {"seconds": elapsed, "response": response, "workers": workers}

    def check(row, length, hit, eligible, approximate):
        for worker in row["workers"]:
            policy = worker["policy"]["policy"]
            if (
                policy["prompt_tokens"] != length
                or policy["cached_tokens"] != hit
                or worker["eligible_tokens"] != eligible
            ):
                raise ValueError(
                    "Actual N/H/R differs from the paired calibration condition"
                )
            counts = worker["lpa"]["mla_queries_skipped"] if worker["lpa"] else {}
            expected = (
                {str(layer): eligible for layer in (35, 39, 43)}
                if approximate and eligible
                else {}
            )
            if counts != expected:
                raise ValueError(("Actual approximation differs", counts, expected))

    save()
    try:
        with server.request_lock():
            state, info = server.running_head(profile)
            if not info["State"]["Running"]:
                raise ValueError("The dedicated calibration server is not running")
            report.update(
                status="running", container=state["name"], image=info["Image"]
            )
            ids = post(
                "/tokenize",
                {
                    "model": profile["api"]["served_model_name"],
                    "prompt": text,
                    "add_special_tokens": False,
                },
            )["tokens"]
            tail = profile["lpa"]["tail"]
            for hit in (0,) if args.cold_only else (0, args.cached_prefix_tokens):
                for eligible in args.eligible_tokens:
                    length = hit + tail + eligible
                    if length + 1 > profile["context"]["max_model_len"] or length > len(
                        ids
                    ):
                        raise ValueError(
                            "Calibration case exceeds available context or validation data"
                        )
                    prompt = ids[:length]
                    case = {
                        "N": length,
                        "H": hit,
                        "R": eligible,
                        "prompt_sha256": hashlib.sha256(
                            json.dumps(prompt).encode()
                        ).hexdigest(),
                        "samples": [],
                    }
                    report["cases"].append(case)
                    for repeat in range(args.repeats + 1):
                        for arm, mode in (
                            ("off", "off"),
                            ("lpa", "auto"),
                            ("restored", "off"),
                        ):
                            reset()
                            if hit:
                                prime = generate(prompt[: hit + 1], "off")
                                check(prime, hit + 1, 0, max(0, hit + 1 - tail), False)
                            row = generate(prompt, mode)
                            check(row, length, hit, eligible, mode == "auto")
                            row.update(arm=arm, repeat=repeat, warmup=repeat == 0)
                            case["samples"].append(row)
                            save()
                    timings = {
                        arm: [
                            row["seconds"]
                            for row in case["samples"]
                            if row["arm"] == arm and not row["warmup"]
                        ]
                        for arm in ("off", "lpa", "restored")
                    }
                    case["median_seconds"] = {
                        arm: statistics.median(values)
                        for arm, values in timings.items()
                    }
                    case["ranges_seconds"] = {
                        arm: [min(values), max(values)]
                        for arm, values in timings.items()
                    }
                    save()
                    print(hit, eligible, case["median_seconds"], flush=True)
            reset()
            report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
