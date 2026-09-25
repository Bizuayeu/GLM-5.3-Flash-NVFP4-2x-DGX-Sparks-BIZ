"""Exclusive full-target component A/B/A and indexer observation; no LPA/MTP."""

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
from functools import partial
from pathlib import Path

from glm53_setup import model_http, server, server_config
from glm53_setup.io import write_json
from glm53_setup.validation.indexer_overlap import compare_candidates


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--corpus", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    return cli


def complete_tokens(profile, ids, count):
    began = time.perf_counter()
    response = server.post(
        profile,
        "/v1/completions",
        {
            "model": profile["api"]["served_model_name"],
            "prompt": ids,
            "max_tokens": count,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 42,
            "return_token_ids": True,
        },
    )
    elapsed = time.perf_counter() - began
    if (
        response["usage"]["completion_tokens"] != count
        or len(response["choices"][0].get("token_ids") or []) != count
    ):
        raise ValueError("Incomplete token-count evidence")
    return {"seconds": elapsed, "response": response}


def toggle_profiler(profile, endpoint):
    with model_http.open_response(
        server.api_origin(profile),
        "/" + endpoint,
        body={},
    ) as response:
        response.read()


def wait_ready(
    profile,
    *,
    open_response=model_http.open_response,
    sleep=time.sleep,
    clock=time.monotonic,
    deadline=1800,
):
    """Poll /health until it answers 200; an authentication failure ends the wait."""
    limit = clock() + deadline
    while clock() < limit:
        try:
            with open_response(
                server.api_origin(profile), "/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except model_http.ModelHTTPError as error:
            if error.code in (401, 403):
                raise
            sleep(5)
        except (urllib.error.URLError, TimeoutError):
            sleep(5)
    raise TimeoutError("Model readiness deadline exceeded")


def mode_agreement(modes):
    """Whether every mode produced one completion, and whether the native runs did."""
    signatures = {
        name: {tuple(s["response"]["choices"][0]["token_ids"]) for s in row["samples"]}
        for name, row in modes.items()
    }
    return {
        "tokens_equal": len(set.union(*signatures.values())) == 1,
        "native_repeatable": len(signatures["off"] | signatures["restored"]) == 1,
    }


def overlap_rows(rows):
    """Candidate overlap between layers one, two and three apart at each query."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_position"], {})[row["layer"]] = row
    overlap = []
    for layers in grouped.values():
        order = sorted(layers)
        for gap in (1, 2, 3):
            for first, last in zip(order, order[gap:]):
                overlap.append(
                    {
                        "source_layer": first,
                        "target_layer": last,
                        **compare_candidates(layers[first], layers[last]),
                    }
                )
    return overlap


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    profile = server_config.load(args.config)
    if not profile["validation"]["component_worker"]:
        cli.error("Explicit component worker profile required")
    args.output.mkdir(parents=True, exist_ok=False)
    complete = partial(complete_tokens, profile)
    profile_toggle = partial(toggle_profiler, profile)
    report = {
        "status": "waiting",
        "profile": profile,
        "speed": [],
        "indexer": [],
        "profiles": [],
    }

    def save():
        write_json(args.output / "result.json", report)

    def post(path, body):
        return server.post(profile, path, body)

    rpc = partial(server.collective_rpc, profile)

    save()
    try:
        wait_ready(profile)
        current, info = server.running_head(profile)
        report["image"] = info["Image"]
        report["container"] = current["name"]
        with server.request_lock():
            text_bytes = args.corpus.read_bytes()
            report["corpus_sha256"] = hashlib.sha256(text_bytes).hexdigest()
            documents = [
                json.loads(line) for line in text_bytes.decode("utf-8").splitlines()
            ]
            text = "\n\n".join(
                d["text"] for d in documents if d["split"] == "validation"
            )
            ids = post(
                "/tokenize",
                {
                    "model": profile["api"]["served_model_name"],
                    "prompt": text,
                    "add_special_tokens": False,
                },
            )["tokens"][:8192]
            if len(ids) != 8192:
                raise ValueError("Insufficient validation corpus")
            report["status"] = "running"
            save()
            try:
                for length in (64, 2048, 8192):
                    for count in (1, 128):
                        case = {
                            "input_tokens": length,
                            "output_tokens": count,
                            "modes": {},
                        }
                        report["speed"].append(case)
                        for name, enabled in (
                            ("off", False),
                            ("fused", True),
                            ("restored", False),
                        ):
                            configuration = rpc("unpack_configure", enabled=enabled)
                            complete(ids[:length], count)
                            samples = [
                                complete(ids[:length], count)
                                for _ in range(5 if count == 1 else 3)
                            ]
                            case["modes"][name] = {
                                "configuration": configuration,
                                "samples": samples,
                                "median_seconds": statistics.median(
                                    s["seconds"] for s in samples
                                ),
                            }
                            save()
                            print(
                                length,
                                count,
                                name,
                                case["modes"][name]["median_seconds"],
                                flush=True,
                            )
                        case.update(mode_agreement(case["modes"]))
                        save()
                if profile["profiling"]["enabled"]:
                    for enabled in (False, True):
                        rpc("unpack_configure", enabled=enabled)
                        for count in (1, 33):
                            complete(ids[:64], count)
                            profile_toggle("start_profile")
                            try:
                                row = complete(ids[:64], count)
                            finally:
                                profile_toggle("stop_profile")
                            report["profiles"].append(
                                {
                                    "fused": enabled,
                                    "input_tokens": 64,
                                    "output_tokens": count,
                                    **row,
                                }
                            )
                            save()
                rpc("unpack_configure", enabled=False)
                for length in (2048, 8192):
                    case = {"input_tokens": length, "timing": [], "capture": []}
                    report["indexer"].append(case)
                    complete(ids[:length], 1)
                    for iteration in range(5):
                        rpc(
                            "indexer_capture_start",
                            request_id=f"validation-{length}-timing-{iteration}",
                        )
                        began = time.perf_counter()
                        complete(ids[:length], 1)
                        elapsed = time.perf_counter() - began
                        workers = rpc("indexer_capture_finish")
                        case["timing"].append({"seconds": elapsed, "workers": workers})
                        save()
                    rpc(
                        "indexer_capture_start",
                        request_id=f"validation-{length}-capture",
                        positions=[p for p in (2047, 4095, 8191) if p < length],
                    )
                    complete(ids[:length], 1)
                    case["capture"] = rpc("indexer_capture_finish")
                    for worker in case["capture"]:
                        worker["overlap"] = overlap_rows(worker["rows"])
                    save()
                    print("indexer", length, "captured", flush=True)
                report["status"] = "complete"
                save()
            finally:
                rpc("indexer_capture_abort")
                rpc("unpack_configure", enabled=False)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
