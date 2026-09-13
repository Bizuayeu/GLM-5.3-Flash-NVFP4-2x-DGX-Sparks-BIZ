"""Exclusive full-target component A/B/A and indexer observation; no LPA/MTP."""

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
from pathlib import Path

from glm53_setup import model_http, startup, startup_config
from glm53_setup.config import ROOT
from glm53_setup.io import write_json
from glm53_setup.validation.indexer_overlap import compare_candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile = startup_config.load(args.config)
    if not profile["validation"]["component_worker"]:
        parser.error("Explicit component worker profile required")
    args.output.mkdir(parents=True, exist_ok=False)
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
        return startup.post(profile, path, body)

    def rpc(method, **kwargs):
        return post(
            "/collective_rpc", {"method": method, "kwargs": kwargs, "timeout": 600}
        )["results"]

    def complete(ids, count):
        began = time.perf_counter()
        response = post(
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

    def profile_toggle(endpoint):
        with model_http.open_response(
            f"http://127.0.0.1:{profile['api']['port']}",
            "/" + endpoint,
            body={},
        ) as response:
            response.read()

    save()
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                with model_http.open_response(
                    f"http://127.0.0.1:{profile['api']['port']}", "/health", timeout=5
                ) as response:
                    if response.status == 200:
                        break
            except model_http.ModelHTTPError as error:
                if error.code in (401, 403):
                    raise
                time.sleep(5)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(5)
        else:
            raise TimeoutError("Model readiness deadline exceeded")
        current = startup.read_json(ROOT / "state/startup-rank0.json")
        info = startup.inspect_owned(
            current["name"], startup_config.fingerprint(profile)
        )
        report["image"] = info["Image"]
        report["container"] = current["name"]
        import fcntl

        with (ROOT / "state/startup-request.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
                        signatures = {
                            name: {
                                tuple(s["response"]["choices"][0]["token_ids"])
                                for s in row["samples"]
                            }
                            for name, row in case["modes"].items()
                        }
                        case["tokens_equal"] = len(set.union(*signatures.values())) == 1
                        case["native_repeatable"] = (
                            len(signatures["off"] | signatures["restored"]) == 1
                        )
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
                        grouped = {}
                        for row in worker["rows"]:
                            grouped.setdefault(row["query_position"], {})[
                                row["layer"]
                            ] = row
                        worker["overlap"] = []
                        for position, layers in grouped.items():
                            order = sorted(layers)
                            for gap in (1, 2, 3):
                                for first, last in zip(order, order[gap:]):
                                    worker["overlap"].append(
                                        {
                                            "source_layer": first,
                                            "target_layer": last,
                                            **compare_candidates(
                                                layers[first], layers[last]
                                            ),
                                        }
                                    )
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
