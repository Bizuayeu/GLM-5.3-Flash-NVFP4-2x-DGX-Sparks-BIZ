"""Exercise teacher attention-input replay on a verified four-layer fixture."""

import argparse
import json
import math
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1, 3, 4, 5, 127, 128, 129, 511, 512, 513],
    )
    parser.add_argument("--cut", type=int, default=2)
    parser.add_argument("--skip-mla-queries", action="store_true")
    args = parser.parse_args(argv)
    status = json.loads((args.fixture / "fixture-status.json").read_text())
    config = json.loads((args.fixture / "config.json").read_text())
    if not (
        status.get("all_tensor_bytes_verified")
        and config.get("_test_fixture_only")
        and config["text_config"]["num_hidden_layers"] == 4
    ):
        raise ValueError("A verified four-layer fixture is required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "loading", "scope": "fixture-oracle-replay", "cases": []}

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    from vllm import LLM, SamplingParams

    from .run_fixture import encode_output

    llm = LLM(
        model=str(args.fixture),
        tensor_parallel_size=1,
        language_model_only=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=16384,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        block_size=256,
        kv_cache_dtype="fp8",
        kv_cache_memory_bytes=512 * 1024**2,
        gpu_memory_utilization=0.20,
        seed=42,
        worker_extension_cls="glm53_setup.runtime.llkv.LLKVWorkerExtension",
        kernel_config={
            "enable_flashinfer_autotune": False,
            "enable_cutedsl_warmup": False,
            "enable_jit_warmup": False,
            "moe_backend": "marlin",
            "linear_backend": "marlin",
        },
    )
    base = llm.get_tokenizer().encode(
        "Tokyo is the capital of Japan. The sequence is 2, 4, 6, 8. ",
        add_special_tokens=False,
    )
    params = SamplingParams(
        temperature=0,
        max_tokens=16,
        ignore_eos=True,
        logprobs=10,
        seed=42,
        detokenize=False,
    )
    report["status"] = "running"
    save()
    for length in args.lengths:
        ids = (base * (length // len(base) + 1))[:length]
        case = {"length": length, "modes": {}}
        report["cases"].append(case)
        for mode in ("capture", "off", "oracle_full_mlp", "oracle", "off"):
            key = mode if mode not in case["modes"] else "restored"
            llm.collective_rpc(
                "llkv_configure",
                kwargs={
                    "mode": "oracle" if mode == "oracle_full_mlp" else mode,
                    "cut": args.cut,
                    "prompt_length": length,
                    "tail": 1,
                    "profile": mode == "off",
                    "verify_state": True,
                    "skip_mlp": mode != "oracle_full_mlp",
                    "skip_mla_queries": args.skip_mla_queries,
                },
            )
            start = time.monotonic()
            result = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[
                0
            ]
            row = encode_output(result)
            row["seconds"] = time.monotonic() - start
            row["workers"] = llm.collective_rpc("llkv_report")
            case["modes"][key] = row
            save()
        reference = case["modes"]["off"]
        baseline = case["modes"]["capture"]
        oracle = case["modes"]["oracle"]
        restored = case["modes"]["restored"]
        case["baseline_token_equal"] = (
            reference["token_ids"] == baseline["token_ids"] == restored["token_ids"]
        )
        case["oracle_token_equal"] = reference["token_ids"] == oracle["token_ids"]
        differences = [
            abs(row[token] - oracle["logprobs"][i][token])
            for i, row in enumerate(reference["logprobs"])
            for token in row
            if token in oracle["logprobs"][i]
        ]
        case["max_shared_logprob_error"] = max(differences, default=None)
        exact = case["modes"]["oracle_full_mlp"]
        case["oracle_full_mlp_equal"] = reference["token_ids"] == exact["token_ids"]
        case["oracle_full_mlp_logprob_error"] = max(
            abs(row[token] - exact["logprobs"][i][token])
            for i, row in enumerate(reference["logprobs"])
            for token in row
            if token in exact["logprobs"][i]
        )
        case["finite"] = all(
            math.isfinite(v)
            for mode in case["modes"].values()
            for row in mode["logprobs"]
            for v in row.values()
        )
        case["passed"] = (
            case["baseline_token_equal"]
            and case["oracle_token_equal"]
            and case["finite"]
            and case["oracle_full_mlp_equal"]
        )
        save()
    report["passed"] = all(case["passed"] for case in report["cases"])
    report["status"] = "complete"
    save()
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
