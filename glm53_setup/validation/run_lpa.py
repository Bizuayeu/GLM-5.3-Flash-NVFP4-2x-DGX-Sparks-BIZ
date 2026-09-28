"""Exercise teacher attention-input replay on a verified four-layer fixture."""

import argparse
import math
import time
from pathlib import Path

from ..io import write_json
from ..server_config import speculative_config
from .run_fixture import read_fixture


def parser():
    """The runner's argument interface; the engine settings follow from it."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fixture", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1, 3, 4, 5, 127, 128, 129, 511, 512, 513],
    )
    cli.add_argument("--cut", type=int, default=2)
    cli.add_argument("--skip-mla-queries", action="store_true")
    cli.add_argument(
        "--mtp", type=int, choices=[1, 2, 3], help="Opt-in MTP coexistence check"
    )
    return cli


def engine_kwargs(args, compilation_mode):
    """What LLM() is constructed with; these settings define the measurement.

    ``compilation_mode`` is passed in rather than imported, so these settings stay readable without the GPU stack.
    """
    return {
        "model": str(args.fixture),
        "tensor_parallel_size": 1,
        "language_model_only": True,
        "enforce_eager": True,
        "compilation_config": {
            "mode": compilation_mode,
            "cudagraph_mode": "NONE",
            "cudagraph_capture_sizes": [
                n for n in (1, 2, 4) if n <= (args.mtp or 0) + 1
            ],
        },
        "profiler_config": None,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_model_len": 16384,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 512,
        "block_size": 256,
        "kv_cache_dtype": "fp8",
        "kv_cache_memory_bytes": 512 * 1024**2,
        "gpu_memory_utilization": 0.2,
        "seed": 42,
        "worker_extension_cls": "glm53_setup.runtime.lpa.LPAWorkerExtension",
        "speculative_config": speculative_config(args.mtp) if args.mtp else None,
        "kernel_config": {
            "enable_flashinfer_autotune": False,
            "enable_cutedsl_warmup": False,
            "enable_jit_warmup": False,
            "moe_backend": "marlin",
            "linear_backend": "marlin",
        },
    }


def configure_kwargs(mode, args, length):
    """What the worker's lpa_configure receives for one replay mode."""
    return {
        "mode": "oracle" if mode == "oracle_full_mlp" else mode,
        "cut": args.cut,
        "prompt_length": length,
        "tail": 1,
        "profile": mode == "off",
        "verify_state": True,
        "skip_mlp": mode != "oracle_full_mlp",
        "skip_mla_queries": args.skip_mla_queries,
        "allow_mtp": bool(args.mtp),
    }


def assess_case(modes):
    """One length's verdict: the oracle replays reproduce the exact run's tokens
    and captured state, and every logprob and state error is finite."""
    reference = modes["off"]
    baseline = modes["capture"]
    oracle = modes["oracle"]
    restored = modes["restored"]
    exact = modes["oracle_full_mlp"]
    verdict = {
        "baseline_token_equal": (
            reference["token_ids"] == baseline["token_ids"] == restored["token_ids"]
        ),
        "oracle_token_equal": reference["token_ids"] == oracle["token_ids"],
        "max_shared_logprob_error": max(
            (
                abs(row[token] - oracle["logprobs"][i][token])
                for i, row in enumerate(reference["logprobs"])
                for token in row
                if token in oracle["logprobs"][i]
            ),
            default=None,
        ),
        "oracle_full_mlp_equal": reference["token_ids"] == exact["token_ids"],
        "oracle_full_mlp_logprob_error": max(
            abs(row[token] - exact["logprobs"][i][token])
            for i, row in enumerate(reference["logprobs"])
            for token in row
            if token in exact["logprobs"][i]
        ),
        "finite": all(
            math.isfinite(v)
            for mode in modes.values()
            for row in mode["logprobs"]
            for v in row.values()
        ),
        "state_finite": all(
            error["finite"]
            for mode in modes.values()
            for worker in mode["workers"]
            for error in worker["state_errors"]
        ),
        "oracle_active_state_equal": all(
            error["max_abs"] == 0
            for mode in ("oracle_full_mlp", "oracle")
            for worker in modes[mode]["workers"]
            for error in worker["state_errors"]
        ),
        "state_comparisons": sum(
            len(worker["state_errors"])
            for mode in ("oracle_full_mlp", "oracle")
            for worker in modes[mode]["workers"]
        ),
    }
    verdict["passed"] = (
        verdict["baseline_token_equal"]
        and verdict["oracle_token_equal"]
        and verdict["finite"]
        and verdict["oracle_full_mlp_equal"]
        and verdict["state_finite"]
        and verdict["oracle_active_state_equal"]
        and verdict["state_comparisons"] > 0
    )
    return verdict


def main(argv=None):
    args = parser().parse_args(argv)
    read_fixture(args.fixture)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "loading",
        "scope": "fixture-oracle-replay",
        "mtp": args.mtp,
        "decode_graphs": False,
        "cases": [],
    }

    def save():
        write_json(args.output / "result.json", report)

    save()
    try:
        from vllm import LLM, SamplingParams
        from vllm.config.compilation import CompilationMode

        from .run_fixture import encode_output

        llm = LLM(**engine_kwargs(args, CompilationMode.NONE))
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
                    "lpa_configure", kwargs=configure_kwargs(mode, args, length)
                )
                start = time.monotonic()
                result = llm.generate(
                    [{"prompt_token_ids": ids}], params, use_tqdm=False
                )[0]
                row = encode_output(result)
                row["seconds"] = time.monotonic() - start
                row["workers"] = llm.collective_rpc("lpa_report")
                case["modes"][key] = row
                save()
            case.update(assess_case(case["modes"]))
            save()
        report["passed"] = all(case["passed"] for case in report["cases"])
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
