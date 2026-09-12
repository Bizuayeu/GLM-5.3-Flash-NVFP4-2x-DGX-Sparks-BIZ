"""Four-layer eager/graph comparison with unchanged FP32 attention arithmetic."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--fused-unpack", action="store_true")
    parser.add_argument("--async-index-checks", action="store_true")
    parser.add_argument("--mtp", type=int, choices=[1, 3])
    args = parser.parse_args(argv)
    config = json.loads((args.fixture / "config.json").read_text())
    status = json.loads((args.fixture / "fixture-status.json").read_text())
    if (
        not config.get("_test_fixture_only")
        or config["text_config"]["num_hidden_layers"] != 4
        or not status.get("all_tensor_bytes_verified")
    ):
        raise ValueError("Verified four-layer fixture required")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        GLM53_FUSED_UNPACK="1" if args.fused_unpack else "0",
        GLM53_ASYNC_INDEX_CHECKS="1" if args.graphs or args.async_index_checks else "0",
        NVIDIA_TF32_OVERRIDE="0",
    )
    import glm53_reference
    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationMode

    from glm53_setup.runtime import reference_attention
    from glm53_setup.validation.profile_trace import read_trace
    from glm53_setup.validation.run_fixture import encode_output

    deployed = Path(glm53_reference.__file__).read_bytes()
    if deployed != Path(reference_attention.__file__).read_bytes():
        raise ValueError("Deployed glm53_reference differs from the component source")
    if (
        "GLM53_ASYNC_INDEX_CHECKS"
        not in glm53_reference.sparse_nope_reference.__code__.co_consts
    ):
        raise ValueError("Loaded attention module lacks graph-safe checks")

    report = {
        "status": "loading",
        "scope": "four-layer fixture, no LPA; MTP/fusion recorded separately",
        "graphs_requested": args.graphs,
        "fused_unpack": args.fused_unpack,
        "async_index_checks": args.graphs or args.async_index_checks,
        "mtp": args.mtp,
        "cases": [],
        "attention_source_sha256": hashlib.sha256(deployed).hexdigest(),
    }

    def save():
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    save()
    try:
        llm = LLM(
            model=str(args.fixture),
            tensor_parallel_size=1,
            language_model_only=True,
            enforce_eager=not args.graphs,
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
            compilation_config={
                "mode": CompilationMode.NONE,
                "cudagraph_mode": "FULL_DECODE_ONLY" if args.graphs else "NONE",
                "cudagraph_capture_sizes": [
                    n for n in (1, 2, 4) if n <= (args.mtp or 0) + 1
                ],
            },
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": args.mtp,
                "moe_backend": "triton",
            }
            if args.mtp
            else None,
            profiler_config={
                "profiler": "torch",
                "torch_profiler_dir": str(args.output / "profiles"),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": False,
                "torch_profiler_with_memory": False,
                "torch_profiler_use_gzip": True,
                "ignore_frontend": True,
                "torch_profiler_dump_cuda_time_total": False,
            },
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
            max_tokens=32,
            ignore_eos=True,
            logprobs=10,
            seed=42,
            detokenize=False,
        )
        report["status"] = "running"
        save()
        for length in (64, 2048, 8192):
            ids = (base * (length // len(base) + 1))[:length]
            prompt = [{"prompt_token_ids": ids}]
            llm.generate(prompt, params, use_tqdm=False)
            case = {"input_tokens": length, "samples": []}
            report["cases"].append(case)
            for _ in range(3):
                began = time.perf_counter()
                result = llm.generate(prompt, params, use_tqdm=False)[0]
                row = {"seconds": time.perf_counter() - began, **encode_output(result)}
                if len(row["token_ids"]) != 32 or not all(
                    math.isfinite(v)
                    for values in row["logprobs"]
                    for v in values.values()
                ):
                    raise ValueError("Incomplete or nonfinite fixture output")
                case["samples"].append(row)
                save()
            print(length, "generated", flush=True)
        prompt = [{"prompt_token_ids": (base * 5)[:64]}]
        llm.start_profile()
        try:
            llm.generate(prompt, params, use_tqdm=False)
        finally:
            llm.stop_profile()
        traces = list((args.output / "profiles").rglob("*.gz"))
        if not traces:
            raise ValueError("No profile evidence")
        report["graph_launches_observed"] = sum(
            1
            for path in traces
            for event in read_trace(path).get("traceEvents", [])
            if event.get("ph") == "X"
            and event.get("cat") in {"cuda_runtime", "cuda_driver"}
            and "graphlaunch" in event.get("name", "").lower()
        )
        if args.graphs and not report["graph_launches_observed"]:
            raise ValueError("Graph replay was not observed")
        report["status"] = "complete"
        save()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
