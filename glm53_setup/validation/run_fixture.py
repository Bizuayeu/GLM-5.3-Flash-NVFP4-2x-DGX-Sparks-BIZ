"""Run a guarded single-GPU GLM fixture and record state-consistency checks."""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .summarize_fixture import assess_outputs


def inspect_model(model):
    import torch

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    modules = []
    dtypes = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtypes[key] = dtypes.get(key, 0) + parameter.numel() * parameter.element_size()
    for name, module in model.named_modules():
        quant = getattr(module, "quant_method", None)
        impl = getattr(module, "impl", None)
        if quant is not None or impl is not None:
            row = {"name": name, "module": type(module).__name__}
            if quant is not None:
                row["quant_method"] = type(quant).__name__
                row["quant_internals"] = {
                    k: v.value
                    if isinstance(v, Enum)
                    else v.__name__
                    if isinstance(v, type)
                    else v
                    if isinstance(v, (bool, int, float, str, type(None)))
                    else str(v)
                    if isinstance(v, torch.dtype)
                    else type(v).__name__
                    for k, v in vars(quant).items()
                    if not k.startswith("_") and not k.endswith("config")
                }
            if impl is not None:
                row.update(
                    backend=type(impl).__name__,
                    reference_nope=getattr(impl, "_glm53_reference_nope", None),
                )
            modules.append(row)
    return {
        "parameters_bytes_by_dtype": dtypes,
        "modules": modules,
        "fp32_precision": torch.backends.cuda.matmul.fp32_precision,
    }


class FixtureWorkerExtension:
    """Named local RPC avoids transferring executable callables via pickle."""

    def inspect_fixture(self):
        return inspect_model(self.get_model())


def encode_output(output):
    generated = output.outputs[0]
    return {
        "prompt_token_ids": list(output.prompt_token_ids),
        "token_ids": list(generated.token_ids),
        "finish_reason": generated.finish_reason,
        "logprobs": [
            {str(token): float(value.logprob) for token, value in position.items()}
            for position in (generated.logprobs or [])
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, choices=[128, 512], default=512)
    parser.add_argument("--context", type=int, choices=[2048, 16384], default=2048)
    parser.add_argument("--backend", choices=["auto", "marlin"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads((args.fixture / "config.json").read_text())
    status = json.loads((args.fixture / "fixture-status.json").read_text())
    if (
        not config.get("_test_fixture_only")
        or config["text_config"]["num_hidden_layers"] != 4
        or status["status"] != "complete"
        or status.get("all_tensor_bytes_verified") is not True
    ):
        raise ValueError("Only a byte-verified four-layer test fixture is allowed")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "loading",
        "full_model_inference_validated": False,
        "tp_size": 1,
    }

    def save():
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    save()
    from vllm import LLM, SamplingParams

    kwargs = {
        "model": str(args.fixture),
        "tensor_parallel_size": 1,
        "language_model_only": True,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_model_len": args.context,
        "max_num_seqs": 2,
        "max_num_batched_tokens": args.chunk,
        "block_size": 256,
        "kv_cache_dtype": "fp8",
        "kv_cache_memory_bytes": 512 * 1024**2,
        "gpu_memory_utilization": 0.20,
        "seed": 42,
        "worker_extension_cls": "glm53_setup.validation.run_fixture.FixtureWorkerExtension",
        "kernel_config": {
            "enable_flashinfer_autotune": False,
            "enable_cutedsl_warmup": False,
            "enable_jit_warmup": False,
            "moe_backend": args.backend,
            "linear_backend": args.backend,
        },
    }
    report["engine_args"] = kwargs
    save()
    try:
        started = time.monotonic()
        llm = LLM(**kwargs)
        report["load_seconds"] = time.monotonic() - started
        report["model_inspection"] = llm.collective_rpc("inspect_fixture")
        report["status"] = "generating"
        save()
        tokenizer = llm.get_tokenizer()
        prompts = [
            tokenizer.encode(
                "Tokyo is the capital of Japan. Repeat the last word.",
                add_special_tokens=False,
            ),
            tokenizer.encode(
                "The sequence is 2, 4, 6, 8. Continue the sequence.",
                add_special_tokens=False,
            ),
        ]
        sampling = SamplingParams(
            temperature=0,
            max_tokens=16,
            ignore_eos=True,
            logprobs=5,
            seed=42,
            detokenize=False,
        )

        outputs = {}

        def generate(token_lists, label):
            start = time.monotonic()
            result = [
                encode_output(o)
                for o in llm.generate(
                    [{"prompt_token_ids": ids} for ids in token_lists],
                    sampling,
                    use_tqdm=False,
                )
            ]
            (args.output / (label + ".json")).write_text(json.dumps(result, indent=2))
            outputs[label] = result
            report.setdefault("timings", {})[label] = time.monotonic() - start
            save()
            return result

        first = generate([prompts[0]], "a-first")[0]
        if args.smoke:
            report.update(
                status="complete",
                passed=len(first["token_ids"]) == len(first["logprobs"]) == 16
                and all(
                    math.isfinite(x) for row in first["logprobs"] for x in row.values()
                ),
                scope="load-and-short-generate",
            )
            save()
            if not report["passed"]:
                raise RuntimeError(
                    "Smoke generation is incomplete or contains non-finite logprobs"
                )
            return
        generate([prompts[1]], "b")
        generate([prompts[0]], "a-after-b")
        generate(prompts, "a-b-batch")
        # Force already-generated prefixes through fresh prefill, then compare
        # the selected next token/logprob with the earlier incremental decode.
        for position in [1, 7, 15]:
            generate([prompts[0] + first["token_ids"][:position]], f"forced-{position}")
        long_length = 1300 if args.context == 2048 else 8705
        report["long_prompt_tokens"] = long_length
        long_prompt = (prompts[0] * (long_length // len(prompts[0]) + 1))[:long_length]
        long_result = generate([long_prompt], "long")[0]
        if args.context > 2048:
            generate([long_prompt + long_result["token_ids"][:1]], "long-forced")
        report.update(
            status="complete",
            **assess_outputs(outputs),
            scope="single-node-four-layer-state-consistency",
        )
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    if not report.get("passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
