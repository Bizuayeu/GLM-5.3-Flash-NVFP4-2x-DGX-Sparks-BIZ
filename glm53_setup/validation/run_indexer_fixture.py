"""Check the capture adapter on a four-layer model, with LPA and MTP absent."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    status = json.loads((args.fixture / "fixture-status.json").read_text())
    config = json.loads((args.fixture / "config.json").read_text())
    if (
        not status.get("all_tensor_bytes_verified")
        or not config.get("_test_fixture_only")
        or config["text_config"]["num_hidden_layers"] != 4
    ):
        raise ValueError("Verified four-layer fixture required")
    args.output.mkdir(parents=True, exist_ok=False)
    from vllm import LLM, SamplingParams

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
        worker_extension_cls="glm53_setup.runtime.indexer_worker.IndexerCaptureWorker",
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
        temperature=0, max_tokens=1, ignore_eos=True, seed=42, detokenize=False
    )
    report = {
        "scope": "single-indexer fixture adapter; not cross-layer overlap or language quality",
        "cases": [],
    }
    for length in (2048, 8192):
        prompt = [{"prompt_token_ids": (base * (length // len(base) + 1))[:length]}]
        case = {"length": length, "outputs": {}}
        for mode in ("off", "capture", "restored", "timing"):
            if mode in {"capture", "timing"}:
                case[mode + "_start"] = llm.collective_rpc(
                    "indexer_capture_start",
                    kwargs={
                        "request_id": f"fixture-{length}-{mode}",
                        "positions": [length - 1] if mode == "capture" else [],
                    },
                )
            try:
                case["outputs"][mode] = list(
                    llm.generate(prompt, params, use_tqdm=False)[0].outputs[0].token_ids
                )
                if mode in {"capture", "timing"}:
                    case[mode] = llm.collective_rpc("indexer_capture_finish")
            finally:
                llm.collective_rpc("indexer_capture_abort")
        case["tokens_equal"] = len({tuple(v) for v in case["outputs"].values()}) == 1
        for worker in case["capture"]:
            if not worker["rows"] or worker["events"] == 0 or not worker["layer_ms"]:
                raise ValueError("No real indexer observations")
        report["cases"].append(case)
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        if not case["tokens_equal"]:
            raise ValueError("Capture/restoration changed fixture output")
        print(length, "capture and restoration passed", flush=True)


if __name__ == "__main__":
    main()
