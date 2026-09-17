"""Read a four-layer fixture teacher-forced: top-k rows, full-vocabulary
log-probabilities and sparse-MLA candidate sets, for a later A/B against
another checkpoint of the same fixture."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..io import write_json
from ..runtime.indexer_worker import IndexerCaptureWorker

LONG_TOKENS = 8192
LONG_KEEP_LAST = 1024
CANDIDATE_STRIDE = 64


def candidate_positions(length, first=2304, stride=CANDIDATE_STRIDE):
    """Query positions past the 2,048-candidate limit, where selection is not trivial."""
    return sorted(set(range(first, length, stride)) | {length - 1})


def cycle_tokens(token_lists, length):
    """One long prompt from the short texts in order, repeated to ``length``."""
    joined = [token for tokens in token_lists for token in tokens]
    if not joined:
        raise ValueError("No tokens to repeat")
    return (joined * (length // len(joined) + 1))[:length]


class AgreementWorker(IndexerCaptureWorker):
    """Keeps a float32 log-softmax of the prompt logits the runner scores anyway."""

    def logprob_capture_start(self, prompt_tokens, keep_last):
        # The V2 model runner scores prompt rows through this module-level name;
        # sampled-token log-probabilities use their own import and are not seen.
        import torch
        from vllm.v1.worker.gpu.sample import prompt_logprob

        if hasattr(self, "logprob_capture"):
            raise ValueError("A log-probability capture is already attached")
        state = {
            "original": prompt_logprob.compute_topk_scores,
            "first_kept": max(0, prompt_tokens - 1 - keep_last),
            "seen": 0,
            "rows": [],
            "targets": [],
        }

        def compute(logits, num_logprobs, token_ids, **kwargs):
            start, count = state["seen"], logits.shape[0]
            state["seen"] += count
            skip = max(0, state["first_kept"] - start)
            if skip < count:
                rows = logits[skip:].log_softmax(dim=-1, dtype=torch.float32)
                state["rows"].append(rows.cpu())
                state["targets"].append(token_ids[skip:].detach().cpu())
            return state["original"](logits, num_logprobs, token_ids, **kwargs)

        prompt_logprob.compute_topk_scores = compute
        self.logprob_capture = state
        return {"rank": self.rank, "first_kept": state["first_kept"]}

    def logprob_capture_finish(self, prompt_token_ids, path=None):
        import numpy as np
        import torch

        state = self.logprob_capture
        self.logprob_capture_abort()
        # Every prompt token is scored; the last row predicts past the prompt.
        if state["seen"] != len(prompt_token_ids):
            raise ValueError("Captured rows do not cover the prompt")
        rows = torch.cat(state["rows"])[:-1].contiguous()
        expected = prompt_token_ids[1 + state["first_kept"] :]
        if torch.cat(state["targets"])[:-1].tolist() != expected:
            raise ValueError("Captured rows are not aligned with the prompt tokens")
        report = {
            "rank": self.rank,
            "rows": rows.shape[0],
            "vocab": rows.shape[1],
            "first_kept": state["first_kept"],
        }
        previous = getattr(self, "logprob_previous", None)
        if previous is not None and previous.shape == rows.shape:
            kl = (previous.exp() * (previous - rows)).sum(-1)
            report["repeat"] = {
                "kl_mean": float(kl.mean()),
                "kl_max": float(kl.max()),
                "argmax_agreement": float(
                    (previous.argmax(-1) == rows.argmax(-1)).float().mean()
                ),
                "bit_identical": bool(torch.equal(previous, rows)),
            }
        self.logprob_previous = rows
        if path is not None:
            np.save(path, rows.numpy())
        return report

    def logprob_capture_forget(self):
        self.logprob_previous = None
        return {"rank": self.rank}

    def logprob_capture_abort(self):
        state = self.__dict__.pop("logprob_capture", None)
        if state is not None:
            from vllm.v1.worker.gpu.sample import prompt_logprob

            prompt_logprob.compute_topk_scores = state["original"]
        return {"rank": self.rank, "detached": True}

    def inspect_fixture(self):
        from .run_fixture import inspect_model

        return inspect_model(self.get_model())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
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
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "loading",
        "scope": "four-layer component reading; not language quality of the full model",
        "full_model_inference_validated": False,
        "fixture_status": status,
        "quantization_producer": (config.get("quantization_config") or {}).get(
            "producer"
        ),
    }

    def save():
        write_json(args.output / "result.json", report)

    save()
    from vllm import LLM, SamplingParams

    from ..agreement import TEXTS, TOP_K, agreement, drift, score_text

    kwargs = {
        "model": str(args.fixture),
        "tensor_parallel_size": 1,
        "language_model_only": True,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_model_len": 16384,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 512,
        "block_size": 256,
        "kv_cache_dtype": "fp8",
        "kv_cache_memory_bytes": 512 * 1024**2,
        "gpu_memory_utilization": 0.20,
        "seed": 42,
        "worker_extension_cls": (
            "glm53_setup.validation.run_agreement_fixture.AgreementWorker"
        ),
        "kernel_config": {
            "enable_flashinfer_autotune": False,
            "enable_cutedsl_warmup": False,
            "enable_jit_warmup": False,
            "moe_backend": "marlin",
            "linear_backend": "marlin",
        },
    }
    report["engine_args"] = kwargs
    try:
        started = time.monotonic()
        llm = LLM(**kwargs)
        report["load_seconds"] = time.monotonic() - started
        report["model_inspection"] = llm.collective_rpc("inspect_fixture")
        report["status"] = "reading"
        save()
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            prompt_logprobs=TOP_K,
            seed=42,
            detokenize=False,
        )
        prompts = {
            name: tokenizer.encode(text, add_special_tokens=False)
            for name, text in TEXTS.items()
        }
        prompts["long-cycle"] = cycle_tokens(list(prompts.values()), LONG_TOKENS)
        report["texts"], report["self_agreement"], report["candidates"] = [], [], []
        for name, token_ids in prompts.items():
            long = name == "long-cycle"
            llm.collective_rpc("logprob_capture_forget")
            scored = []
            for repeat in range(args.repeats):
                if long:
                    llm.collective_rpc(
                        "indexer_capture_start",
                        kwargs={
                            "request_id": name,
                            "positions": candidate_positions(len(token_ids)),
                            "max_bytes": 64 * 1024**2,
                        },
                    )
                llm.collective_rpc(
                    "logprob_capture_start",
                    kwargs={
                        "prompt_tokens": len(token_ids),
                        "keep_last": LONG_KEEP_LAST if long else len(token_ids),
                    },
                )
                try:
                    output = llm.generate(
                        [{"prompt_token_ids": token_ids}], sampling, use_tqdm=False
                    )[0]
                    captured = llm.collective_rpc(
                        "logprob_capture_finish",
                        kwargs={
                            "prompt_token_ids": token_ids,
                            "path": str(args.output / f"logprobs-{name}.npy")
                            if repeat == 0
                            else None,
                        },
                    )[0]
                    if long:
                        rows = llm.collective_rpc("indexer_capture_finish")[0]["rows"]
                        report["candidates"].append({"repeat": repeat, "rows": rows})
                finally:
                    llm.collective_rpc("logprob_capture_abort")
                    llm.collective_rpc("indexer_capture_abort")
                response = {
                    "choices": [
                        {
                            "prompt_logprobs": [
                                entry
                                and {
                                    str(token): {"logprob": value.logprob}
                                    for token, value in entry.items()
                                }
                                for entry in output.prompt_logprobs
                            ]
                        }
                    ]
                }
                row = score_text(name, token_ids, response, TOP_K)
                row.update(repeat=repeat, full_vocabulary=captured)
                scored.append(row)
                save()
            report["texts"].extend(scored)
            if len(scored) >= 2:
                check = {"name": name}
                check.update(agreement(scored[0]["rows"], scored[1]["rows"], TOP_K))
                check.update(drift(scored[0]["logprobs"], scored[1]["logprobs"]))
                check["full_vocabulary"] = scored[1]["full_vocabulary"].get("repeat")
                report["self_agreement"].append(check)
        # Instrument health only: repeats inside one process must pick the same argmax.
        report.update(
            status="complete",
            passed=bool(report["self_agreement"])
            and all(c["argmax_agreement"] == 1.0 for c in report["self_agreement"]),
        )
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
