"""Prove exact-cache isolation with a deliberately non-teacher projector."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from ..config import REVISION, TEACHER_PRECISION


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    configuration = json.loads((args.fixture / "config.json").read_text())
    status = json.loads((args.fixture / "fixture-status.json").read_text())
    model = configuration["text_config"]
    if (
        not configuration.get("_test_fixture_only")
        or model["num_hidden_layers"] != 4
        or status["status"] != "complete"
        or not status["all_tensor_bytes_verified"]
    ):
        raise ValueError("A byte-verified four-layer fixture is required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "loading",
        "scope": "cache provenance fixture, not model quality",
        "requests": [],
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2))

    save()
    try:
        import torch

        width = model["hidden_size"]
        projector = args.output / "synthetic-projector.pt"
        torch.save(
            {
                "format_version": 2,
                "teacher_revision": REVISION,
                "teacher_precision": TEACHER_PRECISION,
                "cut": 0,
                "layers": 4,
                "test_fixture_only": True,
                "weights": {
                    i: {
                        "mean": torch.zeros(width),
                        "down": torch.zeros(width, 1),
                        "up": torch.zeros(1, width),
                        "scale": torch.full((width,), 0.5),
                        "bias": torch.linspace(-0.25, 0.25, width),
                    }
                    for i in range(1, 4)
                },
            },
            projector,
        )
        digest = hashlib.sha256(projector.read_bytes()).hexdigest()
        report["synthetic_projector_sha256"] = digest
        os.environ.update(
            VLLM_ENABLE_V1_MULTIPROCESSING="0",
            NVIDIA_TF32_OVERRIDE="0",
            GLM53_FUSED_UNPACK="0",
            GLM53_ASYNC_INDEX_CHECKS="0",
            GLM53_APC_LPA_CONFIG=json.dumps(
                {
                    "cut": 0,
                    "tail": 512,
                    "break_even": 1024,
                    "projector_path": str(projector.resolve()),
                    "projector_sha256": digest,
                    "skip_mla_queries": True,
                }
            ),
            GLM53_APC_LPA_AUDIT=str((args.output / "cache-audit.jsonl").resolve()),
        )
        from vllm import LLM, SamplingParams
        from vllm.v1.request import Request

        from .run_fixture import encode_output

        llm = LLM(
            model=str(args.fixture),
            tensor_parallel_size=1,
            language_model_only=True,
            enforce_eager=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            async_scheduling=False,
            max_model_len=32768,
            max_num_seqs=1,
            max_num_batched_tokens=512,
            block_size=256,
            kv_cache_dtype="fp8",
            kv_cache_memory_bytes=1024**3,
            gpu_memory_utilization=0.20,
            seed=42,
            worker_extension_cls="glm53_setup.validation.apc_fixture_worker.APCFixtureWorker",
            kernel_config={
                "moe_backend": "marlin",
                "linear_backend": "marlin",
                "enable_flashinfer_autotune": False,
                "enable_cutedsl_warmup": False,
                "enable_jit_warmup": False,
            },
        )
        client = llm.llm_engine.engine_core
        if type(client).__name__ != "InprocClient":
            raise ValueError("Fixture requires the explicit in-process engine")
        core = client.engine_core
        manager = core.scheduler.kv_cache_manager
        block = manager.coordinator.scheduler_block_size
        length = 2 * block + 1024
        if block < 1 or length + 16 > 32768:
            raise ValueError("Fixture must cross two actual shared-cache blocks")
        base = llm.get_tokenizer().encode(
            "The archive contains numbered records. Preserve the code and continue the sequence. ",
            add_special_tokens=False,
        )
        ids = (base * (length // len(base) + 1))[:length]
        report.update(status="running", block_size=block, prompt_tokens=length)

        def generate(label, prompt, mode):
            params = SamplingParams(
                temperature=0,
                seed=42,
                max_tokens=16,
                ignore_eos=True,
                logprobs=10,
                detokenize=False,
                extra_args={"glm53_lpa_mode": mode},
            )
            result = llm.generate(
                [{"prompt_token_ids": prompt}], params, use_tqdm=False
            )[0]
            row = {
                "label": label,
                "cached_tokens": result.num_cached_tokens,
                "output": encode_output(result),
                "workers": llm.collective_rpc("apc_lpa_report"),
            }
            if (
                not result.finished
                or len(row["output"]["token_ids"]) != 16
                or not all(
                    math.isfinite(x)
                    for values in row["output"]["logprobs"]
                    for x in values.values()
                )
            ):
                raise ValueError("Incomplete or nonfinite fixture generation")
            report["requests"].append(row)
            save()
            print(label, "cached", row["cached_tokens"], flush=True)
            return row

        def lookup():
            request = Request(
                "fixture-probe",
                ids,
                SamplingParams(max_tokens=1),
                None,
                block_hasher=core.request_block_hasher,
            )
            blocks, hit, _ = manager.get_computed_blocks(request)
            physical = [
                [None if b.is_null else b.block_id for b in group]
                for group in blocks.blocks
            ]
            return hit, physical

        def assert_policy(row, hit, expected):
            if row["cached_tokens"] != hit:
                raise ValueError("Unexpected actual cache hit")
            for worker in row["workers"]:
                policy = worker["policy"]["policy"]
                if policy["cached_tokens"] != hit:
                    raise ValueError("Worker H differs from cache lookup")
                counts = worker["lpa"]["mla_queries_skipped"] if worker["lpa"] else {}
                if counts != ({3: expected} if expected else {}):
                    raise ValueError(
                        ("Incorrect actual query omission", counts, expected)
                    )

        assert llm.reset_prefix_cache()
        no_hit = generate("cold-approximate", ids, "auto")
        assert_policy(no_hit, 0, length - 512)
        if lookup()[0] != 0:
            raise ValueError("A cold approximate request populated shared cache")

        controls = []
        for label in ("control", "trial", "restored"):
            assert llm.reset_prefix_cache()
            generate(label + "-exact-prime", ids[: block + 1], "off")
            hit, physical = lookup()
            if hit != block:
                raise ValueError(
                    "Exact priming did not restore the full joint boundary"
                )
            before = llm.collective_rpc(
                "apc_fixture_cache_hashes", kwargs={"block_ids": physical}
            )
            if label == "trial":
                approximate = generate("partial-hit-approximate", ids, "auto")
                assert_policy(approximate, block, length - 512 - block)
                after_hit, after_blocks = lookup()
                if after_hit != block:
                    raise ValueError("Approximate suffix entered shared APC")
                after = llm.collective_rpc(
                    "apc_fixture_cache_hashes", kwargs={"block_ids": after_blocks}
                )
                report["shared_prefix_hashes"] = {"before": before, "after": after}
                if before != after:
                    raise ValueError("Approximation changed shared exact-prefix bytes")
            exact = generate(label + "-exact-after-prefix", ids, "off")
            assert_policy(exact, block, 0)
            if lookup()[0] != 2 * block:
                raise ValueError(
                    "Ordinary recomputation did not grow exact shared cache"
                )
            controls.append(exact)
        report["exact_control_tokens_equal"] = (
            len({tuple(x["output"]["token_ids"]) for x in controls}) == 1
        )
        teacher = controls[1]["output"]
        altered = approximate["output"]
        common = teacher["logprobs"][0].keys() & altered["logprobs"][0].keys()
        delta = max(
            (
                abs(teacher["logprobs"][0][k] - altered["logprobs"][0][k])
                for k in common
            ),
            default=0,
        )
        report["synthetic_state_distinguished"] = (
            teacher["token_ids"] != altered["token_ids"] or delta > 1e-4
        )
        report["first_distribution_delta"] = delta
        if not report["synthetic_state_distinguished"]:
            raise ValueError("Synthetic projector failed to distinguish approximation")
        if not report["exact_control_tokens_equal"]:
            raise ValueError(
                "Exact post-approximation tokens differ; inspect retained numerical controls"
            )
        report.update(status="complete", passed=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
