"""Prove exact-cache isolation with a deliberately non-teacher projector."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from ..config import REVISION, TEACHER_PRECISION
from ..io import write_json
from ..runtime.apc_runtime import MODE_KEY
from ..runtime.lpa import PROJECTOR_FORMAT
from ..server_config import speculative_config
from .run_fixture import read_fixture

# The synthetic LPA configuration under test: no cut, a 512-token exact tail,
# and approximation only when more than 1024 tokens would be omitted.
TAIL = 512
BREAK_EVEN = 1024
MAX_MODEL_LEN = 32768
OUTPUT_TOKENS = 16


def parser():
    """The runner's argument interface; the engine settings follow from it."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fixture", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--mtp", type=int, choices=(1, 3))
    cli.add_argument("--fused-unpack", action="store_true")
    cli.add_argument("--async-index-checks", action="store_true")
    cli.add_argument("--async-scheduling", action="store_true")
    cli.add_argument(
        "--retention-interval",
        type=int,
        help="Explicit native checkpoint-retention candidate",
    )
    cli.add_argument(
        "--history",
        action="store_true",
        help="Additional edit/branch and shared-state checks",
    )
    return cli


def engine_kwargs(args):
    """What LLM() is constructed with; these settings define the measurement."""
    return {
        "model": str(args.fixture),
        "tensor_parallel_size": 1,
        "language_model_only": True,
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "async_scheduling": args.async_scheduling,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 512,
        "block_size": 256,
        "kv_cache_dtype": "fp8",
        "kv_cache_memory_bytes": 1024**3,
        "gpu_memory_utilization": 0.2,
        "seed": 42,
        "worker_extension_cls": "glm53_setup.validation.apc_fixture_worker.APCFixtureWorker",
        "speculative_config": speculative_config(args.mtp) if args.mtp else None,
        "kernel_config": {
            "moe_backend": "marlin",
            "linear_backend": "marlin",
            "enable_flashinfer_autotune": False,
            "enable_cutedsl_warmup": False,
            "enable_jit_warmup": False,
        },
        **(
            {"prefix_cache_retention_interval": args.retention_interval}
            if args.retention_interval is not None
            else {}
        ),
    }


def lpa_config(projector, digest):
    """The GLM53_APC_LPA_CONFIG the workers read; the projector is the synthetic one."""
    return json.dumps(
        {
            "cut": 0,
            "tail": TAIL,
            "break_even": BREAK_EVEN,
            "projector_path": str(projector),
            "projector_sha256": digest,
            "skip_mla_queries": True,
        }
    )


def write_synthetic_projector(path, width):
    """A format-2 projector with fixed, visibly non-teacher weights for four layers."""
    import torch

    torch.save(
        {
            "format_version": PROJECTOR_FORMAT,
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
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_policy(row, hit, expected):
    """Every worker saw the same hit H, and omitted exactly the expected queries."""
    if row["cached_tokens"] != hit:
        raise ValueError("Unexpected actual cache hit")
    for worker in row["workers"]:
        policy = worker["policy"]["policy"]
        if policy["cached_tokens"] != hit:
            raise ValueError("Worker H differs from cache lookup")
        counts = worker["lpa"]["mla_queries_skipped"] if worker["lpa"] else {}
        if counts != ({3: expected} if expected else {}):
            raise ValueError(("Incorrect actual query omission", counts, expected))


def first_distribution_delta(teacher, altered):
    """Largest logprob gap at the first position, over the tokens both reported."""
    common = teacher["logprobs"][0].keys() & altered["logprobs"][0].keys()
    return max(
        (abs(teacher["logprobs"][0][k] - altered["logprobs"][0][k]) for k in common),
        default=0,
    )


def history_positions(block, length):
    """Edit positions: the first tokens, around the block boundary, and along the prompt."""
    return sorted(
        {
            3,
            4,
            5,
            block - 1,
            block,
            block + 1,
            length // 10,
            length // 2,
            length * 9 // 10,
        }
    )


def expected_omission(prompt_tokens, hit):
    """Queries the policy omits for a prompt with hit H: the eligible span past the break-even."""
    eligible = max(0, prompt_tokens - min(TAIL, prompt_tokens) - hit)
    return eligible if eligible > BREAK_EVEN else 0


def main(argv=None):
    args = parser().parse_args(argv)
    configuration, _ = read_fixture(args.fixture)
    model = configuration["text_config"]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "loading",
        "scope": "cache provenance fixture, not model quality",
        "mtp": args.mtp,
        "fused_unpack": args.fused_unpack,
        "async_index_checks": args.async_index_checks,
        "async_scheduling": args.async_scheduling,
        "requests": [],
    }

    def save():
        write_json(args.output / "result.json", report)

    save()
    try:
        projector = args.output / "synthetic-projector.pt"
        digest = write_synthetic_projector(projector, model["hidden_size"])
        report["synthetic_projector_sha256"] = digest
        os.environ.update(
            VLLM_ENABLE_V1_MULTIPROCESSING="0",
            NVIDIA_TF32_OVERRIDE="0",
            GLM53_FUSED_UNPACK=str(int(args.fused_unpack)),
            GLM53_ASYNC_INDEX_CHECKS=str(int(args.async_index_checks)),
            GLM53_APC_LPA_CONFIG=lpa_config(projector.resolve(), digest),
            GLM53_APC_LPA_AUDIT=str((args.output / "cache-audit.jsonl").resolve()),
        )
        from vllm import LLM, SamplingParams
        from vllm.v1.request import Request

        from .run_fixture import encode_output

        llm = LLM(**engine_kwargs(args))
        client = llm.llm_engine.engine_core
        if type(client).__name__ != "InprocClient":
            raise ValueError("Fixture requires the explicit in-process engine")
        core = client.engine_core
        manager = core.scheduler.kv_cache_manager
        block = manager.coordinator.scheduler_block_size
        # The pinned MTP/EAGLE lookup drops a trailing block to replay draft
        # lookahead. Prime that extra exact block; H remains the returned hit.
        replay_margin = block if manager.coordinator.eagle_group_ids else 0
        prime_length = block + replay_margin + 1
        length = 2 * block + replay_margin + 1024
        if block < 1 or length + OUTPUT_TOKENS > MAX_MODEL_LEN:
            raise ValueError("Fixture must cross two actual shared-cache blocks")
        base = llm.get_tokenizer().encode(
            "The archive contains numbered records. Preserve the code and continue the sequence. ",
            add_special_tokens=False,
        )
        ids = (base * (length // len(base) + 1))[:length]
        report.update(
            status="running",
            block_size=block,
            prompt_tokens=length,
            prime_tokens=prime_length,
            replay_margin=replay_margin,
        )

        def generate(label, prompt, mode):
            params = SamplingParams(
                temperature=0,
                seed=42,
                max_tokens=OUTPUT_TOKENS,
                ignore_eos=True,
                logprobs=10,
                detokenize=False,
                extra_args={MODE_KEY: mode},
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
                or len(row["output"]["token_ids"]) != OUTPUT_TOKENS
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

        def lookup(prompt=None):
            request = Request(
                "fixture-probe",
                ids if prompt is None else prompt,
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

        def cache_hashes(physical):
            return llm.collective_rpc(
                "apc_fixture_cache_hashes", kwargs={"block_ids": physical}
            )

        assert llm.reset_prefix_cache()
        no_hit = generate("cold-approximate", ids, "auto")
        assert_policy(no_hit, 0, length - TAIL)
        if lookup()[0] != 0:
            raise ValueError("A cold approximate request populated shared cache")

        controls = []
        for label in ("control", "trial", "restored"):
            assert llm.reset_prefix_cache()
            generate(label + "-exact-prime", ids[:prime_length], "off")
            hit, physical = lookup()
            if hit != block:
                raise ValueError(
                    (
                        "Exact priming did not restore the full joint boundary",
                        hit,
                        block,
                    )
                )
            before = cache_hashes(physical)
            if label == "trial":
                approximate = generate("partial-hit-approximate", ids, "auto")
                assert_policy(approximate, block, length - TAIL - block)
                after_hit, after_blocks = lookup()
                if after_hit != block:
                    raise ValueError("Approximate suffix entered shared APC")
                after = cache_hashes(after_blocks)
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
        delta = first_distribution_delta(teacher, altered)
        report["synthetic_state_distinguished"] = (
            teacher["token_ids"] != altered["token_ids"] or delta > 1e-4
        )
        report["first_distribution_delta"] = delta
        if args.history:
            report["cache_layout"] = llm.collective_rpc("apc_cache_layout")
            report["history"] = []
            for position in history_positions(block, length):
                for branch in (False, True):
                    assert llm.reset_prefix_cache()
                    generate(f"history-prime-{position}-{branch}", ids, "off")
                    original_hit, original_blocks = lookup()
                    original_hashes = cache_hashes(original_blocks)
                    edited = list(ids)
                    edited[position] = next(
                        token for token in base if token != ids[position]
                    )
                    if branch:
                        edited = edited[: position + 1] + base[:16]
                    hit, _ = lookup(edited)
                    if hit > position:
                        raise ValueError(
                            "Edited prefix reused state after the changed token"
                        )
                    row = generate(
                        f"history-approximate-{position}-{branch}", edited, "auto"
                    )
                    assert_policy(row, hit, expected_omission(len(edited), hit))
                    after_hit, after_blocks = lookup()
                    after_hashes = cache_hashes(after_blocks)
                    if after_hit != original_hit or original_hashes != after_hashes:
                        raise ValueError(
                            "Edited/branched request changed the original shared prefix"
                        )
                    exact = generate(
                        f"history-exact-revisit-{position}-{branch}", ids, "off"
                    )
                    assert_policy(exact, original_hit, 0)
                    report["history"].append(
                        {
                            "edit_token": position,
                            "branch": branch,
                            "original_hit": original_hit,
                            "edited_hit": hit,
                            "shared_bytes_equal": True,
                        }
                    )
                    save()
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
