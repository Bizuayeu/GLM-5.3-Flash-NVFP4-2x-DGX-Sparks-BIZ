"""Name the first module whose output differs when one process repeats a request.

The first pass keeps every hooked module's output on the GPU; later passes compare
against it call by call. A module that differs while everything executed before it
matched received identical inputs, so the difference was made inside it.
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

from ..io import write_json
from ..runtime.moe_token_order import canonical_expert_order as canonical_order
from .run_fixture import read_fixture

# Decoder layers and the parts below them; deeper leaves add calls, not information.
HOOKED = re.compile(
    r"(^|\.)(embed_tokens|norm|lm_head|layers\.\d+"
    r"(\.(self_attn|mlp)(\.(indexer|gate|experts|shared_experts))?)?)$"
)


def first_tensors(output):
    """Floating tensors of a module output, in order; tuples and lists are flattened."""
    import torch

    items = output if isinstance(output, (tuple, list)) else [output]
    return [t for t in items if isinstance(t, torch.Tensor) and t.is_floating_point()]


def summarize(rows):
    """First differing call in execution order, and every module that ever differed."""
    differing = [row for row in rows if row["max_abs"] > 0 or row["shape_changed"]]
    modules = {}
    for row in differing:
        entry = modules.setdefault(row["module"], {"calls": 0, "max_abs": 0.0})
        entry["calls"] += 1
        entry["max_abs"] = max(entry["max_abs"], row["max_abs"])
    return {
        "calls_compared": len(rows),
        "calls_differing": len(differing),
        "first_difference": differing[0] if differing else None,
        "modules_differing": modules,
    }


def block_size_of(args, kwargs):
    """The block size ``moe_align_block_size`` was called with.

    Second positional argument at the pinned call site, ``block_size`` by name
    otherwise. The buffers it returns are worst-case ones whose lengths need not
    divide into each other, so their ratio is not the block size.
    """
    return args[1] if len(args) > 1 else kwargs["block_size"]


def ordered_align(original):
    """``moe_align_block_size`` with the served order fix applied to its result."""

    def ordered(*args, **kwargs):
        result = original(*args, **kwargs)
        block = block_size_of(args, kwargs)
        result[0].copy_(canonical_order(result[0], result[1], result[2], block))
        return result

    return ordered


def watched_align(original, log, canonicalize):
    """``moe_align_block_size`` that fingerprints what it hands the Marlin kernel.

    With ``canonicalize`` the token order inside each expert is made the same on
    every call, by the function the served patch uses.
    """
    import hashlib

    def digest(tensor):
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()[:16]

    def watched(*args, **kwargs):
        result = original(*args, **kwargs)
        block = block_size_of(args, kwargs)
        sorted_ids, expert_ids, padded = result[:3]
        # Both buffers are allocated for the worst case; only this prefix is written.
        valid = int(padded)
        as_a_set = canonical_order(sorted_ids, expert_ids, padded, block)
        if canonicalize:
            # Same blocks, same experts; only the order inside each expert changes.
            result[0].copy_(as_a_set)
        log.append(
            {
                "sorted_token_ids": digest(sorted_ids[:valid]),
                "as_a_set_per_expert": digest(as_a_set[:valid]),
                "expert_ids": digest(expert_ids[: -(-valid // block)]),
                "num_tokens_post_padded": valid,
                "unwritten_tail": digest(sorted_ids[valid:])
                if valid < sorted_ids.numel()
                else None,
            }
        )
        return result

    return watched


class RepeatTraceWorker:
    def repeat_trace_start(self, compare):
        import torch

        if hasattr(self, "repeat_trace"):
            raise ValueError("A repeat trace is already attached")
        state = {"handles": [], "calls": {}, "rows": [], "order": 0}
        reference = getattr(self, "repeat_reference", {}) if compare else {}
        if not compare:
            self.repeat_reference = reference

        def hook(name):
            def after(module, args, output):
                tensors = first_tensors(output)
                if not tensors:
                    return
                call = state["calls"].get(name, 0)
                state["calls"][name] = call + 1
                state["order"] += 1
                if not compare:
                    reference[(name, call)] = [t.detach().clone() for t in tensors]
                    return
                kept = reference.get((name, call))
                row = {
                    "order": state["order"],
                    "module": name,
                    "call": call,
                    "rows": int(tensors[0].shape[0]) if tensors[0].ndim else 1,
                    "max_abs": 0.0,
                    "differing_elements": 0,
                    "shape_changed": kept is None
                    or [t.shape for t in kept] != [t.shape for t in tensors],
                }
                if not row["shape_changed"]:
                    for old, new in zip(kept, tensors):
                        delta = (new.float() - old.float()).abs()
                        row["max_abs"] = max(row["max_abs"], float(delta.max()))
                        row["differing_elements"] += int(torch.count_nonzero(delta))
                state["rows"].append(row)

            return after

        for name, module in self.get_model().named_modules():
            if HOOKED.search(name):
                state["handles"].append(module.register_forward_hook(hook(name)))
        self.repeat_trace = state
        return {"rank": self.rank, "hooked": len(state["handles"])}

    def repeat_trace_zero_moe_buffers(self):
        """Diagnostic: hand the Marlin MoE kernel zeroed scratch buffers on every call.

        If repeats become identical, the kernel reads scratch memory it did not write.
        """
        from vllm.model_executor.layers.fused_moe.experts import marlin_moe

        original = marlin_moe.fused_marlin_moe
        self.repeat_zeroed = counts = {"calls": 0, "buffers": 0}

        def zeroed(*args, **kwargs):
            counts["calls"] += 1
            for key in ("intermediate_cache13", "intermediate_cache2", "output"):
                if kwargs.get(key) is not None:
                    kwargs[key].zero_()
                    counts["buffers"] += 1
            if kwargs.get("workspace") is not None:
                kwargs["workspace"].zero_()
                counts["buffers"] += 1
            return original(*args, **kwargs)

        marlin_moe.fused_marlin_moe = zeroed
        return {"rank": self.rank, "patched": "fused_marlin_moe"}

    def repeat_trace_canonical_only(self):
        """The order fix alone, as a runtime patch would apply it: no log, no sync."""
        from vllm.model_executor.layers.fused_moe.experts import marlin_moe

        marlin_moe.moe_align_block_size = ordered_align(marlin_moe.moe_align_block_size)
        return {"rank": self.rank, "patched": "moe_align_block_size, order only"}

    def repeat_trace_watch_align(self, canonicalize=False):
        """Diagnostic: fingerprint what moe_align_block_size hands the Marlin kernel.

        With ``canonicalize`` the token order inside each expert is made the same on
        every call; if repeats then match, the kernel's result depends on that order.
        """
        from vllm.model_executor.layers.fused_moe.experts import marlin_moe

        self.repeat_align = log = []
        marlin_moe.moe_align_block_size = watched_align(
            marlin_moe.moe_align_block_size, log, canonicalize
        )
        return {"rank": self.rank, "patched": "moe_align_block_size"}

    def repeat_trace_align_log(self):
        log = list(getattr(self, "repeat_align", []))
        getattr(self, "repeat_align", []).clear()
        return {"rank": self.rank, "calls": log}

    def repeat_trace_zeroed_counts(self):
        return {"rank": self.rank, **getattr(self, "repeat_zeroed", {})}

    def repeat_trace_finish(self):
        state = self.__dict__.pop("repeat_trace")
        for handle in state["handles"]:
            handle.remove()
        return {"rank": self.rank, "rows": state["rows"], "calls": state["order"]}

    def repeat_trace_forget(self):
        self.__dict__.pop("repeat_reference", None)
        return {"rank": self.rank}


def parser():
    """The runner's argument interface; the engine settings follow from it."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fixture", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--long-tokens", type=int, default=8192)
    cli.add_argument(
        "--watch-align",
        action="store_true",
        help="diagnostic: compare the expert token ordering between passes",
    )
    cli.add_argument(
        "--verify-canonical",
        action="store_true",
        help="record one plain pass, install the order fix, compare the next passes",
    )
    cli.add_argument(
        "--timing",
        action="store_true",
        help="no hooks: time prefill and decode, with --canonical-align as the variable",
    )
    cli.add_argument(
        "--canonical-align",
        action="store_true",
        help="diagnostic: fix the token order inside each expert before the kernel",
    )
    cli.add_argument(
        "--zero-moe-buffers",
        action="store_true",
        help="diagnostic: zero the Marlin MoE scratch buffers before every call",
    )
    return cli


def engine_kwargs(args):
    """What LLM() is constructed with; these settings define the measurement."""
    return {
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
        "gpu_memory_utilization": 0.2,
        "seed": 42,
        "worker_extension_cls": "glm53_setup.validation.run_repeat_trace.RepeatTraceWorker",
        "kernel_config": {
            "enable_flashinfer_autotune": False,
            "enable_cutedsl_warmup": False,
            "enable_jit_warmup": False,
            "moe_backend": "marlin",
            "linear_backend": "marlin",
        },
    }


def main(argv=None):
    args = parser().parse_args(argv)
    config, _ = read_fixture(args.fixture, layers=None)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "loading",
        "scope": "which module first differs between identical requests in one process",
        "layers": config["text_config"]["num_hidden_layers"],
        "cases": [],
    }

    def save():
        write_json(args.output / "result.json", report)

    save()
    try:
        from vllm import LLM, SamplingParams

        from ..agreement import TEXTS
        from .run_agreement_fixture import cycle_tokens

        llm = LLM(**engine_kwargs(args))
        if args.zero_moe_buffers:
            report["zero_moe_buffers"] = llm.collective_rpc(
                "repeat_trace_zero_moe_buffers"
            )
        if args.timing:
            import statistics
            import time

            if args.canonical_align:
                report["patch"] = llm.collective_rpc("repeat_trace_canonical_only")
            tokenizer = llm.get_tokenizer()
            base = tokenizer.encode(
                next(iter(TEXTS.values())), add_special_tokens=False
            )
            cases = {
                "prefill_2048": ((base * 20)[:2048], 1),
                "decode_128": (base[:64], 128),
            }
            report["timing"] = {}
            for name, (token_ids, new_tokens) in cases.items():
                params = SamplingParams(
                    temperature=0, max_tokens=new_tokens, ignore_eos=True, seed=42
                )
                seconds = []
                for _ in range(7):
                    began = time.perf_counter()
                    llm.generate(
                        [{"prompt_token_ids": token_ids}], params, use_tqdm=False
                    )
                    seconds.append(time.perf_counter() - began)
                report["timing"][name] = {
                    "seconds": [round(v, 4) for v in seconds],
                    "median_after_two_warmups": round(
                        statistics.median(seconds[2:]), 4
                    ),
                }
            report.update(status="complete", canonical_align=args.canonical_align)
            return
        if args.watch_align or args.canonical_align:
            args.watch_align = True
            report["watch_align"] = llm.collective_rpc(
                "repeat_trace_watch_align",
                kwargs={"canonicalize": args.canonical_align},
            )
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(
            temperature=0, max_tokens=1, ignore_eos=True, seed=42, detokenize=False
        )
        prompts = {
            name: tokenizer.encode(text, add_special_tokens=False)
            for name, text in TEXTS.items()
        }
        prompts["long-cycle"] = cycle_tokens(list(prompts.values()), args.long_tokens)
        for name, token_ids in prompts.items():
            llm.collective_rpc("repeat_trace_forget")
            case = {"name": name, "prompt_tokens": len(token_ids), "repeats": []}
            for repeat in range(args.repeats):
                llm.collective_rpc("repeat_trace_start", kwargs={"compare": repeat > 0})
                try:
                    llm.generate(
                        [{"prompt_token_ids": token_ids}], sampling, use_tqdm=False
                    )
                finally:
                    traced = llm.collective_rpc("repeat_trace_finish")[0]
                if args.verify_canonical and not repeat and "patch" not in report:
                    report["patch"] = llm.collective_rpc("repeat_trace_canonical_only")
                if repeat:
                    case["repeats"].append(summarize(traced["rows"]))
                    case["repeats"][-1]["last_module"] = traced["rows"][-1]
                else:
                    case["calls_recorded"] = traced["calls"]
                if args.watch_align:
                    calls = llm.collective_rpc("repeat_trace_align_log")[0]["calls"]
                    if not repeat:
                        first_pass = calls
                    else:
                        case["repeats"][-1]["align"] = {
                            "calls": len(calls),
                            "ordering_differs": sum(
                                a["sorted_token_ids"] != b["sorted_token_ids"]
                                for a, b in zip(first_pass, calls)
                            ),
                            "unwritten_tail_differs": sum(
                                a["unwritten_tail"] != b["unwritten_tail"]
                                for a, b in zip(first_pass, calls)
                            ),
                            "membership_differs": sum(
                                a["as_a_set_per_expert"] != b["as_a_set_per_expert"]
                                or a["expert_ids"] != b["expert_ids"]
                                for a, b in zip(first_pass, calls)
                            ),
                        }
            report["cases"].append(case)
            save()
        llm.collective_rpc("repeat_trace_forget")
        if args.zero_moe_buffers:
            report["zeroed_counts"] = llm.collective_rpc("repeat_trace_zeroed_counts")
        report.update(
            status="complete", finished_at=datetime.now(timezone.utc).isoformat()
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
