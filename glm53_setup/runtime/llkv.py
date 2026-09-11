"""Opt-in experimental attention-input replay; never changes checkpoint weights.

The vLLM worker extension is for a serial, eager, text-only experiment. It
retains each attention/state update and skips only unneeded historical MLP
rows. No runtime patch or GPU dependency is activated by importing this file.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    mode: str
    cut: int
    prompt_length: int
    tail: int = 1

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in {
            "off",
            "capture",
            "oracle",
            "identity",
            "predict",
        }:
            raise ValueError("Unsupported experiment mode")
        if any(
            type(value) is not int
            for value in (self.cut, self.prompt_length, self.tail)
        ):
            raise ValueError("Cut, prompt length and tail must be integers")
        if self.cut < 0 or self.prompt_length < 1:
            raise ValueError("Invalid cut or prompt length")
        if not 1 <= self.tail <= self.prompt_length:
            raise ValueError("The exact tail must include the final prompt token")

    def approximate_count(self, positions):
        if any(type(p) is not int or p < 0 for p in positions) or any(
            b != a + 1 for a, b in zip(positions, positions[1:])
        ):
            raise ValueError("Only one contiguous, unpadded sequence is supported")
        return sum(p < self.prompt_length - self.tail for p in positions)


class AttentionInputExperiment:
    def __init__(self, model):
        import torch

        self.torch = torch
        self.layers = sorted(
            [m for m in model.modules() if type(m).__name__ == "Glm5NextDecoderLayer"],
            key=lambda m: m.layer_idx,
        )
        if not self.layers or [m.layer_idx for m in self.layers] != list(
            range(len(self.layers))
        ):
            raise ValueError("Expected all contiguous GLM language layers")
        if any(m.is_mtp_layer or m.is_sequence_parallel for m in self.layers):
            raise ValueError("MTP and sequence-parallel MoE are not supported")
        self.spec = None
        self.capture = {}
        self.oracle = {}
        self.predictor = None
        self.predictor_identity = None
        self.reference_mask = None
        self.skip_mla_queries = False
        self.current_positions = []
        self.source = None
        self.counts = {}
        self.events = []
        self.operation_events = []
        self.profile = False
        self.skip_mlp = True
        self.verify_state = False
        self.state_reference = {}
        self.state_errors = []
        self.handles = []
        for layer in self.layers:
            # cc-defer: Historical mHC and KDA output work still run; reduce them
            # only if profiling shows a material limit after MLA query suppression.
            index = layer.layer_idx
            self.handles.append(
                layer.self_attn.register_forward_pre_hook(
                    self._attention_hook(index), with_kwargs=True
                )
            )
            original = layer.mlp.forward
            if layer.layer_kind == "kda":
                self.handles.append(
                    layer.self_attn.register_forward_hook(self._state_hook(index))
                )
            layer.mlp.forward = self._profiled_forward(
                f"{index}:mlp", self._mlp_forward(index, original)
            )
            layer.self_attn.forward = self._profiled_forward(
                f"{index}:attention",
                self._attention_forward(
                    index, layer.layer_kind, layer.self_attn.forward
                ),
            )
            self.handles.append(
                layer.register_forward_pre_hook(self._layer_start(index))
            )
            self.handles.append(layer.register_forward_hook(self._layer_end(index)))

    def _layer_start(self, index):
        def hook(module, args):
            if self.profile:
                event = self.torch.cuda.Event(enable_timing=True)
                event.record()
                self.events.append([index, event, None])

        return hook

    def _attention_forward(self, index, kind, original):
        def forward(*args, **kwargs):
            spec = self.spec
            if (
                not self.skip_mla_queries
                or not self.skip_mlp
                or kind != "mla"
                or spec is None
                or index < spec.cut
                or spec.mode not in {"oracle", "identity", "predict"}
            ):
                return original(*args, **kwargs)
            count = spec.approximate_count(self.current_positions)
            if not count:
                return original(*args, **kwargs)
            before = self.reference_mask.counts.get(index, 0)
            with self.reference_mask.activate(
                count, len(self.current_positions), index
            ):
                result = original(*args, **kwargs)
            if self.reference_mask.counts.get(index, 0) - before != count:
                raise ValueError(
                    "Pinned reference attention was not invoked as expected"
                )
            return result

        return forward

    def _profiled_forward(self, label, original):
        def forward(*args, **kwargs):
            if not self.profile:
                return original(*args, **kwargs)
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                self.operation_events.append((label, start, end))

        return forward

    def _layer_end(self, index):
        def hook(module, args, output):
            if self.profile:
                event = self.torch.cuda.Event(enable_timing=True)
                event.record()
                self.events[-1][2] = event

        return hook

    def configure(
        self,
        mode,
        cut,
        prompt_length,
        tail=1,
        profile=False,
        predictor_path=None,
        verify_state=False,
        skip_mlp=True,
        skip_mla_queries=False,
    ):
        spec = ExperimentSpec(mode, cut, prompt_length, tail)
        if cut >= len(self.layers):
            raise ValueError("Cut must precede the final layer")
        if type(skip_mla_queries) is not bool:
            raise ValueError("skip_mla_queries must be boolean")
        if skip_mla_queries and self.reference_mask is None:
            import glm53_reference

            from .llkv_query import ReferenceQueryMask

            if isinstance(glm53_reference.sparse_nope_reference, ReferenceQueryMask):
                raise ValueError("Reference attention already has an experiment owner")
            self.reference_mask = ReferenceQueryMask(
                glm53_reference.sparse_nope_reference
            )
            glm53_reference.sparse_nope_reference = self.reference_mask
        self.skip_mla_queries = skip_mla_queries
        if self.reference_mask is not None:
            self.reference_mask.counts = {}
        requested_mode = mode
        if mode in {"predict", "identity"} and tail == prompt_length:
            mode = "off"
            spec = ExperimentSpec(mode, cut, prompt_length, tail)
        if mode == "oracle":
            self.oracle = {
                i: self.torch.cat(chunks, dim=0) for i, chunks in self.capture.items()
            }
            if set(self.oracle) != set(range(cut, len(self.layers))) or any(
                tensor.shape[0] != prompt_length for tensor in self.oracle.values()
            ):
                raise ValueError("Oracle must contain this entire prompt and suffix")
        if mode == "capture":
            self.capture = {}
            self.oracle = {}
            self.state_reference = {}
        if mode == "predict":
            if not predictor_path:
                raise ValueError("Predictor artifact is required")
            path = Path(predictor_path).resolve()
            stat = path.stat()
            identity = (
                str(path),
                stat.st_mtime_ns,
                stat.st_size,
                cut,
                len(self.layers),
            )
            if identity != self.predictor_identity:
                self.predictor = self._load_predictor(path, cut)
                self.predictor_identity = identity
        self.spec = spec
        self.profile = profile
        self.verify_state = verify_state
        self.skip_mlp = skip_mlp
        self.state_errors = []
        self.source = None
        self.current_positions = []
        self.counts = {"attention_tokens": {}, "mlp_skipped_tokens": {}}
        self.events = []
        self.operation_events = []
        return {
            "mode": mode,
            "requested_mode": requested_mode,
            "layers": len(self.layers),
            "cut": cut,
            "prompt_length": prompt_length,
            "tail": tail,
            "skip_mla_queries": skip_mla_queries,
        }

    def _load_predictor(self, predictor_path, cut):
        artifact = self.torch.load(
            predictor_path, map_location="cpu", weights_only=True
        )
        version = artifact.get("format_version", 1)
        if version not in (1, 2):
            raise ValueError("Unsupported projector artifact version")
        if artifact["cut"] != cut or artifact["layers"] != len(self.layers):
            raise ValueError("Predictor does not match the layer boundary")
        if set(artifact["weights"]) != set(range(cut + 1, len(self.layers))):
            raise ValueError("Projector layers are incomplete or unexpected")
        width = self.layers[cut].hidden_size
        for weights in artifact["weights"].values():
            expected_keys = {"mean", "down", "up", "bias"} | (
                {"scale"} if version == 2 else set()
            )
            if set(weights) != expected_keys:
                raise ValueError("Unexpected projector tensors")
            shapes = {"mean": (width,), "bias": (width,), "scale": (width,)}
            rank = weights["down"].shape[1] if weights["down"].ndim == 2 else 0
            shapes.update(down=(width, rank), up=(rank, width))
            if rank < 1 or any(
                tuple(t.shape) != shapes[k]
                or t.dtype != self.torch.float32
                or not bool(self.torch.isfinite(t).all())
                for k, t in weights.items()
            ):
                raise ValueError("Invalid projector shapes, precision or finite values")
        device = next(self.layers[0].parameters()).device
        return {
            int(i): {k: t.to(device) for k, t in weights.items()}
            for i, weights in artifact["weights"].items()
        }

    def _state_hook(self, index):
        def hook(module, args, output):
            spec = self.spec
            if (
                not self.verify_state
                or spec is None
                or index < spec.cut
                or not self.current_positions
                or self.current_positions[-1] >= spec.prompt_length
            ):
                return
            from vllm.forward_context import get_forward_context

            metadata = get_forward_context().attn_metadata[module.prefix]
            indices = metadata.non_spec_state_indices_tensor.reshape(-1).long()
            for kind, cache in zip(("conv", "recurrent"), module.kv_cache):
                value = cache.index_select(0, indices).detach().float().cpu()
                key = (index, self.current_positions[-1], kind)
                if spec.mode == "capture":
                    self.state_reference[key] = value
                elif key in self.state_reference:
                    expected = self.state_reference[key]
                    difference = value - expected
                    self.state_errors.append(
                        {
                            "layer": index,
                            "position": key[1],
                            "kind": kind,
                            "shape": list(value.shape),
                            "dtype": str(cache.dtype),
                            "max_abs": difference.abs().max().item(),
                            "relative_rms": (
                                difference.square().mean().sqrt()
                                / expected.square().mean().sqrt().clamp_min(1e-30)
                            ).item(),
                            "finite": bool(self.torch.isfinite(value).all()),
                        }
                    )

        return hook

    def _attention_hook(self, index):
        def hook(module, args, kwargs):
            spec = self.spec
            if spec is None or index < spec.cut:
                return
            x = kwargs["hidden_states"]
            if index == spec.cut:
                self.current_positions = kwargs["positions"].detach().cpu().tolist()
                approximate = spec.approximate_count(self.current_positions)
                if len(self.current_positions) != x.shape[0]:
                    raise ValueError("Padded or packed attention input is unsupported")
                self.source = (
                    x.detach().clone()
                    if spec.mode in {"identity", "predict"} and approximate
                    else None
                )
            positions = self.current_positions
            count = spec.approximate_count(positions)
            prompt_count = sum(p < spec.prompt_length for p in positions)
            if spec.mode == "capture" and prompt_count:
                self.capture.setdefault(index, []).append(
                    x[:prompt_count].detach().cpu().clone()
                )
            if spec.mode not in {"oracle", "identity", "predict"} or not count:
                return
            if spec.mode == "oracle":
                start = positions[0]
                replacement = self.oracle[index][start : start + count].to(x.device)
            elif spec.mode == "identity" or index == spec.cut:
                replacement = self.source[:count]
            else:
                w = self.predictor[index]
                source = self.source[:count].float()
                # Fitted low-rank residual map, evaluated in FP32 before BF16 cast.
                replacement = (
                    source * w.get("scale", 1.0)
                    + ((source - w["mean"]) @ w["down"]) @ w["up"]
                    + w["bias"]
                )
            updated = x.clone()
            updated[:count] = replacement.to(dtype=x.dtype)
            kwargs = dict(kwargs, hidden_states=updated)
            counters = self.counts["attention_tokens"]
            counters[index] = counters.get(index, 0) + count
            return args, kwargs

        return hook

    def _mlp_forward(self, index, original):
        def forward(x, *args, **kwargs):
            spec = self.spec
            if (
                spec is None
                or not self.skip_mlp
                or spec.mode not in {"oracle", "identity", "predict"}
                or index < spec.cut
            ):
                return original(x, *args, **kwargs)
            count = spec.approximate_count(self.current_positions)
            if not count:
                return original(x, *args, **kwargs)
            if x.shape[0] != len(self.current_positions):
                raise ValueError("MLP sharding/padding changed the token layout")
            output = self.torch.zeros_like(x)
            if count < x.shape[0]:
                output[count:] = original(x[count:].contiguous(), *args, **kwargs)
            counters = self.counts["mlp_skipped_tokens"]
            counters[index] = counters.get(index, 0) + count
            return output

        return forward

    def report(self, output=None):
        self.torch.cuda.synchronize()
        milliseconds = {}
        for index, start, end in self.events:
            milliseconds[index] = milliseconds.get(index, 0.0) + start.elapsed_time(end)
        operations = {}
        for label, start, end in self.operation_events:
            operations[label] = operations.get(label, 0.0) + start.elapsed_time(end)
        if output:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=False)
            for index, chunks in self.capture.items():
                self.torch.save(
                    self.torch.cat(chunks, dim=0), directory / f"layer-{index}.pt"
                )
        return {
            "counts": self.counts,
            "layer_ms": milliseconds,
            "operation_ms": operations,
            "mla_queries_skipped": self.reference_mask.counts
            if self.reference_mask is not None
            else {},
            "state_errors": self.state_errors,
            "captured_tokens": {
                i: sum(t.shape[0] for t in ts) for i, ts in self.capture.items()
            },
        }


class LLKVWorkerExtension:
    """Explicit RPC entry points; no arbitrary source/eval is accepted."""

    def llkv_configure(self, **kwargs):
        config = self.vllm_config
        if config.scheduler_config.max_num_seqs != 1:
            raise ValueError("LLKV experiments require max_num_seqs=1")
        if config.speculative_config or config.cache_config.enable_prefix_caching:
            raise ValueError("Speculation and prefix caching must be disabled")
        if not config.model_config.enforce_eager:
            raise ValueError("LLKV experiments require eager execution")
        if not hasattr(self, "llkv_experiment"):
            self.llkv_experiment = AttentionInputExperiment(self.get_model())
        return self.llkv_experiment.configure(**kwargs)

    def llkv_report(self, output=None):
        if output:
            output = str(Path(output) / f"rank-{self.rank}")
        return self.llkv_experiment.report(output)
