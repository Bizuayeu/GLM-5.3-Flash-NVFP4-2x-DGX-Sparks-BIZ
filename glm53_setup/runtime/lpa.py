"""Opt-in experimental attention-input replay; never changes checkpoint weights.

The vLLM worker extension is for a serial, eager, text-only experiment. It
retains each attention/state update and skips only unneeded historical MLP
rows. No runtime patch or GPU dependency is activated by importing this file.
"""

from dataclasses import dataclass
from pathlib import Path

from ..config import REVISION, TEACHER_PRECISION

# The projector artifact this loader reads; train_lpa and the APC/LPA fixture write it.
PROJECTOR_FORMAT = 2
# Late layers write attention state from the projection p and read with their
# own input x; split-self writes from x through the same path, as a control.
SPLIT_MODES = frozenset({"split", "split-self"})


@dataclass(frozen=True)
class ExperimentSpec:
    mode: str
    cut: int
    prompt_length: int
    tail: int = 1
    approximate_start: int = 0

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in {
            "off",
            "capture",
            "oracle",
            "identity",
            "predict",
            *SPLIT_MODES,
        }:
            raise ValueError("Unsupported experiment mode")
        if any(
            type(value) is not int
            for value in (
                self.cut,
                self.prompt_length,
                self.tail,
                self.approximate_start,
            )
        ):
            raise ValueError("Cut, prompt length and tail must be integers")
        if self.cut < 0 or self.prompt_length < 1:
            raise ValueError("Invalid cut or prompt length")
        if not 1 <= self.tail <= self.prompt_length:
            raise ValueError("The exact tail must include the final prompt token")
        if not 0 <= self.approximate_start <= self.prompt_length:
            raise ValueError("Invalid first approximation position")
        if self.mode in SPLIT_MODES and (self.tail != 1 or self.approximate_start):
            raise ValueError("Split modes cover every token; no tail or start")

    def approximate_count(self, positions):
        start, stop = self.approximate_span(positions)
        return stop - start

    def validate_scheduled_start(self, positions, expected, speculative_decode=False):
        if expected is None:
            return False
        actual = positions[0] if positions else None
        if actual == expected:
            return False
        # V2 keeps an optimistic CPU cursor during asynchronous MTP decode;
        # the GPU subtracts rejected draft tokens. Never relax a prompt boundary.
        if (
            speculative_decode
            and actual is not None
            and self.prompt_length <= actual < expected
        ):
            return True
        raise ValueError(
            "Model positions differ from the scheduled APC/LPA boundary: "
            f"expected={expected}, actual={actual}, prompt={self.prompt_length}"
        )

    def approximate_span(self, positions):
        if any(type(p) is not int or p < 0 for p in positions) or any(
            b != a + 1 for a, b in zip(positions, positions[1:])
        ):
            raise ValueError("Only one contiguous, unpadded sequence is supported")
        if not positions:
            return 0, 0
        start = min(len(positions), max(0, self.approximate_start - positions[0]))
        stop = min(
            len(positions), max(0, self.prompt_length - self.tail - positions[0])
        )
        return (start, stop) if stop > start else (0, 0)


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
        self.expected_position = None
        self.speculative_decode = False
        self.speculative_position_corrections = 0
        self.source = None
        self.counts = {}
        self.events = []
        self.operation_events = []
        self.profile = False
        self.skip_mlp = True
        self.verify_state = False
        self.state_reference = {}
        self.state_errors = []
        self.split_inputs = {}
        self.split_installed = False
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
            start, stop = spec.approximate_span(self.current_positions)
            count = stop - start
            if not count:
                return original(*args, **kwargs)
            before = self.reference_mask.counts.get(index, 0)
            with self.reference_mask.activate(
                count, len(self.current_positions), index, start=start
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
        approximate_start=0,
    ):
        spec = ExperimentSpec(mode, cut, prompt_length, tail, approximate_start)
        if cut >= len(self.layers):
            raise ValueError("Cut must precede the final layer")
        if type(skip_mla_queries) is not bool:
            raise ValueError("skip_mla_queries must be boolean")
        if mode in SPLIT_MODES and skip_mla_queries:
            raise ValueError("Split modes keep every MLA query")
        if skip_mla_queries and self.reference_mask is None:
            import glm53_reference

            from .lpa_query import ReferenceQueryMask

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
            spec = ExperimentSpec(mode, cut, prompt_length, tail, approximate_start)
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
        if mode in {"predict", "split"}:
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
        if mode in SPLIT_MODES and not self.split_installed:
            self._install_split()
        self.spec = spec
        self.profile = profile
        self.verify_state = verify_state
        self.skip_mlp = skip_mlp
        self.state_errors = []
        self.source = None
        self.current_positions = []
        self.expected_position = None
        self.speculative_decode = False
        self.speculative_position_corrections = 0
        self.counts = {"attention_tokens": {}, "mlp_skipped_tokens": {}}
        if mode in SPLIT_MODES:
            self.counts["split_tokens"] = {}
        self.split_inputs = {}
        self.events = []
        self.operation_events = []
        return {
            "mode": mode,
            "requested_mode": requested_mode,
            "layers": len(self.layers),
            "cut": cut,
            "prompt_length": prompt_length,
            "tail": tail,
            "approximate_start": approximate_start,
            "skip_mla_queries": skip_mla_queries,
        }

    def _load_predictor(self, predictor_path, cut):
        artifact = self.torch.load(
            predictor_path, map_location="cpu", weights_only=True
        )
        if artifact.get("format_version") != PROJECTOR_FORMAT:
            raise ValueError("Unsupported projector artifact version")
        if (
            artifact.get("teacher_revision") != REVISION
            or artifact.get("teacher_precision") != TEACHER_PRECISION
        ):
            raise ValueError(
                "Projector teacher does not match the pinned model/precision"
            )
        if artifact["cut"] != cut or artifact["layers"] != len(self.layers):
            raise ValueError("Predictor does not match the layer boundary")
        if set(artifact["weights"]) != set(range(cut + 1, len(self.layers))):
            raise ValueError("Projector layers are incomplete or unexpected")
        width = self.layers[cut].hidden_size
        for weights in artifact["weights"].values():
            expected_keys = {"mean", "down", "up", "bias", "scale"}
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
                value = cache.index_select(0, indices)
                if kind == "conv":
                    # Prefill writes only the first kernel_width-1 slots. MTP
                    # reserves extra rollback slots that are not live yet.
                    axis = 2 if module._conv_state_dim_first else 1
                    value = value.narrow(axis, 0, module.conv_size - 1)
                value = value.detach().float().cpu()
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
            if spec.mode == "off" and not self.verify_state:
                return
            x = kwargs["hidden_states"]
            if index == spec.cut:
                self.current_positions = kwargs["positions"].detach().cpu().tolist()
                corrected = spec.validate_scheduled_start(
                    self.current_positions,
                    self.expected_position,
                    self.speculative_decode,
                )
                self.speculative_position_corrections += int(corrected)
                approximate = spec.approximate_count(self.current_positions)
                if len(self.current_positions) != x.shape[0]:
                    raise ValueError("Padded or packed attention input is unsupported")
                self.source = (
                    x.detach().clone()
                    if (spec.mode in {"identity", "predict"} and approximate)
                    or spec.mode == "split"
                    else None
                )
            if spec.mode in SPLIT_MODES:
                if index > spec.cut:
                    self.split_inputs[index] = (
                        x if spec.mode == "split-self" else self._project(index, x)
                    )
                    counters = self.counts["split_tokens"]
                    counters[index] = counters.get(index, 0) + x.shape[0]
                return
            positions = self.current_positions
            span_start, span_stop = spec.approximate_span(positions)
            count = span_stop - span_start
            prompt_count = sum(p < spec.prompt_length for p in positions)
            if spec.mode == "capture" and prompt_count:
                self.capture.setdefault(index, []).append(
                    x[:prompt_count].detach().cpu().clone()
                )
            if spec.mode not in {"oracle", "identity", "predict"} or not count:
                return
            if spec.mode == "oracle":
                start = positions[span_start]
                replacement = self.oracle[index][start : start + count].to(x.device)
            elif spec.mode == "identity" or index == spec.cut:
                replacement = self.source[span_start:span_stop]
            else:
                replacement = self._predict(index, self.source[span_start:span_stop])
            updated = x.clone()
            updated[span_start:span_stop] = replacement.to(dtype=x.dtype)
            kwargs = dict(kwargs, hidden_states=updated)
            counters = self.counts["attention_tokens"]
            counters[index] = counters.get(index, 0) + count
            return args, kwargs

        return hook

    def _predict(self, index, source):
        w = self.predictor[index]
        source = source.float()
        # Fitted low-rank residual map, evaluated in FP32 before BF16 cast.
        return (
            source * w["scale"]
            + ((source - w["mean"]) @ w["down"]) @ w["up"]
            + w["bias"]
        )

    def _project(self, index, x):
        if self.source.shape != x.shape:
            raise ValueError("Split source and layer input differ in shape")
        return self._predict(index, self.source).to(dtype=x.dtype)

    def _install_split(self):
        """Wrap the late-layer projections that write attention state.

        Call sites and column splits follow the pinned overlays and vLLM
        385dce36: KDA k_proj, v_proj and the b|f_a columns of in_proj_bfg_a;
        MLA the kv_a columns of fused_qkv_a_proj, the indexer's wk columns of
        wk_weights_proj and its kpool gate score. Queries, weights_proj, g_a
        and everything after attention still read the layer input.
        """
        # Every layer is checked before any forward is replaced.
        plans = []
        for layer in self.layers:
            index, attn = layer.layer_idx, layer.self_attn
            if layer.layer_kind == "kda":
                wrapped = [
                    (attn.k_proj, None),
                    (attn.v_proj, None),
                    (
                        attn.in_proj_bfg_a,
                        slice(0, attn.local_num_heads + attn.head_dim),
                    ),
                ]
                indexer = None
            else:
                indexer = attn.indexer
                if (
                    indexer is None
                    or attn.mla_attn.fused_qkv_a_proj is not attn.fused_qkv_a_proj
                    or attn.mla_attn.indexer is not indexer
                ):
                    raise ValueError("Unexpected MLA projection layout")
                wrapped = [
                    (attn.fused_qkv_a_proj, slice(attn.q_lora_rank, None)),
                    (indexer.wk_weights_proj, slice(0, indexer.head_dim)),
                ]
            plans.append((index, attn, wrapped, indexer))
        for index, attn, wrapped, indexer in plans:
            for module, columns in wrapped:
                module.forward = self._split_projection(index, module.forward, columns)
            if indexer is not None:
                indexer.indexer_op.forward = self._split_gate(
                    index, indexer, indexer.indexer_op.forward
                )
            self.handles.append(attn.register_forward_hook(self._split_release(index)))
        self.split_installed = True

    def _split_release(self, index):
        def hook(module, args, output):
            self.split_inputs.pop(index, None)

        return hook

    def _split_input(self, index, x):
        spec = self.spec
        if spec is None or spec.mode not in SPLIT_MODES or index <= spec.cut:
            return None
        p = self.split_inputs.get(index)
        if p is None or p.shape != x.shape:
            raise ValueError("Split projection input is missing or misaligned")
        return p

    def _split_projection(self, index, original, columns):
        """Whole output from p, or only ``columns`` from p and the rest from x."""

        def forward(x, *args, **kwargs):
            p = self._split_input(index, x)
            if p is None:
                return original(x, *args, **kwargs)
            written = original(p, *args, **kwargs)
            if columns is None:
                return written
            read = original(x, *args, **kwargs)
            output = read[0].clone()
            output[..., columns] = written[0][..., columns]
            return (output, *read[1:])

        return forward

    def _split_gate(self, index, indexer, original):
        def forward(hidden_states, *args, **kwargs):
            p = self._split_input(index, hidden_states)
            if p is None:
                return original(hidden_states, *args, **kwargs)
            if kwargs.get("gate_score") is None:
                raise ValueError("Indexer gate score was not passed by keyword")
            gate = self.torch.nn.functional.linear(p, indexer.index_kpool_compress_gate)
            return original(hidden_states, *args, **dict(kwargs, gate_score=gate))

        return forward

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
            start, stop = spec.approximate_span(self.current_positions)
            count = stop - start
            if not count:
                return original(x, *args, **kwargs)
            if x.shape[0] != len(self.current_positions):
                raise ValueError("MLP sharding/padding changed the token layout")
            output = self.torch.zeros_like(x)
            if start:
                output[:start] = original(x[:start].contiguous(), *args, **kwargs)
            if stop < x.shape[0]:
                output[stop:] = original(x[stop:].contiguous(), *args, **kwargs)
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
            "speculative_position_corrections": self.speculative_position_corrections,
            "captured_tokens": {
                i: sum(t.shape[0] for t in ts) for i, ts in self.capture.items()
            },
        }


class LPAWorkerExtension:
    """Explicit RPC entry points; no arbitrary source/eval is accepted."""

    def apc_lpa_report(self):
        from .apc_worker import report

        return report(self)

    def apc_cache_layout(self):
        """Read-only worker layout evidence; no cache tensors or keys are exported."""
        cache = self.model_runner.kv_cache_config
        return {
            "rank": self.rank,
            "num_blocks": cache.num_blocks,
            "retention_interval": cache.prefix_cache_retention_interval,
            "groups": [
                {
                    "kind": type(group.kv_cache_spec).__name__,
                    "block_size": group.kv_cache_spec.block_size,
                    "layers": list(group.layer_names),
                }
                for group in cache.kv_cache_groups
            ],
        }

    def lpa_configure(self, allow_mtp=False, **kwargs):
        config = self.vllm_config
        if config.scheduler_config.max_num_seqs != 1:
            raise ValueError("LPA experiments require max_num_seqs=1")
        if type(allow_mtp) is not bool:
            raise ValueError("allow_mtp must be boolean")
        speculative = config.speculative_config
        if speculative and (
            not allow_mtp
            or speculative.method != "mtp"
            or speculative.num_speculative_tokens not in (1, 2, 3)
        ):
            raise ValueError("Only explicitly enabled MTP k=1/k=2/k=3 is supported")
        if config.cache_config.enable_prefix_caching:
            raise ValueError("Prefix caching must be disabled")
        if speculative and kwargs.get("mode") in SPLIT_MODES:
            raise ValueError("Split modes do not support MTP")
        from .graph_policy import lpa_execution_supported

        if not lpa_execution_supported(config):
            raise ValueError("LPA requires eager execution")
        if not hasattr(self, "lpa_experiment"):
            # get_model() is the target model, never model_runner.drafter.model.
            # The constructor rejects any MTP layer in this module tree. Draft
            # proposals/cache and acceptance/rollback remain owned by vLLM.
            self.lpa_experiment = AttentionInputExperiment(self.get_model())
        return self.lpa_experiment.configure(**kwargs)

    def lpa_report(self, output=None):
        if not hasattr(self, "lpa_experiment"):
            raise ValueError("Configure LPA before requesting a report")
        if output:
            output = str(Path(output) / f"rank-{self.rank}")
        return self.lpa_experiment.report(output)
