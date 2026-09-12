"""Fixture-only PP observation; hashes actual stage-boundary tensors."""

import hashlib
import json
from pathlib import Path


class PipelineFixtureWorker:
    def pipeline_info(self):
        import torch
        from vllm.distributed import get_pp_group

        model = self.get_model().language_model.model
        buffers = model.make_empty_intermediate_tensors(1, torch.bfloat16, self.device)
        return {
            "rank": self.rank,
            "first": get_pp_group().is_first_rank,
            "last": get_pp_group().is_last_rank,
            "layers": [layer.layer_idx for layer in model._active_layers],
            "buffers": {
                name: {
                    "shape": list(buffers[name].shape),
                    "dtype": str(buffers[name].dtype),
                }
                for name in ("hidden_states", "residual", "post", "comb")
            },
        }

    def pipeline_observe(self):
        import torch
        from vllm.distributed import get_pp_group

        if hasattr(self, "_pipeline_handles"):
            raise ValueError("Pipeline fixture observation already installed")
        model = self.get_model().language_model.model
        path = Path("/out") / f"pipeline-state-rank{self.rank}.jsonl"
        layer_path = Path("/out") / f"pipeline-layer-rank{self.rank}.jsonl"
        if path.exists() or layer_path.exists():
            raise ValueError("Do not overwrite a pipeline observation")
        sequence = 0
        step = 0
        positions = []

        def describe(value):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite pipeline fixture tensor")
            raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }

        def record(tensors, direction):
            nonlocal sequence
            sequence += 1
            record = {"sequence": sequence, "direction": direction, "tensors": {}}
            for name in ("hidden_states", "residual", "post", "comb"):
                value = tensors[name]
                expected_dtype = (
                    torch.float32 if name in ("post", "comb") else torch.bfloat16
                )
                if value.dtype != expected_dtype:
                    raise ValueError(f"Unexpected {name} dtype at PP boundary")
                record["tensors"][name] = describe(value)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

        def before(module, args, kwargs):
            record(kwargs["intermediate_tensors"], "received")

        def after(module, args, kwargs, output):
            record(output, "sent")

        def begin_step(module, args, kwargs):
            nonlocal step, positions
            step += 1
            positions = kwargs["positions"].detach().cpu().tolist()

        def layer_hook(index, kind):
            def hook(module, args, output):
                from vllm.forward_context import get_forward_context

                row = {
                    "step": step,
                    "layer": index,
                    "positions": positions,
                    "attention_output": describe(output),
                    "kda_state": {},
                }
                if kind == "kda":
                    metadata = get_forward_context().attn_metadata[module.prefix]
                    indices = metadata.non_spec_state_indices_tensor.reshape(-1).long()
                    for name, cache in zip(("conv", "recurrent"), module.kv_cache):
                        value = cache.index_select(0, indices)
                        if name == "conv":
                            axis = 2 if module._conv_state_dim_first else 1
                            value = value.narrow(axis, 0, module.conv_size - 1)
                        row["kda_state"][name] = describe(value)
                with layer_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")

            return hook

        handles = [model.register_forward_pre_hook(begin_step, with_kwargs=True)]
        for layer in model._active_layers:
            handles.append(
                layer.self_attn.register_forward_hook(
                    layer_hook(layer.layer_idx, layer.layer_kind)
                )
            )
        if not get_pp_group().is_first_rank:
            handles.append(model.register_forward_pre_hook(before, with_kwargs=True))
        if not get_pp_group().is_last_rank:
            handles.append(model.register_forward_hook(after, with_kwargs=True))
        self._pipeline_handles = handles
        return {
            "rank": self.rank,
            "observation": str(path),
            "layer_observation": str(layer_path),
            "hooks": len(handles),
        }
