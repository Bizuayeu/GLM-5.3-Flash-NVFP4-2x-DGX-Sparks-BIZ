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
        if path.exists():
            raise ValueError("Do not overwrite a pipeline observation")
        sequence = 0

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
                raw = (
                    value.detach()
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                record["tensors"][name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

        def before(module, args, kwargs):
            record(kwargs["intermediate_tensors"], "received")

        def after(module, args, kwargs, output):
            record(output, "sent")

        handles = []
        if not get_pp_group().is_first_rank:
            handles.append(model.register_forward_pre_hook(before, with_kwargs=True))
        if not get_pp_group().is_last_rank:
            handles.append(model.register_forward_hook(after, with_kwargs=True))
        self._pipeline_handles = handles
        return {"rank": self.rank, "observation": str(path), "hooks": len(handles)}
