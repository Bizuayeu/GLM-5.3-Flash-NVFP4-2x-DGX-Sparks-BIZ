"""Fixture-only hashes of the exact blocks selected by the real cache lookup."""

import hashlib

from ..runtime.lpa import LPAWorkerExtension


class APCFixtureWorker(LPAWorkerExtension):
    def apc_fixture_cache_hashes(self, block_ids):
        import torch

        config = self.model_runner.kv_cache_config
        if len(block_ids) != len(config.kv_cache_groups):
            raise ValueError("Fixture cache groups differ from the lookup")
        modules = self.vllm_config.compilation_config.static_forward_context
        result = {}

        def inspect(value, prefix, ids):
            if isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    inspect(item, f"{prefix}/{index}", ids)
                return
            if (
                not isinstance(value, torch.Tensor)
                or value.shape[0] % config.num_blocks
            ):
                raise ValueError(f"Unrecognized fixture cache view: {prefix}")
            ratio = value.shape[0] // config.num_blocks
            if ratio < 1:
                raise ValueError("Empty fixture cache view")
            for logical, block_id in enumerate(ids):
                if block_id is None:
                    continue
                tensor = value.narrow(0, block_id * ratio, ratio).detach().contiguous()
                raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
                result[f"{prefix}/{logical}"] = {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }

        for group, ids in zip(config.kv_cache_groups, block_ids):
            if not ids:
                continue
            for name in group.layer_names:
                inspect(modules[name].kv_cache, name, ids)
        if not result:
            raise ValueError("Fixture lookup did not expose any cached state")
        return result
