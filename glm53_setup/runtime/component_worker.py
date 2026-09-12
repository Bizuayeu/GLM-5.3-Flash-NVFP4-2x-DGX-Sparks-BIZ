"""Exclusive component diagnostics before LPA/MTP integration."""

import os

from .indexer_worker import IndexerCaptureWorker


class ComponentWorker(IndexerCaptureWorker):
    def unpack_configure(self, enabled):
        cfg = self.vllm_config
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (
            cfg.speculative_config
            or cfg.scheduler_config.max_num_seqs != 1
            or cfg.cache_config.enable_prefix_caching
            or not cfg.model_config.enforce_eager
        ):
            raise ValueError(
                "Component diagnostics require eager, one sequence, no MTP/prefix cache"
            )
        if hasattr(self, "lpa_experiment") or hasattr(self, "indexer_capture"):
            raise ValueError("Finish the active observation before changing unpack")
        previous = os.environ.get("GLM53_FUSED_UNPACK") == "1"
        os.environ["GLM53_FUSED_UNPACK"] = "1" if enabled else "0"
        return {"rank": self.rank, "previous": previous, "fused_unpack": enabled}
