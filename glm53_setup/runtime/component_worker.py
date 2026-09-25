"""Exclusive CUDA/indexer component diagnostics: eager, one sequence, no LPA/MTP/APC."""

import os

from .indexer_worker import IndexerCaptureWorker


class ComponentWorker(IndexerCaptureWorker):
    def _guard_mode_change(self, enabled):
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
            raise ValueError("Finish the active observation before changing mode")

    def unpack_configure(self, enabled):
        self._guard_mode_change(enabled)
        previous = os.environ.get("GLM53_FUSED_UNPACK") == "1"
        os.environ["GLM53_FUSED_UNPACK"] = "1" if enabled else "0"
        return {"rank": self.rank, "previous": previous, "fused_unpack": enabled}

    def index_checks_configure(self, asynchronous):
        self._guard_mode_change(asynchronous)
        previous = os.environ.get("GLM53_ASYNC_INDEX_CHECKS") == "1"
        os.environ["GLM53_ASYNC_INDEX_CHECKS"] = "1" if asynchronous else "0"
        return {
            "rank": self.rank,
            "previous": previous,
            "asynchronous_index_checks": asynchronous,
        }
