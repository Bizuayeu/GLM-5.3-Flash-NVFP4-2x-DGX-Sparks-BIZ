"""Indexer observation RPCs independent of LPA and speculative decoding."""

import re

from .indexer_capture import IndexerCapture


class IndexerCaptureWorker:
    def indexer_capture_start(self, request_id, positions=(), max_bytes=1048576):
        cfg = self.vllm_config
        if (
            cfg.speculative_config
            or cfg.scheduler_config.max_num_seqs != 1
            or cfg.cache_config.enable_prefix_caching
            or not cfg.model_config.enforce_eager
        ):
            raise ValueError(
                "Capture requires eager, one sequence, no MTP/prefix cache"
            )
        if hasattr(self, "indexer_capture") or hasattr(self, "lpa_experiment"):
            raise ValueError("A capture/LPA experiment is already attached")
        bindings = {}
        for name, module in self.get_model().named_modules():
            if type(module).__name__ == "SparseAttnIndexerKpool":
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if match is None or int(match[1]) in bindings:
                    raise ValueError("Ambiguous indexer module binding")
                bindings[int(match[1])] = module
        capture = IndexerCapture(bindings, request_id, positions, max_bytes)
        capture.__enter__()
        self.indexer_capture = capture
        return {"rank": self.rank, "layers": sorted(bindings), "request_id": request_id}

    def indexer_capture_finish(self):
        capture = self.indexer_capture
        del self.indexer_capture
        capture.__exit__(None, None, None)
        return {"rank": self.rank, **capture.report()}

    def indexer_capture_abort(self):
        if hasattr(self, "indexer_capture"):
            capture = self.indexer_capture
            del self.indexer_capture
            capture.__exit__(None, None, None)
        return {"rank": self.rank, "detached": True}
