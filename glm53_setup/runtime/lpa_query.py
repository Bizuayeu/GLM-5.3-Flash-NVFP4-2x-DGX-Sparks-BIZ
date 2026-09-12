"""Request-scoped suppression of unused, read-only reference MLA queries."""

from contextlib import contextmanager
from contextvars import ContextVar


class ReferenceQueryMask:
    """Cache writes stay upstream; retained queries keep every candidate column."""

    def __init__(self, original):
        self.original = original
        self.scope = ContextVar("glm53_lpa_query_scope", default=None)
        self.counts = {}

    @contextmanager
    def activate(self, count, total, layer):
        if not 0 <= count <= total:
            raise ValueError("Invalid query window")
        token = self.scope.set((count, total, layer))
        try:
            yield
        finally:
            self.scope.reset(token)

    def __call__(self, query, cache, indices, scale):
        scope = self.scope.get()
        if scope is None or scope[0] == 0:
            return self.original(query, cache, indices, scale)
        import torch

        count, total, layer = scope
        if (
            query.ndim != 3
            or query.shape[0] != total
            or query.shape[-1] != 512
            or indices.ndim != 2
            or indices.shape[0] != total
            or cache.dtype != torch.uint8
            or cache.shape[-1] != 656
        ):
            raise ValueError("Reference MLA query layout changed; refusing masking")
        output = torch.zeros_like(query)
        if count < total:
            output[count:] = self.original(query[count:], cache, indices[count:], scale)
        self.counts[layer] = self.counts.get(layer, 0) + count
        return output
