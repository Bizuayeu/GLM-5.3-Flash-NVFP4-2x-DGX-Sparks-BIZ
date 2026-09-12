"""Bounded, scoped observation of logical candidates at the kpool op boundary."""


class IndexerCapture:
    def __init__(
        self, bindings, request_id, positions=(), max_bytes=1048576, max_events=4096
    ):
        if not bindings or not isinstance(request_id, str) or not request_id:
            raise ValueError("Explicit layer bindings and request identity required")
        if any(type(p) is not int or p < 0 for p in positions):
            raise ValueError("Invalid query positions")
        if (
            type(max_bytes) is not int
            or max_bytes < 1
            or type(max_events) is not int
            or max_events < 1
        ):
            raise ValueError("Positive capture budgets required")
        self.bindings = dict(bindings)
        if any(type(layer) is not int or layer < 0 for layer in self.bindings) or len(
            {id(module) for module in self.bindings.values()}
        ) != len(self.bindings):
            raise ValueError("Distinct modules with nonnegative layer IDs required")
        self.request_id = request_id
        self.positions = set(positions)
        self.max_bytes = max_bytes
        self.max_events = max_events
        self.used_bytes = 0
        self.events = []
        self.rows = []
        self.handles = []
        self.pending = {}
        self.closed = False

    def __enter__(self):
        if self.closed or self.handles or self.events or self.rows:
            raise ValueError("Capture session cannot be reused")
        try:
            for layer, module in self.bindings.items():
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._before(layer), with_kwargs=True
                    )
                )
                self.handles.append(
                    module.register_forward_hook(self._after(layer), with_kwargs=True)
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()
        self.closed = True

    def _before(self, layer):
        def hook(module, args, kwargs):
            import torch

            positions = kwargs.get("positions")
            if positions is None:
                return  # Dummy/model-load path is not measured data.
            if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
                raise ValueError("Integer query positions required")
            if layer in self.pending or len(self.events) >= self.max_events:
                raise ValueError("Reentrant call or indexer event budget exceeded")
            start = torch.cuda.Event(enable_timing=True) if positions.is_cuda else None
            if start is not None:
                start.record()
            self.pending[layer] = (start, positions)

        return hook

    def _after(self, layer):
        def hook(module, args, kwargs, output):
            import torch

            pending = self.pending.pop(layer, None)
            if pending is None:
                return
            start, positions = pending
            end = torch.cuda.Event(enable_timing=True) if start is not None else None
            if end is not None:
                end.record()
            self.events.append((layer, start, end))
            if not self.positions:
                return  # Timing-only run avoids host position reads/candidate copies.
            if (
                output.ndim != 2
                or output.shape[0] < positions.numel()
                or positions.ndim != 1
            ):
                raise ValueError("Unexpected kpool output/query layout")
            selected = [
                (i, p)
                for i, p in enumerate(positions.detach().cpu().tolist())
                if p in self.positions
            ]
            required = len(selected) * output.shape[1] * output.element_size()
            if self.used_bytes + required > self.max_bytes:
                raise ValueError("Indexer capture byte budget exceeded")
            for i, position in selected:
                # The backend overwrites this buffer in later layers: clone now.
                self.rows.append((layer, position, output[i].detach().clone()))
            self.used_bytes += required

        return hook

    def report(self):
        layer_ms = {}
        for layer, start, end in self.events:
            if start is not None:
                end.synchronize()
                layer_ms[layer] = layer_ms.get(layer, 0.0) + start.elapsed_time(end)
        rows = []
        for layer, position, tensor in self.rows:
            values = tensor.cpu().tolist()
            if any(type(v) is not int or v < -1 or v > position for v in values):
                raise ValueError("Output is not causal logical token indices")
            rows.append(
                {
                    "request_id": self.request_id,
                    "layer": layer,
                    "query_position": position,
                    "coordinate_space": "logical_tokens",
                    "indices": values,
                }
            )
        return {
            "rows": rows,
            "layer_ms": layer_ms,
            "events": len(self.events),
            "captured_bytes": self.used_bytes,
            "timing_only": not self.positions,
        }
