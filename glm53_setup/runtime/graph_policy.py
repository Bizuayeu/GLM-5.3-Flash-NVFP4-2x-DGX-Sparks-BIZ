"""Keep LPA's Python hooks in eager prefill while decode replays native graphs."""

# cc-defer: LPA/Graph routing is fixture preparation, not a supported startup
# combination; qualify its GPU state and full-model integration before enabling.

import os


def force_eager_prefill(batch_state):
    return bool(
        os.environ.get("GLM53_LPA_GRAPH_PREFILL") == "1"
        and batch_state is not None
        and batch_state.has_prefill
    )


def lpa_execution_supported(config):
    if config.model_config.enforce_eager:
        return True
    compilation = config.compilation_config
    return (
        compilation.mode.name == "NONE"
        and compilation.cudagraph_mode.name == "FULL_DECODE_ONLY"
        and os.environ.get("GLM53_LPA_GRAPH_PREFILL") == "1"
        and os.environ.get("GLM53_ASYNC_INDEX_CHECKS") == "1"
    )
