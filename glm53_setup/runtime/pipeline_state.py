"""Typed intermediate buffers for the independent GLM mHC PP fixture."""


def common_layout_names(supported):
    """Retain only layouts supported by every stage, in rank-zero preference."""
    if not supported or not all(supported):
        raise ValueError("No worker reported supported KV cache layouts")
    common = set.intersection(*(set(names) for names in supported))
    if not common:
        raise ValueError(f"No common pipeline KV cache layout: {supported}")
    return [name for name in supported[0] if name in common]


def validate_pipeline(config):
    parallel = config.parallel_config
    if parallel.pipeline_parallel_size == 1:
        return
    if (
        parallel.pipeline_parallel_size != 2
        or parallel.tensor_parallel_size != 1
        or parallel.use_sequence_parallel_moe
        or config.speculative_config is not None
        or not config.model_config.enforce_eager
        or config.cache_config.enable_prefix_caching
        or not config.model_config.hf_text_config.mhc
    ):
        raise ValueError("PP fixture requires PP2/TP1, mHC, eager, no SP/MTP/APC")


def allocate_intermediate(config, batch_size, dtype, device):
    import torch

    if type(batch_size) is not int or batch_size < 0:
        raise ValueError("Invalid pipeline batch_size")
    if not config.mhc or config.mhc_num_residual_streams != 4:
        raise ValueError("Only the pinned four-stream mHC layout is supported")
    if dtype != torch.bfloat16:
        raise ValueError("PP fixture requires BF16 hidden states")
    n, hidden = config.mhc_num_residual_streams, config.hidden_size
    return {
        "hidden_states": torch.zeros((batch_size, hidden), dtype=dtype, device=device),
        "residual": torch.zeros((batch_size, n, hidden), dtype=dtype, device=device),
        "post": torch.zeros((batch_size, n, 1), dtype=torch.float32, device=device),
        "comb": torch.zeros((batch_size, n, n), dtype=torch.float32, device=device),
    }
