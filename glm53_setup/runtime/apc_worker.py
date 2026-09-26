"""Consume scheduler-authored APC/LPA policy before the target forward pass."""

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .apc_policy import PrefillPolicy
from .apc_runtime import POLICY_KEY, RequestState, audit, settings

_WARMUP_OWNER = ContextVar("glm53_apc_lpa_warmup", default=None)


@contextmanager
def warmup_scope(worker):
    if settings() is None:
        yield
        return
    if hasattr(worker, "lpa_experiment") or getattr(worker, "_glm53_apc_requests", {}):
        raise ValueError("APC/LPA warmup must precede real request admission")
    token = _WARMUP_OWNER.set(worker)
    try:
        yield
    finally:
        _WARMUP_OWNER.reset(token)


def validate_worker(config):
    parallel = config.parallel_config
    cache = config.cache_config
    if (
        not config.model_config.enforce_eager
        or config.scheduler_config.max_num_seqs != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.tensor_parallel_size not in (1, 2)
        or parallel.data_parallel_size != 1
        or parallel.enable_expert_parallel
        or not cache.enable_prefix_caching
        # DeepseekV4Attention resolves the public fp8 option during model loading.
        or cache.cache_dtype != "fp8_ds_mla"
        or cache.mamba_cache_mode != "align"
        or cache.enable_mamba_fine_grained_prefix_cache
        or cache.prefix_match_unit is not None
        or config.kv_transfer_config is not None
        or config.ec_transfer_config is not None
    ):
        raise ValueError(
            "APC/LPA requires eager serial local aligned cache with TP1/TP2: "
            + str(
                {
                    "eager": config.model_config.enforce_eager,
                    "sequences": config.scheduler_config.max_num_seqs,
                    "tp": parallel.tensor_parallel_size,
                    "pp": parallel.pipeline_parallel_size,
                    "dp": parallel.data_parallel_size,
                    "ep": parallel.enable_expert_parallel,
                    "prefix": cache.enable_prefix_caching,
                    "dtype": cache.cache_dtype,
                    "mamba": cache.mamba_cache_mode,
                    "fine_grained": cache.enable_mamba_fine_grained_prefix_cache,
                    "match_unit": cache.prefix_match_unit,
                    "external_kv": config.kv_transfer_config is not None,
                    "external_encoder": config.ec_transfer_config is not None,
                }
            )
        )
    speculative = config.speculative_config
    if speculative is not None and (
        speculative.method != "mtp"
        or speculative.num_speculative_tokens not in (1, 2, 3)
    ):
        raise ValueError("APC/LPA supports only the GLM MTP candidate")


def _read_policy(data, config):
    extra = (
        data.sampling_params.extra_args if data.sampling_params is not None else None
    )
    wire = (extra or {}).get(POLICY_KEY)
    required = {"version", "request_id", "policy", "signature", "override"}
    if not isinstance(wire, dict) or wire.keys() != required:
        raise ValueError("Worker did not receive the cache manager's LPA policy")
    if (
        type(wire["version"]) is not int
        or wire["version"] != 1
        or wire["request_id"] != data.req_id
        or wire["signature"] != config.signature
        or wire["override"] not in ("auto", "off")
    ):
        raise ValueError("Worker policy identity mismatch")
    policy = PrefillPolicy.from_dict(wire["policy"])
    if (
        policy.tail != min(config.tail, policy.prompt_tokens)
        or policy.break_even != config.break_even
        or (wire["override"] == "off" and policy.approximate_start is not None)
    ):
        raise ValueError("Worker policy does not match configured computation")
    if (
        data.prompt_token_ids is None
        or len(data.prompt_token_ids) != policy.prompt_tokens
        or min(data.num_computed_tokens, policy.prompt_tokens) != policy.cached_tokens
    ):
        raise ValueError("Worker prompt/restored boundary differs from admission")
    return RequestState(data.req_id, policy, config.signature, wire["override"])


def _verify_projector(worker, config):
    path = Path(config.projector_path)
    stat = path.stat()
    identity = (config.signature, stat.st_mtime_ns, stat.st_size)
    if getattr(worker, "_glm53_apc_projector", None) != identity:
        if hashlib.sha256(path.read_bytes()).hexdigest() != config.projector_sha256:
            raise ValueError("APC/LPA projector digest mismatch")
        worker._glm53_apc_projector = identity


def before_forward(worker, scheduler_output):
    config = settings()
    if config is None or _WARMUP_OWNER.get() is worker:
        return
    if not hasattr(worker, "_glm53_apc_requests"):
        validate_worker(worker.vllm_config)
        worker._glm53_apc_requests = {}
        worker._glm53_apc_owner = None
        worker._glm53_apc_last = None
    policies = worker._glm53_apc_requests
    for request_id in scheduler_output.finished_req_ids:
        policies.pop(request_id, None)
    changed = set()
    positions = {}
    for data in scheduler_output.scheduled_new_reqs:
        state = _read_policy(data, config)
        policies[data.req_id] = state
        changed.add(data.req_id)
        positions[data.req_id] = data.num_computed_tokens
    cached = scheduler_output.scheduled_cached_reqs
    positions.update(zip(cached.req_ids, cached.num_computed_tokens))
    active = [
        key for key, count in scheduler_output.num_scheduled_tokens.items() if count > 0
    ]
    if not active:
        return
    if len(active) != 1 or active[0] not in policies or active[0] not in positions:
        raise ValueError("APC/LPA scheduled request ownership is not serial")
    request_id = active[0]
    state = policies[request_id]
    if state.signature != config.signature:
        raise ValueError("APC/LPA configuration changed during a request")
    if worker._glm53_apc_owner != request_id or request_id in changed:
        policy = state.policy
        mode = "predict" if policy.approximates else "off"
        if mode == "predict":
            _verify_projector(worker, config)
            if not hasattr(worker, "lpa_experiment"):
                from .lpa import AttentionInputExperiment

                worker.lpa_experiment = AttentionInputExperiment(worker.get_model())
        if hasattr(worker, "lpa_experiment"):
            worker.lpa_experiment.configure(
                mode=mode,
                cut=config.cut,
                prompt_length=policy.prompt_tokens,
                tail=policy.tail,
                approximate_start=policy.approximate_start or 0,
                predictor_path=config.projector_path,
                skip_mla_queries=config.skip_mla_queries,
            )
        worker._glm53_apc_owner = request_id
        worker._glm53_apc_last = state
        audit("worker_policy", rank=worker.rank, **state.wire())
    if hasattr(worker, "lpa_experiment"):
        worker.lpa_experiment.expected_position = positions[request_id]
        generated = dict(zip(cached.req_ids, cached.num_output_tokens)).get(
            request_id, 0
        )
        worker.lpa_experiment.speculative_decode = (
            worker.vllm_config.speculative_config is not None
            and generated > 0
            and positions[request_id] >= state.policy.prompt_tokens
        )


def report(worker):
    state = getattr(worker, "_glm53_apc_last", None)
    if state is None:
        return {"policy": None, "lpa": None}
    experiment = getattr(worker, "lpa_experiment", None)
    return {
        "policy": state.wire(),
        "eligible_tokens": state.policy.eligible_tokens,
        "lpa": experiment.report() if experiment is not None else None,
    }
