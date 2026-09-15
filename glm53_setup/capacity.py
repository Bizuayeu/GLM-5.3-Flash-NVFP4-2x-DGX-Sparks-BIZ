"""Read the KV pool as the running engine reports it, without a runtime patch.

vLLM's boot line "GPU KV cache size: N tokens" is max_concurrency x max_model_len
for this hybrid model (MLA, IndexPool tail, KDA state groups, MTP draft), not the
number of conversation tokens the prefix cache can hold. This module separates
the two and only estimates cached-conversation capacity when every group's spec
kind is one it models; otherwise the figure is withheld, not guessed.
"""

import math
import re

STOCK = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens "
    r"per request: ([\d.]+)x"
)
GROUPS = re.compile(r"kv cache group sizes \[([^\]]*)\]")
SCHEDULER_BLOCK = re.compile(r"kv lcm block sizes (\d+)")
CACHE_INFO = re.compile(r"vllm:cache_config_info\{([^}]*)\}")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
MEANING = (
    "The stock 'GPU KV cache size' line is max_concurrency x max_model_len: how "
    "many maximum-length requests the pool holds at once, in token units. It is "
    "not the number of cached conversation tokens."
)


def from_logs(text):
    """Take the last boot's lines; a container log may hold one boot only."""
    result = {}
    stock = STOCK.findall(text)
    if stock:
        tokens, length, concurrency = stock[-1]
        result["stock"] = {
            "kv_cache_size_tokens": int(tokens.replace(",", "")),
            "max_model_len": int(length.replace(",", "")),
            "max_concurrency": float(concurrency),
        }
    groups = GROUPS.findall(text)
    if groups:
        result["group_block_sizes"] = [
            int(item) for item in groups[-1].split(",") if item.strip()
        ]
    block = SCHEDULER_BLOCK.findall(text)
    if block:
        result["scheduler_block_size"] = int(block[-1])
    return result


def from_metrics(text):
    """CacheConfig fields as the API server exports them; 'None' stays None."""
    match = CACHE_INFO.search(text)
    if not match:
        return {}
    labels = dict(LABEL.findall(match.group(1)))
    result = {}
    for key, cast in (
        ("num_gpu_blocks", int),
        ("block_size", int),
        ("kv_cache_size_tokens", int),
        ("kv_cache_max_concurrency", float),
    ):
        value = labels.get(key)
        if value is None or value == "None":
            continue
        try:
            result[key] = cast(value)
        except ValueError:
            continue
    return result


def group_cost(group, tokens, retention_interval):
    """Blocks one cached T-token conversation keeps in this group, or a reason.

    Dense retention with block-aligned hits: attention groups keep every block,
    a KDA (mamba) group keeps one checkpoint per block only when retention is
    dense (interval None), and the private IndexPool tail ring publishes nothing.
    """
    kind, block = group["kind"], group["block_size"]
    if kind in ("FullAttentionSpec", "MLAAttentionSpec"):
        return math.ceil(tokens / block)
    if kind == "MambaSpec":
        if retention_interval is None:
            return math.ceil(tokens / block)
        return f"mamba retention interval {retention_interval} is not modelled"
    if "Kpool" in kind or "Tail" in kind:
        return 0
    return f"{kind} is not modelled"


def conversation_estimate(layout, lengths, retention_interval):
    groups = layout["groups"]
    usable = layout["num_blocks"] - 1  # block id 0 is the null block
    rows = []
    for tokens in lengths:
        costs = [group_cost(group, tokens, retention_interval) for group in groups]
        withheld = [cost for cost in costs if isinstance(cost, str)]
        if withheld:
            return {"withheld": sorted(set(withheld))}
        blocks = sum(costs)
        rows.append(
            {
                "tokens": tokens,
                "blocks": blocks,
                "per_group": costs,
                "conversations": usable // blocks if blocks else None,
            }
        )
    return {
        "usable_blocks": usable,
        "assumptions": (
            "nothing running, dense retention, block-aligned hits; a running "
            "request also holds its tail ring and any partial block"
        ),
        "rows": rows,
    }


def summarize(profile, logs, metrics, layout=None):
    """logs: head container log text; metrics: /metrics text; layout: rank 0's
    apc_cache_layout when the profile loads the LPA worker extension."""
    boot = from_logs(logs)
    info = from_metrics(metrics)
    result = {"meaning": MEANING, **boot, "cache_config_info": info}
    concurrency = boot.get("stock", {}).get("max_concurrency") or info.get(
        "kv_cache_max_concurrency"
    )
    num_blocks = info.get("num_gpu_blocks")
    if layout is not None:
        num_blocks = layout["num_blocks"]
    result["num_gpu_blocks"] = num_blocks
    if concurrency:
        result["full_length_requests_that_fit"] = math.floor(concurrency)
        if num_blocks:
            result["blocks_per_max_length_request"] = round(num_blocks / concurrency)
    if layout is None:
        result["cached_conversations"] = {
            "withheld": [
                "group spec kinds are exported by the LPA worker extension only "
                "(lpa.enabled profiles); block widths alone do not say which "
                "groups publish to the prefix cache"
            ]
        }
    else:
        limit = profile["context"]["max_model_len"]
        lengths = sorted(
            {length for length in (16384, 65536, limit) if length <= limit}
        )
        result["cached_conversations"] = conversation_estimate(
            layout, lengths, layout.get("retention_interval")
        )
    return result
