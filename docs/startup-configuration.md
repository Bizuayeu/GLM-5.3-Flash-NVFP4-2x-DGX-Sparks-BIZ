# One startup configuration

[日本語](startup-configuration.ja.md)

Copy [the commented TOML](../examples/startup.example.toml) to `state/startup.toml` and put the same file on both Linux hosts. This controls the **experimental reference launcher** and its serial chat client. The ordinary `service` command retains its separate, unfinished qualification gate.

| Category | Controls |
|---|---|
| `runtime` | Immutable reference/LPA image IDs, eager execution, seed |
| `context` | Total input/output context, active sequences, prefill chunk budget |
| `profiling` | On-demand CUDA/kernel trace collection for a diagnostic run |
| `cache` | KV bytes per rank, requested block size, prefix cache, memory utilization, experimental fused unpack |
| `mtp` | Enable MTP, draft depth, checkpoint metadata view |
| `lpa` | Enable approximation, first layer, exact tail, query omission, projector and checksum |
| `api` | Loopback/rendezvous ports, served name and parsers |
| `generation` | Client defaults: output tokens, temperature, reasoning and timeout |
| `resources` | Container limit, startup/free-memory reserve, total run deadline |
| `nodes` | Both ranks' measured fabric addresses, interfaces, HCAs and GIDs |

The model/revision and build base stay in [runtime.lock.json](../config/runtime.lock.json). Paths are relative to the TOML file; `mtp.view` is relative to the Hugging Face cache, with the pinned revision appended automatically. Keep credentials out of this file.

## Commands

Inspect generated commands from the checkout root (also works on Windows):

```sh
python -m glm53_setup startup plan --rank 0
python -m glm53_setup startup plan --rank 1
```

On each Linux host, `startup preflight --rank N` checks assets, fabric, image identity and available memory. Start rank 1 first, then rank 0, in separate terminals on their respective hosts:

```sh
python -m glm53_setup startup start --rank 1 --experimental
python -m glm53_setup startup start --rank 0 --experimental
```

The commands stay in the foreground supervising their own containers; keep the terminals running. `resources.run_seconds` **includes model loading**. Increase it before a longer session. Ctrl+C, deadline or low memory stops that rank. Stop both ranks after a distributed failure. Containers and `records/` logs/configuration snapshots are retained; no automatic deletion or restart occurs.

When the API is ready, from another head terminal:

```sh
python -m glm53_setup startup ask --prompt "Reply with exactly GLM-OK."
python -m glm53_setup startup ask --request request.json
python -m glm53_setup startup status --rank 0
python -m glm53_setup startup stop --rank 0
```

Use `--rank 1` on the worker to inspect/stop it. All actions accept `--config path/to/settings.toml`. Restart both ranks after editing settings; the client refuses a profile different from the running container's fingerprint. Generation defaults apply to `startup ask`; external clients supply their own request options.

## Runtime limit and continuous operation

`resources.run_seconds` controls the automatic time limit in seconds, including loading. Set it to `0` for **no time limit**; positive integers retain a bounded session. Negative values are rejected. The `resources.reserve_gib` memory protection remains active in either mode. Settings are read at launch, so editing the file does not change an already running supervisor.

```toml
[resources]
# Change this entry in the existing resources section:
run_seconds = 0
```

This enables a run without a scheduled stop, not a 24/7 availability guarantee. Automatic host-startup integration, coordinated two-rank recovery and redundant failover remain unimplemented. The foreground supervisor must remain alive; production/harness qualification is still pending.

## Current image contract

Use a freshly built image with `GLM53_LPA_API=2`, `glm53_setup.runtime.lpa.LPAWorkerExtension`, `lpa_configure` / `lpa_report`, and `/lpa/projector.pt`. Preflight rejects images without that marker. There is no old-command or old-image fallback. Rebuild from current source, verify the image ID on both hosts and update the configured image before starting.

## Feature combinations and limits

`cache.fused_unpack=false` keeps the Torch reference conversion. Opting in selects a single Triton kernel for the 656-byte MLA cache record's FP8 latent conversion and FP32 scale multiplication. It requires an image with `GLM53_FUSED_UNPACK_SUPPORTED=1`; preflight rejects other images. This candidate is experimental: component equivalence, attention/state checks and unprofiled full-model A/B measurements are required before adoption. It does not change selected candidates or share KV between layers.

Set `mtp.enabled` and `lpa.enabled` independently. MTP selects the view prepared with [prepare_mtp_view.py](../tools/prepare_mtp_view.py) and BF16 Triton drafting. LPA selects `runtime.lpa_image`, mounts the projector read-only and enables the worker extension. Combined use requires an image with the explicit MTP-aware LPA worker; old LPA images reject it.

LPA needs per-request prompt length. The client tokenizes the actual template, configures LPA, generates, checks token-count agreement and resets LPA to off. Prompts fully covered by `lpa.tail` run normally. Use one controlling client only; a host lock serializes this CLI's requests, but does not coordinate arbitrary direct API clients. This convenience client handles non-streaming text/tool chat. General harness and production acceptance remain separate.

Scope: TP=2, text/tools, Marlin W4A16, FP8 KV. LPA requires one active sequence, eager execution and no prefix caching. No-LPA throughput profiles may use more sequences with separate quality/resource checks; see [performance investigation](performance-investigation.md). MTP depths are limited to 1 and 3. Changing context, chunks, cache sizes, cut or tail needs new workload measurements; a schema-valid setting does not certify quality or resource fit. See [LPA](lpa.md) and [MTP](speculative-decoding.md) for evidence.
