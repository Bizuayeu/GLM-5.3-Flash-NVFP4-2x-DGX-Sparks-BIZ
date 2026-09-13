# One startup configuration

[日本語](startup-configuration.ja.md)

Copy [the commented TOML](../examples/startup.example.toml) to `state/startup.toml` and put the same file on both Linux hosts. This controls the **experimental reference launcher** and its serial chat client. The ordinary `service` command retains its separate, unfinished qualification gate.

| Category | Controls |
|---|---|
| `runtime` | Immutable image IDs, eager/decode Graph execution, independent EP/PP and stage boundary, seed |
| `context` | Total input/output context, active sequences, prefill chunk budget |
| `profiling` | On-demand CUDA/kernel trace collection for a diagnostic run |
| `validation` | Separate CUDA/indexer or expert-placement observer workers |
| `cache` | KV bytes per rank, requested block size, prefix cache, memory utilization, experimental fused unpack |
| `mtp` | Enable MTP, draft depth, checkpoint metadata view |
| `lpa` | Enable approximation, first layer, exact tail, break-even threshold, query omission, projector and checksum |
| `api` | Loopback/rendezvous ports, served name and parsers |
| `generation` | Client defaults: output tokens, temperature, reasoning and timeout |
| `resources` | Container limit, startup/free-memory reserve, total run deadline |
| `nodes` | Both ranks' measured fabric addresses, interfaces, HCAs and GIDs |

The model/revision and build base stay in [runtime.lock.json](../config/runtime.lock.json). Paths are relative to the TOML file; `mtp.view` is relative to the Hugging Face cache, with the pinned revision appended automatically. Keep credentials out of this file.

## Commands

**P22 implementation candidate; GPU and crossover validation pending.** When LPA and prefix caching are both enabled, an image with `GLM53_APC_LPA_API=1` is required. The scheduler chooses from the jointly restored prefix and `lpa.break_even_tokens`; its provisional template value is1024, not a measured recommendation. Shared publication stops at the first approximation and remains stopped through the exact tail/decode. `startup ask` leaves the decision to the server in this mode. Use `"vllm_xargs": {"glm53_lpa_mode": "off"}` in a request to compute normally and grow exact shared cache. Ordinary no-APC `startup ask` uses the same threshold with H=0. See the [implementation contract (Japanese)](apc-lpa-design.md). The new threshold key must be explicit in every updated TOML.

`runtime.expert_parallel=false` is the default. The opt-in adds `--enable-expert-parallel` on both ranks while retaining TP=2/DP=1, the current precision and fixed KV budget. It requires an image with `GLM53_EXPERT_PARALLEL_API=1`; this marker identifies configuration support, not successful EP qualification. Initial scope is eager, one/two sequences and no MTP/LPA/fusion/APC. Existing TOMLs must explicitly include the new key; no silent missing-key fallback is provided. See the [independent EP plan](performance-investigation.md#expert-parallel-p21) before use.

`runtime.pipeline_parallel_size=1` retains TP=2. Setting it to2 selects TP=1/PP=2 on the same two nodes and requires `GLM53_PIPELINE_API=1`. `pipeline_split_layer` sets the first stage's layer count; the default candidate24 produces stages24/21, each with21 MoE layers in this pinned model. It is not a memory-fit guarantee. Both stages must contain MLA, so this checkpoint accepts boundaries4–43. Initial scope is one sequence, eager and no EP/MTP/LPA/fusion/APC. The [small-fixture observations](component-validation.md#eight-layer-pp-observations) do not qualify full-model speed, long-context numerical equivalence or production use. Include both keys explicitly when updating a TOML.

`validation.expert_worker=true` exposes the typed `expert_info` diagnostic for actual placement, kernel and parameter metadata. It supports the independent eager TP2 baseline and EP arms with up to two sequences; other validation workers, MTP/LPA/APC and PP are excluded. Layer hashing begins only after an explicit `pipeline_observe` RPC. Do not install those hooks during performance measurement. This is an experimental local control endpoint, not an enterprise qualification receipt.

`runtime.index_checks` accepts `auto` (the default), `sync` or `async`. Auto preserves synchronous checks in eager execution and selects asynchronous checks for Graphs. Explicit async enables the independently measured eager path; it requires `GLM53_ASYNC_INDEX_CHECK_API=1`. Checks are always performed. Invalid indices in async mode can invalidate the CUDA context, requiring both ranks to restart. Graphs reject explicit sync. MTP/LPA/fusion combinations still need their own acceptance results; the default remains auto.

Before changing context or concurrency, review [KV capacity and RAM requirements](#kv-capacity-and-ram-requirements).

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

## KV capacity and RAM requirements

**Retaining B requests simultaneously at their maximum total length C requires capacity for B×C tokens.** `max_model_len` bounds input plus generated tokens per request; `max_num_seqs` limits concurrency. Setting both does not reserve or qualify that worst-case capacity.

This launcher's `cache.kv_cache_memory_bytes` sets a **fixed KV-pool byte budget shared by requests on each rank**. With a 1 GiB setting, changing concurrency from one to two leaves 1 GiB per rank. It is neither 1 GiB per request nor one freely combined pool across both nodes. Explicit bytes override utilization-based KV sizing; `gpu_memory_utilization` is not a total-RAM safety cap in this mode. [vLLM configuration](https://docs.vllm.ai/en/latest/configuration/engine_args/#kv-cache-memory-bytes)

Live requests consume pool blocks according to retained input and generated tokens. Qualifying maximum-length concurrency requires testing that maximum, including output budgets. GLM combines sparse MLA, IndexPool and per-sequence KDA state: use the pinned runtime's cache specs, block alignment, per-group capacities and state slots rather than applying a generic dense-attention bytes/token formula. Include additional speculative state when enabling MTP.

On each node, weights, KV/cache state, activation/indexer workspaces, MTP/Graph allocations, CPU/OS/other load and operating reserve must fit unified RAM. Fixed KV bytes do not fix every other allocation: context, chunks and concurrency can enlarge other buffers. Container limits and `reserve_gib` protect the host; they do not certify fit or uninterrupted operation.

Insufficient KV can cause startup rejection or runtime waiting, preemption and recomputation. The fixed pool does not automatically expand to meet demand. Other allocations or an insufficient RAM budget can still cause OOM or guard stops. [vLLM preemption](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption)

Retain the runtime's reported capacity/concurrency estimates, then test the intended input-plus-output length × concurrency while observing preemption, both ranks' minimum free memory, OOM and guard stops. A current preflight pass does not replace this maximum-capacity test. See the [independent batching measurements](benchmarks.md#independent-active-batching) for actual coverage.

## Current image contract

Use a freshly built image with `GLM53_LPA_API=2`, `glm53_setup.runtime.lpa.LPAWorkerExtension`, `lpa_configure` / `lpa_report`, and `/lpa/projector.pt`. Preflight rejects images without that marker. There is no old-command or old-image fallback. Rebuild from current source, verify the image ID on both hosts and update the configured image before starting.

## Feature combinations and limits

`runtime.enforce_eager=true` remains the default. Setting it to `false` explicitly selects experimental decode Graphs with `CompilationMode.NONE`, `FULL_DECODE_ONLY` and capture size `[1]`. Prefill is uncompiled; the image must carry `GLM53_DECODE_GRAPH_API=1`. Independent evaluation currently requires one sequence and no LPA/MTP/APC; fusion remains a separate setting. This is not full-model acceptance. Startup rejects combined and multi-sequence Graph configurations until their later qualification.

The Graph path checks internal candidate-index bounds asynchronously on the GPU. Invalid indices cause a device assertion rather than being ignored; unlike a regular Python exception, this can render the CUDA context unusable. Stop and reinitialize both ranks after such a failure. Include retained Graph memory and startup capture time in comparisons. Default eager execution retains synchronous checks.

`validation.component_worker=true` selects an explicitly separate observer/validation worker, requiring an image with `GLM53_COMPONENT_API=1`. It requires eager execution, one sequence, no LPA/MTP and no prefix caching. Its typed RPCs collect bounded indexer observations and toggle exact unpack fusion between exclusive requests for A/B/A tests. This is a diagnostic profile, not an enabled Reuse/Reindex serving mode; use one controlling client.

`cache.fused_unpack=false` keeps the Torch reference conversion. Opting in selects a single Triton kernel for the 656-byte MLA cache record's FP8 latent conversion and FP32 scale multiplication. It requires an image with `GLM53_FUSED_UNPACK_SUPPORTED=1`; preflight rejects other images. This candidate is experimental: component equivalence, attention/state checks and unprofiled full-model A/B measurements are required before adoption. It does not change selected candidates or share KV between layers.

Measured CUDA A/B and indexer observations, including their validation limits, are recorded in [component validation](component-validation.md).

Set `mtp.enabled` and `lpa.enabled` independently. MTP selects the view prepared with [prepare_mtp_view.py](../tools/prepare_mtp_view.py) and BF16 Triton drafting. LPA selects `runtime.lpa_image`, mounts the projector read-only and enables the worker extension. Combined use requires an image with the explicit MTP-aware LPA worker; old LPA images reject it.

LPA needs per-request prompt length. The client tokenizes the actual template, configures LPA, generates, checks token-count agreement and resets LPA to off. Prompts fully covered by `lpa.tail` run normally. Use one controlling client only; a host lock serializes this CLI's requests, but does not coordinate arbitrary direct API clients. This convenience client handles non-streaming text/tool chat. General harness and production acceptance remain separate.

Scope: TP=2, text/tools, Marlin W4A16, FP8 KV. LPA requires one active sequence, eager execution and no prefix caching. No-LPA throughput profiles may use more sequences with separate quality/resource checks; see [performance investigation](performance-investigation.md). MTP depths are limited to 1 and 3. Changing context, chunks, cache sizes, cut or tail needs new workload measurements; a schema-valid setting does not certify quality or resource fit. See [LPA](lpa.md) and [MTP](speculative-decoding.md) for evidence.
