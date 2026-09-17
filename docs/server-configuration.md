# One server configuration

[日本語](server-configuration.ja.md)

Copy [the commented TOML](../examples/server.example.toml) to `state/server.toml` and put the same file on both Linux hosts. This controls the launcher and its serial chat client.

| Category | Controls |
|---|---|
| `runtime` | Immutable image IDs, eager/decode Graph execution, independent EP/PP and stage boundary, seed, image-input switch |
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

## Distributed defaults

The distributed TOML selects the serial optimized profile with [image input at 256K](vision.md). This is a configuration choice, not production or harness qualification. Existing `state/server.toml` files are not updated automatically.

| Item | Default |
|---|---|
| Execution | TP=2, eager, one sequence, 262,144 tokens, chunk 2048 ([measured](benchmarks.md#chunk-budget-on-the-200k-image-profile-2026-09-17)) |
| Input | Text, tool calls and images (`runtime.vision = true`); video rejected |
| Cache | FP8, 3 GiB per rank, APC on, `dense` checkpoint retention, fused unpack on, image preprocessing cache 0.1 GiB |
| Speculation/approximation | MTP k=3; LPA off (cut32/tail512/B128 with unused MLA queries skipped when enabled) |
| Checks/parallelism | Async index checks, EP off, no PP split |
| NCCL | `nccl_channels = 8` on both ranks (NCCL alone chooses 64 on the reference pair) |
| MoE token order | `canonical_moe_order = true`: one token order inside each expert, so identical requests repeat; needs a reference image built from this version |
| Generation | temperature=0, max_tokens=4096, reasoning_effort=low, clear_thinking=true |
| Resources | Container 112 GiB, startup free 108 GiB, runtime reserve 3 GiB |
| Lifetime | `run_seconds=0`: no time-based automatic stop; memory supervision remains active |
| Supervision | `stall_seconds=600`: rank 0 also stops when requests are running but no `/metrics` signal moves for 600 s (`engine-stall`); `api.dev_endpoints=false` |
| Warmup | `warmup=true`, `warmup_long_tokens=0`: text, tool and image rungs after readiness; no long rung until set |

The text-only alternative sets `runtime.vision = false` and keeps the length and KV above. It loads no vision tower and keeps no image preprocessing cache, and stays available for text-only serving and for checks with less memory headroom. Its [256K checks](benchmarks.md#real-input-checks-at-256k) ran on 2026-09-14 with a 4 GiB reserve at chunk 512; the template's 3 GiB reserve at chunk 2048 is not validated without images.

**Supply image IDs, both nodes' connection details and the MTP view before launch. Supply the LPA projector/hash only when enabling LPA.** Zero hashes are placeholders to replace for enabled features; missing assets never silently disable features. The [trained projector download](lpa.md#download-the-trained-projector) avoids retraining; [operations](operations.md#artifact-storage-and-paths) owns asset placement. MTP/LPA can be disabled separately; baseline comparisons also explicitly reset APC, retention, fusion and async checks.

The lifetime is fixed at launch. Apply a changed `run_seconds` to running supervisors by restarting through the [two-rank switch procedure](launch-safety.md#all-rail-checks-and-two-rank-switch). Editing the TOML alone does not cancel the existing deadline. Larger contexts require separate capacity checks and real-request validation below.

## Commands

See [launch contracts and operational validation](launch-safety.md) for authenticated clients, allocator unset/empty handling, all-HCA checks and the two-rank pre-stop/switch procedure. These extend P10/P19/P22/E03; implementation does not establish multi-rail traffic or full operational qualification.

**P22 has scoped GPU isolation, crossover, combined and final held-out evidence.** When LPA and prefix caching are both enabled, an image with `GLM53_APC_LPA_API=1` is required. The scheduler chooses from the jointly restored prefix and `lpa.break_even_tokens`; the template uses the conservative measured cutoff 128 from the [P22 calibration](benchmarks.md#apc-first-lpa-crossover-measurement-p22). Shared publication stops at the first approximation and remains stopped through the exact tail/decode. `server ask` leaves the decision to the server in this mode. The template ships `lpa.enabled = false`; enable it per workload for batch inputs only, since an approximated request publishes nothing to the shared cache. While it is enabled, use `"vllm_xargs": {"glm53_lpa_mode": "off"}` in a request to compute normally and grow exact shared cache. `api.prompt_tokens_details = true` (optional key, default off when absent) adds `--enable-prompt-tokens-details` so `usage.prompt_tokens_details.cached_tokens` reports the restored prefix; without it vLLM returns `null` and harness cache displays stay at zero even when the cache hits. Ordinary no-APC `server ask` uses the same threshold with H=0. See the [implementation contract](apc-lpa-design.md). The new threshold key must be explicit in every updated TOML.

`runtime.expert_parallel=false` is the default. The opt-in adds `--enable-expert-parallel` on both ranks while retaining TP=2/DP=1, the current precision and fixed KV budget. It requires an image with `GLM53_EXPERT_PARALLEL_API=1`; this marker identifies configuration support, not successful EP qualification. Initial scope is eager, one/two sequences and no MTP/LPA/fusion/APC. Existing TOMLs must explicitly include the new key; no silent missing-key fallback is provided. See the [independent EP plan](performance-investigation.md#expert-parallel-p21) before use.

`runtime.vision` is optional and false when absent; the template sets `true`. `false` keeps `--language-model-only`, so the vision tower is not loaded and requests stay text/tools only. `true` removes that flag on both ranks and adds `--limit-mm-per-prompt '{"video": 0}'`. **Video input is disabled and rejected even with `vision = true`; only images are accepted.** The reason is startup memory: vLLM profiles by encoding the largest item once, and this checkpoint's video budget (capped at 30,000 tokens, 120,000 patches) is far larger than one image (up to 8,000 tokens). The image count per prompt keeps the vLLM default, because chat harnesses resend earlier images every turn. With vision on, `cache.mm_processor_cache_gb` (optional, 0.1 when absent) sets `--mm-processor-cache-gb`, the preprocessed-image cache that vLLM keeps in both the API and engine processes, which run on rank 0 only; the vLLM default of 4 GiB could take up to 8 GiB of host RAM on the head. A processed image larger than the budget is served uncached (with a warning) rather than rejected. The vision tower is BF16 and excluded from quantization (`model.visual*` in both exclusion lists, and the pinned vLLM builds it with no quantization config). Measurements, the head's memory margin and open items are in [image input at 200K](vision.md). Validation fixtures still load text-only regardless of this key.

`api.dev_endpoints` (optional, false when absent) sets `VLLM_SERVER_DEV_MODE=1` on both ranks, which mounts vLLM's dev routes on the loopback API: `/reset_prefix_cache`, `/reset_mm_cache`, `/collective_rpc`, `/sleep`, `/wake_up` and `/server_info`. LPA, component and expert profiles already run in that mode; the key exists so that a distributed profile (LPA off) can reset the prefix cache without a restart for benchmarks and for the warmup ladder's cleanup. The routes have no authentication ([launch contracts](launch-safety.md#model-api-clients)); leave it off on a kit with other local users.

`resources.stall_seconds` (optional, 0 = off when absent; template 600) and `generation.warmup` / `generation.warmup_long_tokens` (optional, false / 0 when absent) are described under [supervision, stall detection and warmup](operations.md#supervision-stall-detection-and-warmup). Changing any of them changes the profile fingerprint, so they take effect at the next switch.

`runtime.nccl_channels` (optional; absent = NCCL chooses; template 8) sets `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` to the same positive integer on both ranks. On the reference pair NCCL 2.30.7 chooses 64 channels by itself. Each rank's engine opens two communicators, and at 8 channels and MTU 1500 the full model's lowest free memory rose by 2.8 GiB on the head and 3.0 GiB on the peer compared with 64, while prefill did not slow (it was 1% faster); decode varied more between runs than between settings. The [channel-count measurements](nccl-validation.md#channel-count) hold the numbers. A profile written before 1.3.1 has no key, keeps NCCL's choice and its fingerprint; add the key to adopt the template value. Informed by Mia PR #200. Changing it changes the profile fingerprint, so it takes effect at the next switch.

`runtime.canonical_moe_order` (optional; template `true`; absent = the image's default, which is on in images built from this version) sets `GLM53_CANONICAL_MOE_ORDER` on both ranks. The pinned vLLM's `moe_align_block_size` orders the tokens inside an expert by CUDA thread scheduling, the Marlin MoE result depends slightly on that order, and later routers amplify it, so identical requests did not repeat ([what was measured](validation.md#full-model-tp2-experimental-scope); upstream vLLM issue #52525). With the key `true` the reference image sorts each expert's slots by token id before the kernel; `server preflight` then requires `GLM53_MOE_ORDER_API=1` in the image, so a profile cannot ask an older image for it. `false` is the comparison arm and needs no image support. Expert parallelism is left untouched. On the eight-layer fixture the fix made every repeat bit-identical, left prefill unchanged and slowed decode by about 1%; it has not been measured on the full model. It is on by default because a reproducible baseline is what a later A/B of graphs or requantization is read against.

`runtime.derived_checkpoint` (optional table, absent by default; a profile without it keeps its fingerprint) serves a locally requantized copy of the pinned checkpoint for an A/B. It takes `path` (absolute directory on both hosts, mounted read-only at `/derived`), `requant_target` (compared with `quantization_config.producer.requant_target` of that directory) and `overlays`, a list of `{target, source, sha256, base_sha256, marker}`: `source` is an absolute file mounted over `target` in the image's GLM model directory. `server preflight` fails unless the checkpoint declares `MIXED_PRECISION` with that target, no quantized module is declared on the MTP draft layer, each overlay file has the given SHA-256 and contains its marker, and the image's own `target` has `base_sha256`, so an overlay built for another image cannot be mounted. The MTP metadata view is not used with a derived checkpoint: under `MIXED_PRECISION` undeclared modules, the BF16 draft layer among them, load unquantized. The template has no such table. It was used once on the reference pair for the P23 comparison recorded in the [optimization catalog](optimization-catalog.md).

`runtime.pipeline_parallel_size=1` retains TP=2. Setting it to 2 selects TP=1/PP=2 on the same two nodes and requires `GLM53_PIPELINE_API=1`. `pipeline_split_layer` sets the first stage's layer count; the default candidate 24 produces stages 24/21, each with 21 MoE layers in this pinned model. It is not a memory-fit guarantee. Both stages must contain MLA, so this checkpoint accepts boundaries 4–43. Initial scope is one sequence, eager and no EP/MTP/LPA/fusion/APC. The [small-fixture observations](component-validation.md#eight-layer-pp-observations) do not qualify full-model speed, long-context numerical equivalence or production use. Include both keys explicitly when updating a TOML.

`validation.expert_worker=true` exposes the typed `expert_info` diagnostic for actual placement, kernel and parameter metadata. It supports the independent eager TP2 baseline and EP arms with up to two sequences; other validation workers, MTP/LPA/APC and PP are excluded. Layer hashing begins only after an explicit `pipeline_observe` RPC. Do not install those hooks during performance measurement. This is an experimental local control endpoint, not a business-use qualification receipt.

`runtime.index_checks` accepts `auto`, `sync` or `async` (the distribution default). Auto preserves synchronous checks in eager execution and selects asynchronous checks for Graphs. Explicit async enables the independently measured eager path; it requires `GLM53_ASYNC_INDEX_CHECK_API=1`. Checks are always performed. Invalid indices in async mode can invalidate the CUDA context, requiring both ranks to restart. Graphs reject explicit sync. The serial MTP/LPA/fusion combination has scoped P18/P22 evidence.

Before changing context or concurrency, review [KV capacity and RAM requirements](#kv-capacity-and-ram-requirements).

Inspect generated commands from the checkout root (also works on Windows):

```sh
python -m glm53_setup server plan --rank 0
python -m glm53_setup server plan --rank 1
```

On each Linux host, `server preflight --rank N` checks assets, fabric, image identity, that no other container holds the GPU, and available memory ([what each check covers](operations.md#full-model-launch-checks)). Start rank 1 first, then rank 0, in separate terminals on their respective hosts:

```sh
python -m glm53_setup server start --rank 1
python -m glm53_setup server start --rank 0
```

The commands stay in the foreground supervising their own containers; keep the terminals running. `resources.run_seconds` **includes model loading**. Increase it before a longer session. Ctrl+C, deadline or low memory stops that rank. Stop both ranks after a distributed failure. Containers and `records/` logs/configuration snapshots are retained; no automatic deletion or restart occurs.

When the API is ready, from another head terminal:

```sh
python -m glm53_setup server ask --prompt "Reply with exactly GLM-OK."
python -m glm53_setup server ask --request request.json
python -m glm53_setup server status --rank 0
python -m glm53_setup server capacity
python -m glm53_setup server warmup
python -m glm53_setup server mojibake
python -m glm53_setup server agreement
python -m glm53_setup server stop --rank 0
```

`agreement` sends four self-authored texts (Japanese, English, code, mathematics) through `/v1/completions` with `prompt_logprobs` under the request lock, twice each, and writes the rank and log-probability of every actual next token to `records/<stamp>-agreement-r0/result.json`; with `--reference <result.json>` it adds the argmax agreement, top-5 overlap and log-probability drift against that earlier run. Identical requests do not repeat exactly on the served full model: on the reference pair the argmax agreed on about 96% of positions between two passes over the same text, and one log-probability moved by 8 nats, while the four-layer fixture repeats bit-identically. On the fixture the source is the token order inside each expert, which changes from call to call ([validation](validation.md#full-model-tp2-experimental-scope)); `runtime.canonical_moe_order` fixes it in images built from this version. The record therefore passes when every request was answered in the checked shape, and `self_agreement` carries that difference as the yardstick; the per-text mean negative log-probability moved far less between runs (0.003–0.05) and is the steadier reading.

`capacity` reads the running head's boot log and `/metrics` and prints the KV pool as it is: the stock `GPU KV cache size` line decomposed into `num_gpu_blocks`, blocks per maximum-length request and the group block widths, with the note that the stock figure is `max_concurrency × max_model_len`. A cached-conversation estimate (blocks and conversations at 16K, 64K and `max_model_len`, dense retention, block-aligned hits, nothing running) is printed only for a profile that loads the LPA worker extension, whose `apc_cache_layout` RPC names each group's spec kind; otherwise, or when a group kind is not modelled, that figure is withheld rather than guessed. `warmup` runs the request ladder under the request lock and writes `records/<stamp>-warmup-r0/result.json`; it exits nonzero when a rung failed. `mojibake` asks the running head for long Japanese and Korean answers under the same lock, counts broken characters in the answers and the reasoning, writes `records/<stamp>-mojibake-r0/result.json` and exits nonzero unless every answer passed ([what it checks](validation.md#full-model-tp2-experimental-scope)).

Use `--rank 1` on the worker to inspect/stop it. All actions accept `--config path/to/settings.toml`. Restart both ranks after editing settings; the client refuses a profile different from the running container's fingerprint. Generation defaults apply to `server ask`; external clients supply their own request options.

## Runtime limit and continuous operation

`resources.run_seconds` controls the automatic time limit in seconds, including loading. Set it to `0` for **no time limit**; positive integers retain a bounded session. Negative values are rejected. The `resources.reserve_gib` memory protection remains active in either mode. Settings are read at launch, so editing the file does not change an already running supervisor.

```toml
[resources]
# Change this entry in the existing resources section:
run_seconds = 0
```

This enables a run without a scheduled stop, not a 24/7 availability guarantee. Automatic host-startup integration, coordinated two-rank recovery and redundant failover remain unimplemented. The foreground supervisor must remain alive; production/harness qualification is still pending.

## KV capacity and RAM requirements

**Retaining B requests simultaneously at their maximum total length C requires capacity for B×C tokens.** `max_model_len` bounds input plus generated tokens per request; `max_num_seqs` limits concurrency. Setting both does not reserve or qualify that worst-case capacity. The accepted two-sequence scope covers up to 2,112 tokens per request; another two-Spark recipe reports two concurrent 25–100K requests falling to about 4 tok/s combined (tonyd2wild #14, no code adopted).

This launcher's `cache.kv_cache_memory_bytes` sets a **fixed KV-pool byte budget shared by requests on each rank**. With a 1 GiB setting, changing concurrency from one to two leaves 1 GiB per rank. It is neither 1 GiB per request nor one freely combined pool across both nodes. Explicit bytes override utilization-based KV sizing; `gpu_memory_utilization` is not a total-RAM safety cap in this mode. [vLLM configuration](https://docs.vllm.ai/en/latest/configuration/engine_args/#kv-cache-memory-bytes)

Live requests consume pool blocks according to retained input and generated tokens. Qualifying maximum-length concurrency requires testing that maximum, including output budgets. GLM combines sparse MLA, IndexPool and per-sequence KDA state: use the pinned runtime's cache specs, block alignment, per-group capacities and state slots rather than applying a generic dense-attention bytes/token formula. Include additional speculative state when enabling MTP.

On each node, weights, KV/cache state, activation/indexer workspaces, MTP/Graph allocations, CPU/OS/other load and operating reserve must fit unified RAM. Fixed KV bytes do not fix every other allocation: context, chunks and concurrency can enlarge other buffers. Container limits and `reserve_gib` protect the host; they do not certify fit or uninterrupted operation. The earlier 256K text-only profile ran close to that guard on the measured 121 GiB hosts: available memory sat near 4.5 GiB and two supervised stops (`stop-reason: memory-reserve`) occurred at a reserve of 4, the second while serving one 16,859-token approximated request; the reserve then moved to 3. The image profile kept about 4.0–4.2 GiB available on the head while serving with NCCL's 64 channels ([measurements](vision.md#memory-final-profile)); with 8 channels the head stayed at 6.97 GiB or more through the [200K checks on 1.3.1](benchmarks.md#200k-real-input-on-131) and at 6.40 GiB or more with chunk 2048. At 256K with 3 GiB KV it stayed at 5.82 GiB or more ([measurements on 1.5.0](benchmarks.md#measurements-on-150)), and the template reserves 3. The value may be fractional. Before changing it, account for what the guard can catch: the supervisor samples `MemAvailable` every 2 seconds, and one supervised stop took about 9 seconds from the breaching sample to container exit, so a slide of 0.15 GiB/s (measured during a runtime compile burst) can carry the host about 1.6 GiB below the reserve. The reference hosts recorded no kernel or container OOM kill and have 16 GiB of swap for host pages, so the risk below the reserve is a GPU allocation failing inside a worker, which ends the engine abruptly instead of through a supervised stop. Driver `NV_ERR_NO_MEMORY` kernel messages appear with 4 GiB or more available, mostly during load, and do not mark the floor.

Insufficient KV can cause startup rejection or runtime waiting, preemption and recomputation. The fixed pool does not automatically expand to meet demand. Other allocations or an insufficient RAM budget can still cause OOM or guard stops. [vLLM preemption](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption)

The boot line `GPU KV cache size: N tokens, Maximum concurrency for L tokens per request: Cx` is `N = C × L` for this hybrid model (MLA, IndexPool tail, KDA state groups and the MTP draft share one block pool with one id per group per aligned segment). `N` is therefore a concurrency figure in token units, not the number of conversation tokens the prefix cache can hold; `server capacity` prints the decomposition and, where the group kinds are known, the conversation estimate. In both measured image-profile configurations a GiB of KV held 28 blocks of 4,608 tokens and a full-length request of L tokens took ceil(L / 4608) + 16 of them: 61 of 70 at 204,800 tokens with 2.5 GiB and 73 of 84 at 262,144 with 3 GiB, 1.15× both times. The 256K figures were predicted this way before the switch; other lengths, KV sizes or group layouts need their own boot line. Prefix caching also works in whole blocks, so a repeated N-token prompt restores `(floor(N / block) - 1) x block` tokens and nothing at all below two blocks. Measured on the image profile's 4,608-token scheduler block: 3,625 tokens restored none, 14,025 restored 9,216, and 28,025 restored 23,040. Short conversations get no reuse on this profile whatever the hit rate suggests. Retain the runtime's reported capacity/concurrency estimates, then test the intended input-plus-output length × concurrency while observing preemption, both ranks' minimum free memory, OOM and guard stops. A current preflight pass does not replace this maximum-capacity test. See the [independent batching measurements](benchmarks.md#independent-active-batching) for actual coverage.

## Current image contract

Use a freshly built image with `GLM53_LPA_API=2`, `glm53_setup.runtime.lpa.LPAWorkerExtension`, `lpa_configure` / `lpa_report`, and `/lpa/projector.pt`. Preflight rejects images without that marker. There is no old-command or old-image fallback. Rebuild from current source, verify the image ID on both hosts and update the configured image before starting.

## Feature combinations and limits

`runtime.enforce_eager=true` remains the default. Setting it to `false` explicitly selects experimental decode Graphs with `CompilationMode.NONE`, `FULL_DECODE_ONLY` and capture size `[1]`. Prefill is uncompiled; the image must carry `GLM53_DECODE_GRAPH_API=1`. Independent evaluation currently requires one sequence and no LPA/MTP/APC; fusion remains a separate setting. This is not full-model acceptance. Startup rejects combined and multi-sequence Graph configurations until their later qualification.

The Graph path checks internal candidate-index bounds asynchronously on the GPU. Invalid indices cause a device assertion rather than being ignored; unlike a regular Python exception, this can render the CUDA context unusable. Stop and reinitialize both ranks after such a failure. Include retained Graph memory and startup capture time in comparisons. [vLLM #53366](https://github.com/vllm-project/vllm/issues/53366) reports that the compilation-cache hash omits the speculative token count; if compiled Graphs are ever qualified together with MTP, keep a separate cache per k or clear it when k changes. The opt-in above compiles nothing and rejects MTP. Eager checks also follow `runtime.index_checks`; the distribution default is async.

`validation.component_worker=true` selects an explicitly separate observer/validation worker, requiring an image with `GLM53_COMPONENT_API=1`. It requires eager execution, one sequence, no LPA/MTP and no prefix caching. Its typed RPCs collect bounded indexer observations and toggle exact unpack fusion between exclusive requests for A/B/A tests. This is a diagnostic profile, not an enabled Reuse/Reindex serving mode; use one controlling client.

`cache.fused_unpack=true` is the distribution default; false keeps the Torch reference conversion. True selects a single Triton kernel for the 656-byte MLA cache record's FP8 latent conversion and FP32 scale multiplication. It requires an image with `GLM53_FUSED_UNPACK_SUPPORTED=1`; preflight rejects other images. Use is scoped to the component, full-model A/B and serial integration evidence; broad quality and production acceptance remain separate. It does not change selected candidates or share KV between layers.

Measured CUDA A/B and indexer observations, including their validation limits, are recorded in [component validation](component-validation.md).

Set `mtp.enabled` and `lpa.enabled` independently. MTP selects the view prepared with [prepare_mtp_view.py](../tools/prepare_mtp_view.py) and BF16 Triton drafting. LPA selects `runtime.lpa_image`, mounts the projector read-only and enables the worker extension. Combined use requires an image with the explicit MTP-aware LPA worker; old LPA images reject it.

LPA needs per-request prompt length. The client tokenizes the actual template, configures LPA, generates, checks token-count agreement and resets LPA to off. Prompts fully covered by `lpa.tail` run normally. Use one controlling client only; a host lock serializes this CLI's requests, but does not coordinate arbitrary direct API clients. This convenience client handles non-streaming text/tool chat. General harness and production acceptance remain separate.

Scope: TP=2, text/tools, Marlin W4A16, FP8 KV. LPA requires one active sequence and eager execution; prefix caching uses the separate P22 path described above. No-LPA throughput profiles may use more sequences with separate quality/resource checks; see [performance investigation](performance-investigation.md). MTP depths are limited to 1 and 3. Changing context, chunks, cache sizes, cut or tail needs new workload measurements; a schema-valid setting does not certify quality or resource fit. See [LPA](lpa.md) and [MTP](speculative-decoding.md) for evidence.
