# One server configuration

[日本語](server-configuration.ja.md)

Copy [the commented TOML](../examples/server.example.toml) to `state/server.toml` and put the same file on both Linux hosts. This controls the launcher and its serial chat client. The published option has its own example, [server.axl.example.toml](../examples/server.axl.example.toml): the same profile with the `runtime.derived_checkpoint` table for the repacked weights (NVFP4 BIZ AXL) and the shipped overlays, `runtime.prefix_page_dedup`, two active sequences and 6 GiB of KV per rank. The launcher refuses more than 3 GiB of KV without the derived checkpoint: on the reference pair the pinned weights leave the head 5.5 GiB at 3 GiB of KV against a 3 GiB reserve, the repacked ones 10.5 GiB.

| Category | Controls |
|---|---|
| `runtime` | Immutable image IDs, eager/decode Graph execution, independent EP/PP and stage boundary, seed, image-input switch, repeatability switches, derived checkpoint |
| `context` | Total input/output context, active sequences, prefill chunk budget |
| `profiling` | On-demand Torch/CUDA trace collection for a diagnostic run; off for timing measurements |
| `validation` | Separate CUDA/indexer or expert-placement observer workers, memory probe |
| `cache` | KV bytes per rank, requested block size, prefix cache, checkpoint retention, memory utilization, fused unpack |
| `mtp` | Enable MTP, draft depth, checkpoint metadata view |
| `lpa` | Enable approximation, first layer, exact tail, break-even threshold, query omission, projector and checksum |
| `api` | Loopback/rendezvous ports, served name and parsers, dev routes, cached-token usage |
| `generation` | Client defaults: output tokens, temperature, reasoning and timeout; warmup ladder |
| `resources` | Container limit, startup/free-memory reserve, total run deadline, stall detection |
| `nodes` | Both ranks' measured fabric addresses, interfaces, HCAs and GIDs; optional per-host Docker CPU set |

The model/revision and build base stay in [runtime.lock.json](../config/runtime.lock.json). Paths are relative to the TOML file; `mtp.view` is relative to the Hugging Face cache, with the pinned revision appended automatically. Keep credentials out of this file.

## Distributed defaults

The distributed TOML selects the serial optimized profile with [image input at 256K](vision.md). This is a configuration choice; routine-use acceptance is recorded separately in [SETUP step 6](../SETUP.md#6-qualify-the-full-model). Existing `state/server.toml` files are not updated automatically.

| Item | Default |
|---|---|
| Execution | TP=2, eager, one sequence, 262,144 tokens, chunk 2048 ([measured](benchmarks.md#chunk-budget-on-the-200k-image-profile-2026-09-17)) |
| Input | Text, tool calls and images (`runtime.vision = true`); video rejected |
| Cache | FP8, 3 GiB per rank, APC on, `dense` checkpoint retention, fused unpack on, image preprocessing cache 0.1 GiB |
| Prefill attention | `fa2_attention = true`: NoPE attention calls of more than six query rows through FlashInfer FA2 (prefill, and decode steps shared by two sequences), one sequence's decode on the reference path; excludes LPA |
| Speculation/approximation | MTP k=3 ([depths one to five](speculative-decoding.md#depths-one-to-five-2026-09-19-and-20)); LPA off (cut32/tail512/B128 with unused MLA queries skipped when enabled) |
| Checks/parallelism | Async index checks, EP off, no PP split |
| NCCL | `nccl_channels = 8` on both ranks (NCCL alone chooses 64 on the reference pair) |
| Repeatability | `canonical_moe_order`, `stable_indexer_topk` and `inductor_deterministic` all `true`: identical requests repeat bit for bit and every launch computes in the same numerical state ([repeatability switches](#repeatability-switches)) |
| Generation | temperature=0, max_tokens=4096, reasoning_effort=low, clear_thinking=true |
| Resources | Container 112 GiB, startup free 108 GiB, runtime reserve 3 GiB |
| Lifetime | `run_seconds=0`: no time-based automatic stop; memory supervision remains active |
| Supervision | `stall_seconds=600`: rank 0 also stops when requests are running but no `/metrics` signal moves for 600 s (`engine-stall`); `api.dev_endpoints=false` |
| Warmup | `warmup=true`, `warmup_long_tokens=0`: text, tool and image rungs after readiness; no long rung until set |

The text-only alternative sets `runtime.vision = false` and keeps the length and KV above. It loads no vision tower and keeps no image preprocessing cache, and stays available for text-only serving and for checks with less memory headroom. Its [256K checks](benchmarks.md#real-input-checks-at-256k) ran on 2026-09-14 with a 4 GiB reserve at chunk 512; the template's 3 GiB reserve at chunk 2048 is not validated without images.

**Supply image IDs, both nodes' connection details and the MTP view before launch. Supply the LPA projector/hash only when enabling LPA.** Zero hashes are placeholders to replace for enabled features; missing assets never silently disable features. The [trained projector download](lpa.md#download-the-trained-projector) avoids retraining; [operations](operations.md#artifact-storage-and-paths) owns asset placement. MTP/LPA can be disabled separately; baseline comparisons also explicitly reset APC, retention, fusion and async checks.

The lifetime is fixed at launch. Apply a changed `run_seconds` to running supervisors by restarting through the [two-rank switch procedure](launch-safety.md#all-rail-checks-and-two-rank-switch). Editing the TOML alone does not cancel the existing deadline. Larger contexts require separate capacity checks and real-request validation ([KV capacity](#kv-capacity-and-ram-requirements)).

### Optional CPU placement

`nodes[].cpuset_cpus` optionally passes a Docker CPU list (for example, `"5-9,15-19"`) to `--cpuset-cpus` for that rank. Omit it to retain Docker's existing CPU placement. Determine the performance-core IDs separately on each host from its topology and a controlled workload; core numbering is not portable between hosts. The launcher rejects malformed, reversed or overlapping ranges. On each host, `server preflight` checks that the requested CPUs are in the launcher's available affinity mask. After `docker run`, the launcher reads back `HostConfig.CpusetCpus` before recording the rank as started and stops the new container if this check fails. This is a placement control, not a throughput guarantee. Set it on both ranks: on the reference pair (performance cores 5–9 and 15–19 on both hosts) either rank on efficiency cores cut decode to about a third, and the unpinned scheduler happened to keep both on performance cores. The reference pair has served with `5-9,15-19` on both ranks since the switch to 1.18.0, where the preflight check and the read-back held ([measurements on 1.15.0](benchmarks.md#measurements-on-1150)).

## The published option against the defaults

The two examples are one profile with five settings changed; [`tests/test_axl_example.py`](../tests/test_axl_example.py) assembles both docker commands and both environments for each rank with the same image supplied and keeps the difference to these rows and their two consequences.

| Setting in the AXL example | What it adds to the launch |
|---|---|
| `runtime.derived_checkpoint` (`path`, `requant_target = "l"`) | one mount, the repacked checkpoint at `/derived`, read-only |
| `runtime.derived_checkpoint.overlays[0]` (`kda-quant-split.py`) | one mount over the image's `kda.py`, read-only; its hash and the base's are checked at preflight |
| `runtime.derived_checkpoint.overlays[1]` (`mla-quant-split.py`) | one mount over the image's `model.py`, read-only; same check |
| `runtime.prefix_page_dedup = true` | one environment variable, `GLM53_PREFIX_PAGE_DEDUP=1` |
| `context.max_num_seqs = 2` | `--max-num-seqs 2` instead of `1` |
| `cache.kv_cache_memory_bytes = 6442450944` | `--kv-cache-memory-bytes` 6 GiB instead of 3 GiB |

Two arguments follow from the first row and are not settings of their own: the model argument becomes `/derived` instead of the MTP metadata view under `/hf` (the repacked checkpoint declares the BF16 draft layer itself), and the container label carries the profile's fingerprint, which changes with any key. Nothing else in the command or the environment differs, on either rank.

On 2026-09-23 the AXL example, its placeholders replaced by the reference pair's values, passed `server freeze`, `server plan` and every `server preflight` check on both ranks except `startup_memory`, which cannot hold beside a running pair. It differs from the profile the reference pair serves only in `validation.memory_probe` and `api.dev_endpoints` (on in the served profile, for the checks after a switch), the warmup's long rung (65,536 tokens there, none in the example) and the unused LPA placeholders.

## Key reference

Each optional key below keeps the launch unchanged when absent unless stated otherwise. Adding or changing a key changes the profile's fingerprint, so it takes effect at the next [switch](launch-safety.md#all-rail-checks-and-two-rank-switch). A key that needs image support is refused by `server preflight` on an image without the marker listed in the [image contract](#current-image-contract).

### Repeatability switches

Three switches make identical requests repeat bit for bit and every launch compute in the same numerical state; all three are on in both examples. How each source of difference was found and measured is in [validation](validation.md#full-model-tp2-experimental-scope).

`runtime.canonical_moe_order` (template `true`; absent = the image's default, on in images built from 1.6.0) sets `GLM53_CANONICAL_MOE_ORDER` on both ranks. The pinned vLLM's `moe_align_block_size` orders the tokens inside an expert by CUDA thread scheduling, the Marlin MoE result depends slightly on that order, and later routers amplify it, so identical requests did not repeat (upstream vLLM issue #52525). With `true` the reference image sorts each expert's slots by token id before the kernel. New launches need `GLM53_MOE_ORDER_API=2` (images built from 1.7.0); marker 1 is accepted only for a running pair kept as a switch's recovery target ([launch checks](operations.md#full-model-launch-checks)). `false` is the comparison arm and needs no image support. Expert parallelism is left untouched. On the reference pair identical requests repeat bit for bit, decode is not slower and the MTP acceptance length rose. It is on by default because a reproducible baseline is what a later A/B is read against.

`runtime.stable_indexer_topk` (template `true`; absent = the image's default, on in images built from 1.6.0) sets `GLM53_STABLE_INDEXER_TOPK` on both ranks. The kpool indexer folds four tokens into a pool and selects 512 pools per query row; the pinned `persistent_topk` (decode) and `top_k_per_row_prefill` return a different *set* for the same input when pools tie across the 512th rank, and one tie at one step forks a completion. With `true` a tie goes to the lower pool index: decode uses a stable sort (at most six rows, 0.07 to 0.25 ms per call against 0.01 to 0.02 ms, no synchronisation), prefill keeps the kernel and re-selects only the rows whose 512th value is shared by more pools than fit, which costs one synchronisation per call. Needs `GLM53_INDEXER_TOPK_API=1`. `false` is the comparison arm.

`runtime.inductor_deterministic` (template `true` from 1.12.0; absent or `false` = Inductor's timed choice) sets `TORCHINDUCTOR_DETERMINISTIC=1` on both ranks and moves `TORCHINDUCTOR_CACHE_DIR` to `/root/.cache/torchinductor-deterministic`. The replicated indexer normalises its key through a `torch.compile` leaf with three candidate configs; each rank used to pick one by timing at every launch, one of them sums a row in a different order, and when the ranks picked from different classes completions forked where pools tie. In deterministic mode Inductor picks reduction configs without timing, the same on every rank. torch 2.13 and 2.12.1 turn the mode off after the first compiled frame ([pytorch/pytorch#198563](https://github.com/pytorch/pytorch/issues/198563), reported on a GB10 in [vllm-project/vllm#58636](https://github.com/vllm-project/vllm/issues/58636)), so the launcher mounts `glm53_setup/runtime/inductor_pin.py` and a one-line `.pth` that keep the setting forced. Graphs compiled without the mode would come back from the existing cache with their timed candidates, so the mode compiles into a cache of its own; the first launch with the key compiles the indexer's leaves anew (about 45 files per rank). Pointwise leaves still autotune per rank; each element is computed by the same instructions whatever their block size. No image support is needed.

These switches cover a request alone. With `max_num_seqs` of 2 or more a request that shares steps with another can still get a different completion. Its attention changes path: at MTP depth 3 a partner makes a decode step eight query rows, more than six, so it goes through FA2 (`runtime.fa2_attention`, below). That is not the main cause, though: with every attention call forced onto the reference computation, most two-sequence completions still differed from their lone ones ([measurements](benchmarks.md#reachability-in-serving-2026-09-26)). The lead suspects are the NVFP4 Marlin MoE (its split along K follows the number of expert blocks in the step) and the prefill-sized kernels of a step shared with a prefill. For completions that repeat whatever else is in flight, serve `max_num_seqs = 1` ([concurrency scope](validation.md#concurrency-scope)).

`runtime.mla_decode_cpb` (in both examples of 1.14.0 and 1.15.0) was retired in 1.16.0: it never ran in serving, because the reference NoPE attention returns before the decode call its patch changed ([reachability in serving](benchmarks.md#reachability-in-serving-2026-09-26)). 1.18.0 removed it, and a profile that still carries the key is refused with a message to delete it.

### Attention, cache and checkpoint

`runtime.fa2_attention` (false when absent; template `true`) sets `GLM53_FA2_ATTENTION` on both ranks. With `true`, calls of the candidate-preserving NoPE attention with more than six query rows go through FlashInfer's `BatchMLAPagedAttentionWrapper` with the `fa2` backend and page size one, each row's candidates being its KV pages, instead of the reference computation. The packed `fp8_ds_mla` cache stays as it is; the rows a call touches are unpacked to BF16, because FlashInfer 0.6.18 accepts no FP8 MLA KV off SM90. Every selected candidate is kept, so prefix caching, fused unpack and candidate order are unchanged upstream of the call. Calls of up to six rows stay on the reference path: `plan()` needs each row's length on the host, one synchronisation per MLA layer, cheap beside a prefill chunk and costly in a decode step. The threshold counts the rows of the whole call. Every decode step of one sequence (at most six rows, MTP depth five) stays on the reference path, but with two sequences a verification step at depth 3 is eight rows and goes through FA2, whose result for a row then depends slightly on the other sequence's row count and lengths; the reference computation's does not ([measurements](benchmarks.md#reachability-in-serving-2026-09-26)). On the reference pair prefill of 38,962 tokens became about 2.2 times faster ([measurements on 1.6.0](benchmarks.md#measurements-on-160)). The fused unpack takes the element count at run time, so a 256K series no longer compiles one Triton kernel per count. The path excludes LPA. `server preflight` requires an image that carries it (`GLM53_FA2_ATTENTION_API=1`, row `fa2_attention_support`), a recovery target included; the checkout still mounts the path, its dispatch and that fused unpack over the image.

`runtime.prefix_page_dedup` (absent = off, as the pinned pool behaves; set in the AXL example) sets `GLM53_PREFIX_PAGE_DEDUP` on both ranks and needs `GLM53_PREFIX_DEDUP_API=1` (images built from 1.9.0). The pinned block pool registers every full block under its hash even when another block with the same hash is already cached; under a draft the prefix lookup drops the last matching block and recomputes it, so every re-sent history adds one block per KV cache group to the LRU queue under a hash that already has one, and those copies push older histories out first. With the key on, such a block is not registered: it keeps no hash, returns to the front of the free queue when its request ends, and lookups keep hitting the copy cached first. Block ids and numerics are untouched ([measurements on 1.9.0](benchmarks.md#measurements-on-190)).

`runtime.derived_checkpoint` (optional table, absent by default; `enabled = false` inside it keeps the table and serves the pinned snapshot) serves a locally requantized copy of the pinned checkpoint. It takes `path` (absolute directory on both hosts, mounted read-only at `/derived`), `requant_target` (compared with `quantization_config.producer.requant_target` of that directory) and `overlays`, a list of `{target, source, sha256, base_sha256, marker}`: `source` is an absolute file mounted over `target` in the image's GLM model directory. `server preflight` fails unless the checkpoint declares `MIXED_PRECISION` with that target, no quantized module is declared on the MTP draft layer, each overlay file has the given SHA-256 and contains its marker, and the image's own `target` has `base_sha256`, so an overlay built for another image cannot be mounted. The MTP metadata view is not used with a derived checkpoint: under `MIXED_PRECISION` undeclared modules, the BF16 draft layer among them, load unquantized. The two overlays this repository serves are shipped in [`overlays/`](../overlays/README.md): `kda-quant-split.py` over `kda.py` and `mla-quant-split.py` over `model.py`. Two targets exist: `g` (the attention projections) and `l` (the attention projections and `lm_head`); the current `l` checkpoint declares the split KDA input projection (`quantization_config.producer.in_proj_layout = "split-qkv-bfg"`) and loads only with these two overlays, the earlier fused pair belonging to the revision without that key. The reference pair serves `l` with MTP k=3, and that checkpoint is published with its overlays' requirements on its model card ([P23 in the optimization catalog](optimization-catalog.md), [serving profile](benchmarks.md#the-reference-pairs-serving-profile)).

`cache.prefix_cache_retention_interval` (template `"dense"`) and the runtime behaviour when it is absent are described in [APC history qualification](launch-safety.md#apc-history-qualification).

`runtime.index_checks` accepts `auto`, `sync` or `async` (the distribution default). Auto preserves synchronous checks in eager execution and selects asynchronous checks for Graphs. Explicit async enables the independently measured eager path; it requires `GLM53_ASYNC_INDEX_CHECK_API=1`. Checks are always performed. Invalid indices in async mode can invalidate the CUDA context, requiring both ranks to restart. Graphs reject explicit sync.

### Parallelism and transport

`runtime.nccl_channels` (absent = NCCL chooses; template 8) sets `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` to the same positive integer on both ranks. On the reference pair NCCL 2.30.7 chooses 64 channels by itself; at 8 channels and MTU 1500 the full model's lowest free memory rose by 2.8 GiB on the head and 3.0 GiB on the peer, while prefill did not slow ([channel-count measurements](nccl-validation.md#channel-count)). A profile written before 1.3.1 has no key and keeps NCCL's choice; add the key to adopt the template value. Informed by Mia PR #200.

`runtime.expert_parallel=false` is the default. The opt-in adds `--enable-expert-parallel` on both ranks while retaining TP=2/DP=1, the current precision and fixed KV budget, and requires `GLM53_EXPERT_PARALLEL_API=1`. Its scope is eager, one/two sequences and no MTP/LPA/fusion/APC. It was measured on the full model and not adopted ([Expert Parallel](performance-investigation.md#expert-parallel-p21)). Existing TOMLs must include the key explicitly; there is no missing-key fallback.

`runtime.pipeline_parallel_size=1` retains TP=2. Setting it to 2 selects TP=1/PP=2 on the same two nodes and requires `GLM53_PIPELINE_API=1`. `pipeline_split_layer` sets the first stage's layer count; the default candidate 24 produces stages 24/21, each with 21 MoE layers in this pinned model. It is not a memory-fit guarantee. Both stages must contain MLA, so this checkpoint accepts boundaries 4–43. Scope is one sequence, eager and no EP/MTP/LPA/fusion/APC; PP2 was measured and not adopted ([TP versus PP](performance-investigation.md#tp-versus-pp)). Include both keys explicitly when updating a TOML.

### Image input

`runtime.vision` is false when absent; the template sets `true`. `false` keeps `--language-model-only`, so the vision tower is not loaded and requests stay text/tools only. `true` removes that flag on both ranks and adds `--limit-mm-per-prompt '{"video": 0}'`. **Video input is disabled and rejected even with `vision = true`; only images are accepted.** The reason is startup memory: vLLM profiles by encoding the largest item once, and this checkpoint's video budget (capped at 30,000 tokens, 120,000 patches) is far larger than one image (up to 8,000 tokens). The image count per prompt keeps the vLLM default, because chat harnesses resend earlier images every turn. With vision on, `cache.mm_processor_cache_gb` (0.1 when absent) sets `--mm-processor-cache-gb`, the preprocessed-image cache that vLLM keeps in both the API and engine processes, which run on rank 0 only; the vLLM default of 4 GiB could take up to 8 GiB of host RAM on the head. A processed image larger than the budget is served uncached (with a warning) rather than rejected. The vision tower is BF16 and excluded from quantization (`model.visual*` in both exclusion lists). Measurements, the head's memory margin and open items are in [image input](vision.md). Validation fixtures load text-only regardless of this key.

### API and diagnostics

`api.prompt_tokens_details` (off when absent; template `true`) adds `--enable-prompt-tokens-details` so `usage.prompt_tokens_details.cached_tokens` reports the restored prefix; without it vLLM returns `null` and harness cache displays stay at zero even when the cache hits.

`api.dev_endpoints` (false when absent) sets `VLLM_SERVER_DEV_MODE=1` on both ranks, which mounts vLLM's dev routes on the loopback API: `/reset_prefix_cache`, `/reset_mm_cache`, `/collective_rpc`, `/sleep`, `/wake_up` and `/server_info`. LPA, component and expert profiles already run in that mode; the key lets a profile with LPA off reset the prefix cache without a restart, for benchmarks and for the warmup ladder's cleanup. The routes have no authentication ([launch contracts](launch-safety.md#model-api-clients)); leave it off on a kit with other local users.

`validation.memory_probe` (false when absent) mounts a worker extension reached through the dev route `POST /collective_rpc` (for example `{"method": "allocator_stats"}`). It excludes LPA and the other worker extensions (one extension class per launch) and changes nothing else about the launch. Its methods:

| Method | What it returns |
|---|---|
| `allocator_stats` | Per rank, the torch caching allocator's reserved, allocated, active and inactive-split bytes, segment count, allocation retries and `mem_get_info`. On GB10 the GPU shares host memory, so allocator growth appears as falling `MemAvailable` while the worker's RSS and cgroup stay flat; this tells the two apart |
| `host_stats` | The worker's resident anonymous memory, glibc's `mallinfo2` split and torch's pinned host cache; with `{"trim": true}` it calls `malloc_trim(0)` between two readings, separating retained free memory, live objects and memory outside malloc |
| `host_census` | Live Python objects by type and the CPU tensors among them |
| `weight_digest` | Two byte sums of every parameter and buffer as loaded, per layer and overall; `{"tensors": true}` returns the rows. `tools/weight_digest.py` records it after a switch and names the tensors that differ from the previous launch |
| `kernel_hashes` | Hashes of the kpool indexer's computations on fixed inputs inside the worker (head gate, gate score, FWHT quantisation, pool-cache writes, paged MQA logits and the stable top-k) and, from 1.11.1, of one layer's stages with its own weights (`layer`, default 19). `tools/kernel_hashes.py` compares the ranks and an earlier launch |
| `autotuners`, `inductor_state` | Each live Inductor kernel's served config and where it came from; Inductor's settings, the `TORCHINDUCTOR_*` environment and every write to the deterministic switch |
| `fa2_stage` | Runs only part of the FA2 path (`off`, single compaction operations, `compact`, `plan`, `full`) and serves the rest from the reference path, so the stage that starts a memory growth is named without a restart per stage |
| `trace_begin`, `trace_end` | Fingerprints (two wrapping sums over the bytes as int64 words, kept on the device) of what the traced modules, the draft and the sparse NoPE attention take and return for one request; a later identical request names the first differing call. `sync` synchronises after every traced call. Candidate-index tensors are fingerprinted with each row sorted, so only another set of candidates reads as a difference. Cache rows are gathered only for decode-sized calls. `{"keep": true, "export": true}` returns the rows for `trace_differences` to compare two launches offline |

A worker method that raises answers HTTP 500, and so does the next `/collective_rpc` call, whatever it is; the calls after that answer normally (measured on the reference pair, 2026-09-20). Discard the first call after an error. Measure a new reading at the serving shapes before adding it; the four-layer fixture runs shapes too small to show its cost. The routine that uses these readings after every switch is in [launch safety](launch-safety.md#after-a-switch-the-decode-check).

`validation.expert_worker=true` exposes the typed `expert_info` diagnostic for actual placement, kernel and parameter metadata. It supports the independent eager TP2 baseline and EP arms with up to two sequences; other validation workers, MTP/LPA/APC and PP are excluded. Layer hashing begins only after an explicit `pipeline_observe` RPC; do not install those hooks during performance measurement.

`validation.component_worker=true` selects a separate observer worker for CUDA A/B and indexer observations ([feature combinations](#feature-combinations-and-limits)).

`resources.stall_seconds` (0 = off when absent; template 600) and `generation.warmup` / `generation.warmup_long_tokens` (false / 0 when absent) are described under [supervision, stall detection and warmup](operations.md#supervision-stall-detection-and-warmup). Client authentication, `runtime.cuda_allocator_conf`, `nodes[].additional_rails` and the two-rank switch are in [launch contracts](launch-safety.md).

### LPA with prefix caching

P22 has scoped GPU isolation, crossover, combined and final held-out evidence. When LPA and prefix caching are both enabled, an image with `GLM53_APC_LPA_API=1` is required. The scheduler chooses from the jointly restored prefix and `lpa.break_even_tokens`; the template uses the conservative measured cutoff 128 from the [P22 calibration](benchmarks.md#apc-first-lpa-crossover-measurement-p22). Shared publication stops at the first approximation and remains stopped through the exact tail/decode. `server ask` leaves the decision to the server in this mode; ordinary no-APC `server ask` uses the same threshold with H=0. The template ships `lpa.enabled = false`; enable it per workload for batch inputs only, since an approximated request publishes nothing to the shared cache. While it is enabled, use `"vllm_xargs": {"glm53_lpa_mode": "off"}` in a request to compute normally and grow exact shared cache. The threshold key must be explicit in every updated TOML. See the [implementation contract](apc-lpa-design.md) and [LPA](lpa.md).

## Commands

Before changing context or concurrency, review [KV capacity and RAM requirements](#kv-capacity-and-ram-requirements).

Inspect generated commands from the checkout root (also works on Windows):

```sh
python -m glm53_setup server plan --rank 0
python -m glm53_setup server plan --rank 1
```

On each Linux host, `server preflight --rank N` checks assets, fabric, image identity, that no other container holds the GPU, and available memory ([what each check covers](operations.md#full-model-launch-checks)). To start a pair that is not running, start rank 1 first, then rank 0, in separate terminals on their respective hosts; to replace a running pair, use the [two-rank switch](launch-safety.md#all-rail-checks-and-two-rank-switch):

```sh
python -m glm53_setup server start --rank 1
python -m glm53_setup server start --rank 0
```

The commands stay in the foreground supervising their own containers; keep the terminals running. `resources.run_seconds` **includes model loading**. Ctrl+C, deadline or low memory stops that rank. Stop both ranks after a distributed failure. Containers and `records/` logs/configuration snapshots are retained; no automatic deletion or restart occurs.

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

`agreement` sends four self-authored texts (Japanese, English, code, mathematics) through `/v1/completions` with `prompt_logprobs` under the request lock, twice each, and writes the rank and log-probability of every actual next token to `records/<stamp>-agreement-r0/result.json`; with `--reference <result.json>` it adds the argmax agreement, top-5 overlap and log-probability drift against that earlier run. `self_agreement` carries the within-run difference: with the repeatability switches on, the two passes agree bit for bit; before `runtime.canonical_moe_order`, the argmax agreed on about 96% of positions and one log-probability moved by 8 nats. The record passes when every request was answered in the checked shape; the per-text mean negative log-probability is the steadier reading across runs. Within-run agreement is not evidence of agreement across launches ([comparing a candidate](validation.md#comparing-a-candidate-with-an-unchanged-control)).

`capacity` reads the running head's boot log and `/metrics` and prints the KV pool as it is: the stock `GPU KV cache size` line decomposed into `num_gpu_blocks`, blocks per maximum-length request and the group block widths. A cached-conversation estimate (blocks and conversations at 16K, 64K and `max_model_len`) is printed only for a profile that loads the LPA worker extension, whose `apc_cache_layout` RPC names each group's spec kind; otherwise that figure is withheld rather than guessed. `warmup` runs the request ladder under the request lock and writes `records/<stamp>-warmup-r0/result.json`; it exits nonzero when a rung failed. `mojibake` asks the running head for long Japanese and Korean answers under the same lock, counts broken characters in the answers and the reasoning, writes `records/<stamp>-mojibake-r0/result.json` and exits nonzero unless every answer passed ([what it checks](validation.md#full-model-tp2-experimental-scope)).

Use `--rank 1` on the worker to inspect/stop it. All actions accept `--config path/to/settings.toml`. Restart both ranks after editing settings; the client refuses a profile different from the running container's fingerprint. Generation defaults apply to `server ask`; external clients supply their own request options.

## Runtime limit and continuous operation

`resources.run_seconds` controls the automatic time limit in seconds, including loading. Set it to `0` for **no time limit**; positive integers retain a bounded session. Negative values are rejected. The `resources.reserve_gib` memory protection remains active in either mode. Settings are read at launch, so editing the file does not change an already running supervisor.

```toml
[resources]
# Change this entry in the existing resources section:
run_seconds = 0
```

This enables a run without a scheduled stop, not a 24/7 availability guarantee. Automatic host-startup integration, coordinated two-rank recovery outside a switch and redundant failover are not implemented, and the foreground supervisor must remain alive.

## KV capacity and RAM requirements

**Retaining B requests simultaneously at their maximum total length C requires capacity for B×C tokens.** `max_model_len` bounds input plus generated tokens per request; `max_num_seqs` limits concurrency. Setting both does not reserve or qualify that worst-case capacity. The two-sequence evaluation of 2026-09-12 covered up to 2,112 tokens per request; on the published option two ~200K requests were served together on 2026-09-23 ([measurements on 1.10.2](benchmarks.md#measurements-on-1102)); another two-Spark recipe reports two concurrent 25–100K requests falling to about 4 tok/s combined (tonyd2wild #14, no code adopted).

This launcher's `cache.kv_cache_memory_bytes` sets a **fixed KV-pool byte budget shared by requests on each rank**. With a 1 GiB setting, changing concurrency from one to two leaves 1 GiB per rank. It is neither 1 GiB per request nor one freely combined pool across both nodes. Explicit bytes override utilization-based KV sizing; `gpu_memory_utilization` is not a total-RAM safety cap in this mode. [vLLM configuration](https://docs.vllm.ai/en/latest/configuration/engine_args/#kv-cache-memory-bytes)

Live requests consume pool blocks according to retained input and generated tokens. Qualifying maximum-length concurrency requires testing that maximum, including output budgets. GLM combines sparse MLA, IndexPool and per-sequence KDA state: use the pinned runtime's cache specs, block alignment, per-group capacities and state slots rather than applying a generic dense-attention bytes/token formula. Include additional speculative state when enabling MTP.

On each node, weights, KV/cache state, activation/indexer workspaces, MTP/Graph allocations, CPU/OS/other load and operating reserve must fit unified RAM. Fixed KV bytes do not fix every other allocation: context, chunks and concurrency can enlarge other buffers. Container limits and `reserve_gib` protect the host; they do not certify fit or uninterrupted operation. The earlier 256K text-only profile ran close to that guard on the measured 121 GiB hosts: available memory sat near 4.5 GiB and two supervised stops (`stop-reason: memory-reserve`) occurred at a reserve of 4, the second while serving one 16,859-token approximated request; the reserve then moved to 3. The image profile kept about 4.0–4.2 GiB available on the head while serving with NCCL's 64 channels ([measurements](vision.md#memory-final-profile)); with 8 channels the head stayed at 6.97 GiB or more through the [200K checks on 1.3.1](benchmarks.md#200k-real-input-on-131) and at 6.40 GiB or more with chunk 2048. At 256K with 3 GiB KV it stayed at 5.82 GiB or more ([measurements on 1.5.0](benchmarks.md#measurements-on-150)), and the template reserves 3. The value may be fractional. Before changing it, account for what the guard can catch: the supervisor samples `MemAvailable` every 2 seconds, and one supervised stop took about 9 seconds from the breaching sample to container exit, so a slide of 0.15 GiB/s (measured during a runtime compile burst) can carry the host about 1.6 GiB below the reserve. The reference hosts recorded no kernel or container OOM kill and have 16 GiB of swap for host pages, so the risk below the reserve is a GPU allocation failing inside a worker, which ends the engine abruptly instead of through a supervised stop. Driver `NV_ERR_NO_MEMORY` kernel messages appear with 4 GiB or more available, mostly during load, and do not mark the floor.

Insufficient KV can cause startup rejection or runtime waiting, preemption and recomputation. The fixed pool does not automatically expand to meet demand. Other allocations or an insufficient RAM budget can still cause OOM or guard stops. [vLLM preemption](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption)

The boot line `GPU KV cache size: N tokens, Maximum concurrency for L tokens per request: Cx` is `N = C × L` for this hybrid model (MLA, IndexPool tail, KDA state groups and the MTP draft share one block pool with one id per group per aligned segment). `N` is therefore a concurrency figure in token units, not the number of conversation tokens the prefix cache can hold; `server capacity` prints the decomposition and, where the group kinds are known, the conversation estimate. In both measured image-profile configurations a GiB of KV held 28 blocks of 4,608 tokens and a full-length request of L tokens took ceil(L / 4608) + 16 of them: 61 of 70 at 204,800 tokens with 2.5 GiB and 73 of 84 at 262,144 with 3 GiB, 1.15× both times. The 256K figures were predicted this way before the switch; other lengths, KV sizes or group layouts need their own boot line. Prefix caching also works in whole blocks, so a repeated N-token prompt restores `(floor(N / block) - 1) x block` tokens and nothing at all below two blocks. Measured on the image profile's 4,608-token scheduler block: 3,625 tokens restored none, 14,025 restored 9,216, and 28,025 restored 23,040. Short conversations get no reuse on this profile whatever the hit rate suggests. Retain the runtime's reported capacity/concurrency estimates, then test the intended input-plus-output length × concurrency while observing preemption, both ranks' minimum free memory, OOM and guard stops. A current preflight pass does not replace this maximum-capacity test. See the [independent batching measurements](benchmarks.md#independent-active-batching) for actual coverage.

## Current image contract

Build the reference image from the checkout you launch (`python -m glm53_setup build-reference`), verify its ID on both hosts and set it as `reference_image` (and `runtime.lpa_image` for LPA). There is no old-command or old-image fallback. Every image built from the current source carries all markers below; the table says which setting makes `server preflight` require each one, and from which version an image carries it (— : earlier than the changelog records markers).

| Marker | Required by preflight when | Carried from |
|---|---|---|
| `GLM53_REFERENCE_ATTENTION=1` | Always (`reference_attention`) | — |
| `GLM53_LPA_API=2` | `lpa.enabled` (`lpa_worker`) | — |
| `GLM53_APC_LPA_API=1` | LPA with prefix caching (`apc_lpa_support`) | — |
| `GLM53_FUSED_UNPACK_SUPPORTED=1` | `cache.fused_unpack` (`fused_unpack_support`) | — |
| `GLM53_ASYNC_INDEX_CHECK_API=1` | Asynchronous index checks (`async_index_check_support`) | — |
| `GLM53_DECODE_GRAPH_API=1` | `runtime.decode_graphs` (`decode_graph_support`) | — |
| `GLM53_EXPERT_PARALLEL_API=1` | `runtime.expert_parallel` or `validation.expert_worker` (`expert_parallel_support`) | — |
| `GLM53_PIPELINE_API=1` | `runtime.pipeline_parallel_size = 2` (`pipeline_support`) | — |
| `GLM53_COMPONENT_API=1` | `validation.component_worker` (`component_worker`) | — |
| `GLM53_MOE_ORDER_API=2` | `runtime.canonical_moe_order` (`moe_order_support`; marker 1 only for a recovery target) | 1.7.0 (1 from 1.6.0) |
| `GLM53_INDEXER_TOPK_API=1` | `runtime.stable_indexer_topk` (`indexer_topk_support`) | 1.6.0 |
| `GLM53_FA2_ATTENTION_API=1` | `runtime.fa2_attention = true` (`fa2_attention_support`) | 1.6.0 |
| `GLM53_SLOT_MAPPING_GUARD=1` | Never; without it requests above about 250K tokens fault on the published option ([operations](operations.md#full-model-launch-checks)) | 1.7.0 |
| `GLM53_PREFIX_DEDUP_API=1` | `runtime.prefix_page_dedup` (`prefix_dedup_support`) | 1.9.0 |
| `GLM53_KPOOL_SEED_STRIDE=1` | Never ([operations](operations.md#full-model-launch-checks)) | 1.13.0 |

Images built from 1.14.0 through 1.17.0 also carry `GLM53_MLA_DECODE_CPB_API=1` and the unreachable patch of the removed `runtime.mla_decode_cpb`; no check reads them, and they are harmless.

`docker image inspect IMAGE --format '{{json .Config.Env}}'` shows the markers an image carries.

## Feature combinations and limits

`runtime.decode_graphs` (template `false`; absent = eager) is the one switch for decode Graphs: `true` selects `CompilationMode.NONE` and `FULL_DECODE_ONLY`; the one capture size is `num_speculative_tokens + 1` with MTP on (the pinned runtime rounds decode sizes up to that multiple and rejects `[1]`) and `1` otherwise. Prefill is uncompiled. Graphs may be combined with MTP and prefix caching for one sequence: on the four-layer MTP fixture with the expert token order fixed, eager and graph runs gave identical tokens and logprobs at every length ([component validation](component-validation.md#independent-decode-graph-fixture)). LPA requires eager, and startup rejects multi-sequence Graph configurations. On the full model Graphs decoded slower than eager and were not adopted ([decode Graphs on the full model](benchmarks.md#decode-graphs-on-the-full-model)). The earlier spelling `runtime.enforce_eager` (`false` = Graphs) is still read, so existing profiles keep their fingerprint; a profile that carries both keys must not let them contradict.

The Graph path checks internal candidate-index bounds asynchronously on the GPU. Invalid indices cause a device assertion rather than being ignored; unlike a regular Python exception, this can render the CUDA context unusable. Stop and reinitialize both ranks after such a failure. Include retained Graph memory and startup capture time in comparisons. [vLLM #53366](https://github.com/vllm-project/vllm/issues/53366) reports that the compilation-cache hash omits the speculative token count; if compiled Graphs are ever used together with MTP, keep a separate cache per k or clear it when k changes.

`validation.component_worker=true` selects an explicitly separate observer/validation worker, requiring `GLM53_COMPONENT_API=1`. It requires eager execution, one sequence, no LPA/MTP and no prefix caching. Its typed RPCs collect bounded indexer observations and toggle exact unpack fusion between exclusive requests for A/B/A tests. This is a diagnostic profile, not an enabled Reuse/Reindex serving mode; use one controlling client. Measured CUDA A/B and indexer observations are recorded in [component validation](component-validation.md).

`cache.fused_unpack=true` is the distribution default; false keeps the Torch reference conversion. True selects a single Triton kernel for the 656-byte MLA cache record's FP8 latent conversion and FP32 scale multiplication, and requires `GLM53_FUSED_UNPACK_SUPPORTED=1`. It does not change selected candidates or share KV between layers.

Set `mtp.enabled` and `lpa.enabled` independently. MTP selects the view prepared with [prepare_mtp_view.py](../tools/prepare_mtp_view.py) and BF16 Triton drafting; `mtp.num_speculative_tokens` accepts 1 to 5: all five were measured with the requantized checkpoint and 1, 3 and 4 with the pinned one, and both examples use 3 ([speculative decoding](speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21)). LPA selects `runtime.lpa_image`, mounts the projector read-only and enables the worker extension; it excludes `runtime.fa2_attention`, and combined with MTP it accepts depth 1, 2 or 3 (validation refuses 4 and 5) and requires an image with the explicit MTP-aware LPA worker. Depth 2 was checked with LPA on the four-layer fixture only ([component validation](component-validation.md#apc-first-lpa-cache-isolation-p22)); `lpa.py` and `apc_worker.py` are baked into the image, not mounted, so an image built before depth 2 was allowed refuses it on every request.

LPA needs per-request prompt length. The client tokenizes the actual template, configures LPA, generates, checks token-count agreement and resets LPA to off. Prompts fully covered by `lpa.tail` run normally. Use one controlling client only; a host lock serializes this CLI's requests, but does not coordinate arbitrary direct API clients. This convenience client handles non-streaming text/tool chat.

Scope: TP=2, Marlin W4A16, FP8 KV; text, tool calls and images. LPA requires one active sequence and eager execution; prefix caching with LPA uses the P22 path above. More than one active sequence is accepted only for the published option's two-sequence profile ([concurrency scope](validation.md#concurrency-scope)). Changing context, chunks, cache sizes, cut or tail needs new workload measurements; a schema-valid setting does not certify quality or resource fit. See [LPA](lpa.md) and [MTP](speculative-decoding.md) for evidence.
