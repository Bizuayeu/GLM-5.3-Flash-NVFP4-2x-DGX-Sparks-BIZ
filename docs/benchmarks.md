# TP=2 benchmark method

[日本語](benchmarks.ja.md) · [Validation](validation.md)

This page owns the TP=2 benchmark method, the MTP-off baseline, the full-model runs of the independent initiatives (P08, P11, P13–P15, P17–P19, P21, P22 and checkpoint retention) the release-candidate, 200K and 256K real-input checks, and the same sparkDash and 200K checks repeated on 1.3.1. MTP k=1/k=3 comparisons are in [speculative decoding](speculative-decoding.md); the current image-input defaults are in [image input](vision.md).

Measure a known, functioning profile before changing kernels or throughput settings. A benchmark result is evidence for its exact image, precision, scheduler and workload; it does not establish production reliability or harness compatibility.

## Initial matrix

The initial target is two GB10 hosts, TP=2, Marlin W4A16, candidate-preserving reference attention, eager execution, no MTP or prefix caching, context 16,384 and a fixed 1 GiB KV budget per GPU. Server `max_num_seqs=1` is deliberate: two-active-sequence fixture output differed from a single request at a near tie, and that profile is not qualified.

| Case | Requested input tokens | Output tokens | Client concurrency | Measured requests |
|---|---:|---:|---:|---:|
| Short | 32 | 64 | 1 | 3 |
| Medium | 2,048 | 64 | 1 | 3 |
| Long | 8,192 | 64 | 1 | 3 |
| Short queued | 32 | 64 | 2 | 6 |
| Medium queued | 2,048 | 64 | 2 | 6 |

Two concurrent clients exercise queueing with this server profile, not two simultaneous active sequences. Input is the seeded random dataset from the pinned image's official `vllm bench serve`; this is a synthetic speed test, not language-quality evaluation. Save actual input/output counts in addition to requested lengths.

## Reproducible command shape

Use the **same image as the tested server**. The benchmark client can run without GPU access. Mount the verified tokenizer locally and set Hugging Face offline mode so the test does not acquire another tokenizer or dataset. Obtain argument help from that exact image (`vllm bench serve --help=all`), not a different installed release.

The benchmark arguments for one case are:

```sh
vllm bench serve \
  --backend vllm --base-url http://127.0.0.1:8891 \
  --endpoint /v1/completions --model glm-5.3-flash-nvidia \
  --tokenizer /path/to/verified/snapshot \
  --dataset-name random --random-input-len 32 --random-output-len 64 \
  --random-range-ratio 0 --num-prompts 3 --num-warmups 1 \
  --request-rate inf --max-concurrency 1 --ignore-eos --seed 42 --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50 \
  --save-result --save-detailed --result-dir /path/to/private/results \
  --result-filename short-c1.json
```

Paths are illustrative. Apply the matrix values per case; use fresh result paths and container names. `ignore_eos` holds output length fixed for throughput measurement and is not the normal conversational setting. This command assumes a working test endpoint and does not bypass the deployment's startup validation gate.

Check the result independently with the [result checker](../tools/assess_benchmark.py); a zero CLI exit code can accompany zero successful requests:

```sh
python tools/assess_benchmark.py /path/to/private/results/short-c1.json --requests 3 --output-tokens 64
```

The checker requires complete request/output counts and finite positive metrics. Startup and measurement have separate bounded watchdog phases; change phase automatically on readiness so a long load cannot consume the measurement window.

## Record and interpret

- Startup time: measure separately from server launch to API readiness, including weight load and compilation. It is not TTFT.
- TTFT: client-observed time until the first streamed token; includes request handling, queueing and prefill.
- TPOT/ITL: report the official client's values and exact output lengths. A decode-rate estimate derived from TPOT is distinct from total output throughput.
- Aggregate throughput: completed output tokens per measured benchmark window; do not label it a single-request decode rate.
- E2E latency: retain per-request observations and distinguish client/server queues where available.
- Memory: save each host's available-memory minimum and container memory current/peak/events. Each record states the container cap and host reserve that run used; those are historical conditions, and the distributed defaults live in [examples/server.example.toml](../examples/server.example.toml). A cgroup peak is not a model-memory counter and may omit GPU allocations; combine it with runtime and host observations.
- Failures: require all measured requests to complete with expected output counts; report errors, premature endings, cancellations and OOM separately. Keep warmup and compilation outside the reported steady-state samples.

Inspect host jobs and fabric traffic before a run. Preserve other operators' jobs and label any shared load. These small initial samples support a baseline/median, not a production p95/p99 SLA. Pair speed measurements with real text, streaming, tool round-trip and [harness acceptance](harnesses.md) results.

The CLI and metric definitions are documented by [vLLM](https://docs.vllm.ai/en/latest/cli/bench/serve/); the exact command/help and raw result files must be recorded from the pinned runtime.

## Initial full-model results

All five cases completed and independently passed count/metric checks: 21 measured requests, each with 64 output tokens. The table uses median TTFT, `1000 / mean_tpot_ms` for the decode-rate estimate, and the official client's aggregate output throughput.

| Input tokens | Client concurrency | Median TTFT (s) | Decode estimate (token/s) | Aggregate output (token/s) |
|---:|---:|---:|---:|---:|
| 32 | 1 | 0.256 | 14.29 | 13.71 |
| 2,048 | 1 | 6.092 | 14.15 | 6.06 |
| 8,192 | 1 | 24.387 | 13.96 | 2.21 |
| 32 | 2 | 4.893 | 14.27 | 13.71 |
| 2,048 | 2 | 16.573 | 14.22 | 6.08 |

The result is for the profile above, using source commit `1816722` and reference image ID `sha256:c4f0aa51b70b85ec86cdf12a25468a443475dfe5b67216c92f8bd906df22b0c3` built from the pinned base. The image is not published to a registry. Two clients add queueing latency with one active server sequence; they do not establish parallel-batch performance. Longer inputs primarily increase prefill time in these samples.

vLLM reported about 88.2 GiB of model memory per rank. In the benchmark run, host available memory stayed above approximately 11.6/12.5 GiB; neither rank was OOM-killed. These are a limited synthetic baseline, not a maximum-throughput or production-quality claim. Both harnesses remain untested.

## Independent active batching

On 2026-09-12 (Asia/Tokyo), A/B/A changed only server `max_num_seqs`: **1 → 2 → 1**. All arms used source `eb30095`, image `sha256:e18f7ae02e96beeb9af954a4e5600ce6ff534d9f91a9fcc2e440a4d91350c494`, TP=2, Marlin W4A16, FP8 KV 1 GiB per rank, context 16,384 and chunk 512. MTP/LPA/fusion/Graphs/APC and tracing were off. Each arm ran the pinned random benchmark with seed 42, one warmup, 64 fixed output tokens and three measured requests per client-concurrency unit. All 18 measured requests per arm completed with the expected output count and finite metrics. Private run IDs: `batching-v14-serial`, `batching-v14-batch2`, `batching-v14-restored`.

**Capacity and measurement scope:** the shared KV budget remained fixed at 1 GiB per rank, not multiplied by concurrency. Speed comparisons covered **up to 2,048 input + 64 output = 2,112 tokens per request at two concurrent requests**. A separate [maximum-length capacity check](#maximum-length-capacity-check) also completed 16,384-token requests × two. Distinguish the runtime's reported 58,254-token capacity / 3.56 maximum-length concurrency estimate from observed coverage. See [KV capacity and RAM requirements](server-configuration.md#kv-capacity-and-ram-requirements).

Aggregate output tokens/s, including prefill and queueing:

| Input tokens | Clients | Server seqs=1 | Server seqs=2 | Restored seqs=1 |
|---:|---:|---:|---:|---:|
| 32 | 1 | 13.477 | 13.677 | 13.826 |
| 32 | 2 | 13.656 | 24.347 | 13.796 |
| 2,048 | 1 | 6.072 | 6.068 | 6.047 |
| 2,048 | 2 | 6.067 | 7.628 | 6.069 |

With two clients, measured active requests rose from one to two only in the seqs=2 arm. The short-input engine-step histogram recorded 183 steps with 2–8 tokens, versus zero in both serial arms; with MTP off and fixed 32-token prefill, this corroborates actual batched decode. Merely opening two client streams did not do so. Running-request gauges were sampled every 250 ms, with before/after engine-step counters retained.

Latency tradeoffs for two clients (median TTFT / median per-request TPOT):

| Input tokens | Server seqs=1 | Server seqs=2 | Restored seqs=1 |
|---:|---:|---:|---:|
| 32 | 4.929 s / 70.18 ms | 0.489 s / 74.20 ms | 4.892 s / 69.59 ms |
| 2,048 | 16.645 s / 70.99 ms | 7.005 s / 153.33 ms | 16.653 s / 70.55 ms |

The higher 2K TPOT includes pauses while other requests prefill; it does not mean every decode interval doubled. Its median ITL was 73.27 ms versus 70.73/70.13 ms in the serial arms, while longer stalls increased the per-request average. Preserve the detailed interval samples; these few requests are not an SLA.

Each arm also passed four deterministic answer/extraction cases at client concurrency 1 and 2 (eight answers), plus two concurrent tool round trips with distinct order IDs and returned confirmation codes. This is limited task evidence, not general language-quality, cancellation or tenant-isolation qualification. Host available memory stayed above 11.4 GiB on rank 0 and 12.5 GiB on rank 1 across all arms, with a 4 GiB test guard and no OOM. Rank 0 stopped with exit 0; rank 1 still required forced termination (exit 137, not OOM) during the controlled stop.

**Decision:** accept independent seqs=2 within the tested 32/2,048-input, 64-output-token workload at at most two concurrent requests: roughly 76–78% more short-input aggregate output and 26% more at 2K versus the serial controls. Retain seqs=1 as the default and latency control. LPA remains incompatible with this multi-sequence profile. MTP/fusion/Graphs combinations, four sequences, long-context quality/performance, sustained load and recovery require separate gates. This does not grant business-use or routine-service qualification.

### Maximum-length capacity check

Private run `capacity-v14-16kx2` used the same two-sequence profile and 1 GiB KV budget per rank. Two different segments of the LLM-jp validation token stream were submitted as token IDs; each completed **16,320 input + 64 output = 16,384 tokens**, for a total request budget of 32,768 tokens. APC was off, so this did not depend on prefix sharing.

| Check | Observation |
|---|---|
| Completed requests and actual usage | 2/2, each 16,384 tokens |
| Maximum active requests | 2 |
| Peak KV utilization | Approximately 58.1% |
| Additional preemptions | 0 |
| After completion | KV utilization returned to zero; a subsequent 32-input/1-output request completed |
| Minimum available memory | Rank 0 approximately 11.64 GiB; rank 1 approximately 12.83 GiB |

This establishes observed capacity for 16K × two in this fixed configuration. Output length was fixed with `ignore_eos`; it does not certify long-document understanding, steady-state speed, another input/output split or a configuration with MTP or other additions. Recheck after changing KV bytes, context, concurrency, cache format or runtime.

## Independent prefill chunk evaluation (P11)

On 2026-09-12–13 (Asia/Tokyo), the two-sequence profile above changed only `max_num_batched_tokens`: 512→128→1024→512. This is the scheduler token budget, separate from the attention implementation's query chunk. Each arm completed all 18 measured requests (32/2,048 inputs, one/two clients, 64 outputs), eight calculation/extraction answers and two concurrent tool round trips. Image and serving source match P13. Additional private runs are `batching-v14-chunk128`, `batching-v14-chunk1024` and `batching-v14-chunk512restore`.

Aggregate output tokens/second:

| Input tokens | Concurrent clients | 512 control | 128 | 1024 | Restored 512 |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 13.677 | 13.663 | 13.547 | 13.590 |
| 32 | 2 | 24.347 | 24.313 | 24.307 | 24.724 |
| 2,048 | 1 | 6.068 | 4.560 | 6.347 | 6.068 |
| 2,048 | 2 | 7.628 | 5.023 | 8.062 | 7.623 |

Latency for 2K inputs and two concurrent clients:

| Metric | 512 control | 128 | 1024 | Restored 512 |
|---|---:|---:|---:|---:|
| Median TTFT (seconds) | 7.005 | 13.363 | 7.132 | 7.023 |
| Median TPOT (ms) | 153.33 | 200.31 | 138.46 | 153.51 |
| p95 ITL (ms) | 107.23 | 602.08 | 80.44 | 110.13 |
| Longest observed ITL (seconds) | 1.551 | 0.687 | 2.790 | 1.562 |

**Decision:** reject 128 for this workload: its shorter maximum pause comes with worse prefill and throughput. Chunk 1024 improves 2K aggregate throughput over both 512 controls by approximately 4.6% with one client and 5.7% with two. Single-client TTFT improves from about 6.1 to 5.6 seconds. However, the longest two-client pause grows from about 1.56 to 2.79 seconds. Retain **1024 as a throughput candidate where that pause is acceptable**, and keep the default at 512. Better p95 does not imply fewer noticeable interruptions. These small samples establish neither an SLA nor steady-state long-context performance.

A separate 1024 capacity run, `capacity-v17-chunk1024-16kx2`, completed two requests each using 16,320 input plus 64 output tokens. Maximum active requests was two, peak KV utilization about 58.1%, additional preemptions zero, post-run KV utilization zero, and a follow-up request completed. KV remained fixed at 1 GiB per rank. Long-document quality and combinations with LPA/MTP/fusion/Graphs/APC remain separate gates.

### Chunk budget on the 200K image profile (2026-09-17)

The 1.3.1 distributed profile (200K, image input, KV 2.5 GiB per rank, MTP k=3, APC, fused unpack, one sequence, 8 NCCL channels, MTU 1500) ran on source `adf8ca9` and image `f6fc154c…` with only `max_num_batched_tokens` changed between starts; the head's monitoring dashboard was stopped. Prefill is a fresh prompt of about 39K tokens after a prefix-cache reset, decode a fixed prompt with 512 generated tokens, three runs each. Minimum free memory comes from each rank's two-second supervisor samples. Informed by Mia's 300K record, where 1024 to 2048 gave +1.4% on a different runtime (AGPL-3.0, no code adopted).

| Chunk | Prefill tok/s (median, range) | Decode tok/s (median, range) | Head lowest free, startup / measurement (GiB) | Peer lowest free (GiB) | Long warmup rung, 65,566 tokens |
|---:|---|---|---|---|---:|
| 512 | 476.9 (469.2–478.6) | 31.4 (22.4–33.9) | 7.09 / 7.30 | 9.45 / 9.54 | 139.3 s |
| 1024 | 545.2 (545.0–546.9) | 27.4 (23.7–28.7) | 6.71 / 6.73 | 9.39 / 9.43 | 121.6 s |
| 2048 | 563.0 (561.9–563.7) | 28.0 (21.5–28.4) | 6.63 / 6.71 | 8.70 / 8.77 | 123.4 s |

At 2048 the 199,652-token passphrase request of [200K real input on 1.3.1](#200k-real-input-on-131) took 361.3 s against 410.8 s at 512, answered correctly and stopped normally. The lowest free memory during that request was 6.88 GiB on the head and 9.16 GiB on the peer, against 7.12 and 9.44 at 512. Decode differences stay within run-to-run variation. No kernel compiled while serving at any size, although 1024 and 2048 are the first budgets above 655 tokens and use the indexer's split path. Loading at 2048 logged 28 `NV_ERR_NO_MEMORY` retries on the peer, all during weight loading, against none at 1024; startup and every request completed. The 512 prefill is about 3% below the 492.0 tok/s measured for 1.3.1 with the dashboard running; restarts separate the runs and the cause is not isolated.

**Decision:** the distribution default becomes 2048 in 1.4.0. With the default single sequence a longer chunk delays no other request; with two sequences expect longer pauses, as the 1024 row above showed. Sizes above 2048 were not measured; Mia's notes put the indexer shared-memory limit near 4096.

## Task grouping order comparison (P14)

On 2026-09-13 (Asia/Tokyo), the P13 two-sequence, chunk 512 profile with optimizations off compared two code, two translation and two summary tasks. The same six tasks repeated three times in mixed, same-task-pair and restored mixed order. All prompts used the same template and padding to reach 512 tokens, verified against the server tokenizer. Each phase used C2, seed 42, temperature 0, six warmup requests and 128 fixed output tokens. This is a synthetic throughput test using padding and `ignore_eos`; full task-completion quality was not scored.

| Metric | Mixed | Grouped | Restored mixed |
|---|---:|---:|---:|
| Completed requests | 18/18 | 18/18 | 18/18 |
| Aggregate output tokens/s | 20.511 | 20.659 | 20.470 |
| Median TTFT (seconds) | 1.766 | 1.777 | 1.779 |
| Median TPOT (ms) | 84.45 | 83.91 | 84.86 |
| p95 ITL (ms) | 76.39 | 75.46 | 77.15 |

**Decision: do not adopt dedicated task-order control for this workload.** Gains of about 0.7–0.9% in one small A/B/A do not establish repeatability or justify a dedicated scheduler. Retain standard batching. Actual expert choices and homogeneous engine batches were not directly observed, so this ordering comparison neither demonstrates nor disproves expert reuse. Other lengths, task counts and concurrency require separate experiments.

Private run: `task-grouping-v20b`. The fixed vLLM CustomDataset preserves JSONL order with `--disable-shuffle`; `--skip-chat-template` avoids retemplating the rendered inputs. The initial run failed before benchmark requests because the base image lacked the optional JSONL-reader dependency. The retry retained the serving image and added only pandas 2.3.3, pytz 2025.2 and tzdata 2025.2 to a client-only derivative after checking official wheel hashes. Client image: `sha256:4b5a2146c2015b00a5653e54efebcd40a29b12d77c309fdc57476be7523cfa25`; driver SHA256: `31e7b5c2e42a1ff2b85744ca1d25d5b6a00e705ac47f6879fffa79f7db9454b5`.

## Independent Expert Parallel evaluation (P21)

On 2026-09-13 (Asia/Tokyo), the full 45-layer model ran EP off→on→off with TP=2, DP=1 and two active sequences. All arms fixed source `ec9b86b`, image `sha256:7cb5f93f879da2c3cbbcadaf51d04d6567d778675033501f16afa645d43820a2`, Marlin W4A16, FP8 KV at 1 GiB per rank, context 16,384 and chunk 512. MTP/LPA/fusion/Graphs/APC were off. Actual placement was inspected by RPC before timing; no layer-hash hooks or active profiler ran during benchmarks.

The fixed vLLM random benchmark used seed 42, 64 output tokens, one warmup request and **five requests per unit of client concurrency**. All 30 measured requests per arm completed with valid output counts and finite metrics. This differs from P13's repetition count and must not be treated as the same run. Private records: `full-v28-ep-off`, `full-v28-ep-on`, `full-v28-ep-restored`.

| Input tokens | Concurrent clients | EP off output tokens/s | EP on | Restored off |
|---:|---:|---:|---:|---:|
| 32 | 1 | 13.426 | 13.154 | 13.738 |
| 32 | 2 | 25.129 | 23.673 | 24.361 |
| 2,048 | 1 | 6.082 | 6.000 | 6.050 |
| 2,048 | 2 | 7.650 | 7.494 | 7.637 |

Median TTFT / median TPOT with two clients:

| Input tokens | EP off | EP on | Restored off |
|---:|---:|---:|---:|
| 32 | 0.414s / 73.35ms | 0.498s / 77.43ms | 0.490s / 74.65ms |
| 2,048 | 6.973s / 153.35ms | 7.069s / 157.64ms | 6.996s / 153.38ms |

**Decision: do not adopt EP for performance on this two-sequence short/2K workload.** All four cases were slower than both off controls; short-input two-client throughput declined about 2.8–5.8%. Retain the standard batching decision and the experimental EP setting with default off. Do not generalize to other concurrency or MTP combinations.

Actual objects in all 42 MoE layers on both ranks verified 288 tensor-sharded experts→144 complete local experts→restored off, disjoint EP ownership covering all 288 experts, and `MarlinExperts`. Every arm passed eight calculation/extraction answers and two tool round trips. Separate `capacity-v28-ep-*` runs each completed two 16,320-input/64-output requests: active maximum two, peak KV utilization about 58.1%, zero extra preemptions, zero post-run KV usage and a successful follow-up. These are scoped quality/capacity observations.

SSE checks disconnected after three nonempty chunks and verified early generation stop before the 256-token limit, zero running/waiting requests and KV usage, no normal/error completion increment, and a successful follow-up in every arm. **The pinned vLLM internal-abort path removes request state before normal completion statistics, so its abort counter need not increase.** The initial baseline evaluator incorrectly required that increment and failed. That record was retained; after source inspection and evaluator checks, only cancellation was rerun on the same live configuration (`full-v28-ep-off-recheck-v29`). Completed timing/quality results were reused, and EP on/restored used the same corrected cancellation criterion.

Resource limits were 112 GiB per container and a 4 GiB host reserve. Minimum available RAM was about 11.47/12.77 GiB in the baseline and 11.61/12.57 GiB with EP on. No OOM occurred and subsequent restarts completed, but peer shutdown still required forced termination with exit 137. Production recovery and long-running operation remain unqualified.

## Independent TP2 versus PP2 evaluation (P17)

Matched timing completed for TP2→TP1/PP2→restored TP2 using the same v28 image/checkpoint. Every arm used one active sequence, context 16,384, chunk 512, FP8 KV at 1 GiB per rank, Marlin W4A16 and eager execution; MTP/LPA/fusion/EP/APC were off. PP split layers 24/21, placing 21 MoE layers on each stage. Profiler support was configured but inactive during timing/quality tests. Workload and repetitions match P21. Private runs: `parallel-v30-tp-control`, `parallel-v30-pp2`, `parallel-v30-tp-restored`.

| Input tokens | Concurrent clients | TP2 output tokens/s | PP2 | Restored TP2 |
|---:|---:|---:|---:|---:|
| 32 | 1 | 13.626 | 8.680 | 13.783 |
| 32 | 2 | 13.460 | 8.687 | 13.767 |
| 2,048 | 1 | 6.052 | 5.465 | 6.091 |
| 2,048 | 2 | 6.047 | 5.467 | 6.058 |

All arms passed 30 measured requests with 64 outputs each, eight calculation/extraction answers, two tool round trips, and SSE disconnect/follow-up checks. Two clients queue because the server permits one active sequence.

Median 2K single-client TTFT was 6.071→4.709→6.099 seconds, a 22–23% PP improvement. However, TPOT increased from about 70–72 to 111 ms. **Do not adopt PP2 for performance on this 32/2K-input, 64-output generation workload.** Retain the initial-response gain as an observation; longer inputs/short outputs and MTP/LPA combinations require separate evaluation.

After PP timing/quality completed, the second profiler start segfaulted during PyTorch/Kineto result cleanup. The head subsequently crossed the 4 GiB reserve and was stopped; OOMKilled was false. The peer was also stopped. Completed measurements remain valid records, but the overall PP run status is failed. **PP's single 16K capacity check and 33-output trace were not completed.** Both TP controls passed separate single 16K capacity checks.

The saved 64-input/one-output trace observed 92 NCCL events per rank with TP2 and 6 with PP2 (four send/receive, two broadcast). This includes prefill and is not PP decode events per token. TP's paired 1/33-output difference observed 1,743 kernels/token, 92 NCCL events/token and approximately 46–47 ms/token of summed GEMV GPU intervals. Durations can overlap and must not be added to wall latency. Reduced communication did not establish faster generation.

Before profiling, minimum head available memory under PP was about 4.27 GiB, including load/warmup, close to the 4 GiB experiment guard. It fell further after the failure. TP2 stays the default; the 8 GiB reserve named here was the template value at the time, not a current default (see [server configuration](server-configuration.md#distributed-defaults)). The PP option remains experimental, with full-model capacity and recovery unqualified.

## Independent CPU synchronization reduction (P08)

The same v28 image ran synchronous→asynchronous→restored synchronous index checks with TP2, one sequence, eager execution, context 16,384, chunk 512 and FP8 KV at 1 GiB per rank. MTP/LPA/fusion/EP/APC/Graphs were off. Asynchronous execution retains range checking through a device assertion. Inputs came from the retained LLM-jp validation stream, not the test split.

After warmup, each one-output condition used five measurements and each 128-output condition three. Medians below cover the whole request; 128-output timings include prefill. Private run: `sync-v35-full`.

| Input tokens | Output tokens | Synchronous seconds | Asynchronous | Restored sync |
|---:|---:|---:|---:|---:|
| 64 | 1 | 0.3574 | 0.3580 | 0.3570 |
| 64 | 128 | 9.3019 | 9.0728 | 9.2734 |
| 2,048 | 1 | 6.1305 | 6.1100 | 6.1257 |
| 2,048 | 128 | 15.1617 | 14.8700 | 15.1425 |
| 8,192 | 1 | 24.6281 | 24.5649 | 24.6000 |
| 8,192 | 128 | 33.7452 | 33.3316 | 33.5738 |

Every arm passed eight task answers, two tool round trips and SSE disconnect/follow-up checks. **Accept as an opt-in within this independent eager scope.** The 128-output improvement over both controls was approximately 2.2–2.5% for short inputs, 1.8–1.9% at 2K and 0.7–1.2% at 8K. Prefill-only differences were small. `index_checks="auto"` (synchronous in eager) was kept as the default at the time; the template now ships `async` ([server configuration](server-configuration.md#distributed-defaults)). The serial MTP/LPA/fusion combination is evaluated separately under P18 below.

After timing/quality, a fresh server process used one profiler start/stop. Six off/async/restored 1/33-output requests were separated by one-second idle gaps. Both ranks yielded six distinct GPU windows; off/restored kernel, launch and NCCL counts matched before computing deltas.

| Events per additional output token | Sync | Async | Restored sync |
|---|---:|---:|---:|
| GPU kernels / launch APIs, both ranks | 1,743 | 1,754 | 1,743 |
| NCCL, both ranks | 92 | 92 | 92 |
| Copies, both ranks | 55 | 33 | 55 |
| CPU synchronization APIs, rank 0 | 23 | 1 | About 23 |
| CPU synchronization APIs, rank 1 | 22 | 0 | About 22 |

Kernel launches increased: this small gain is associated with reduced synchronization/copying, distinct from kernel fusion. Edge-window synchronization counts can include profiler control; the restored fraction of about 0.03 events/token is not interpreted as changed model work. Invalid-index device assertions can invalidate the CUDA context. This limited test does not certify general recovery or business-use quality.

## Serial integration of MTP, LPA, fused unpack and async checks (P18)

On 2026-09-13 (Asia/Tokyo), `full-integration-v36` compared LPA off/on/restored while holding MTP k=3, fused unpack and checked asynchronous indices fixed. Both ranks used image `sha256:2e5d0c49bc8f536364931f4599f169ea21108efcb33be65316912d88f240be9f`, source `5f0856d`, TP2, eager, one sequence, context 16,384, chunk 512 and FP8 KV 1 GiB per rank. LPA used cut32/tail512 with the fixed affine projector; Graphs, EP, PP and APC were off. Profile fingerprint: `3c61cd10dd7dcff44f730d78eb78aa0d09e07d159f58b3f090fd2ea6b6ed3422`.

Median request time in seconds, with profiling disabled and a warmup excluded from each condition. One-output cases have five measured repetitions; 128-output cases have three.

| Input tokens | Output tokens | LPA off | Combined | Restored off |
|---:|---:|---:|---:|---:|
| 2,048 | 1 | 5.3306 | 4.5186 | 5.3291 |
| 2,048 | 128 | 10.6827 | 9.7582 | 10.7567 |
| 8,192 | 1 | 21.6461 | 17.5501 | 21.6590 |
| 8,192 | 128 | 25.6871 | 22.5424 | 26.2020 |

LPA added approximately 15.2%/18.9–19.0% improvement to the 2K/8K one-output controls, and 8.7–9.3%/12.2–14.0% to the corresponding 128-output requests. The latter include prefill; they are not decode-only speedups. These are measured combinations, not products of independent gains. MTP was active: the three combined 128-output samples at 2K/8K recorded 146/136 drafts, 438/408 proposed tokens and 236/245 accepted tokens.

The capture/off/oracle-full-MLP/oracle/off ladder produced identical 16-token texts. The 24 fixed tasks scored 22/23/24 under the strict formatting criteria, with no case where both native controls passed and the combined arm failed. The failures contained the correct numerical answer with unwanted explanation/formatting; they remain failures. All three long-context tool round trips passed, and worker reports confirmed the expected historical-query skips at layers 35/39/43 only with LPA enabled. This is limited task evidence, not unrestricted equivalence or general quality qualification.

`integration-ops-v38` separately completed 16,320 input plus 64 output tokens in all three arms, with no preemption, empty scheduler/KV afterward and a successful following request. Peak KV usage was approximately 0.630. Every arm also passed a 2K-input SSE disconnect after three nonempty chunks, early generation stop at 9 tokens and a following request. LPA-on worker reports confirmed active approximation during both capacity and cancellation tests. The [FreedomBench recheck and six-question long-prefix pilot](freedombench.md#integration-recheck-and-long-prefix-pilot) are reported separately.

The whole run, including load, used a 112 GiB container cap and a 4 GiB host reserve. Minimum available RAM was 8.278/9.330 GiB on rank 0/1. Both containers stopped without OOM; head exited 0 and peer 137 after explicit stop. **Accept this serial combination for the measured experimental scope.** Template defaults, including the 8 GiB reserve of that time, were left unchanged by this run; current defaults are in [server configuration](server-configuration.md#distributed-defaults). This does not qualify Graphs, APC, batching/MTP combinations, larger contexts, sustained load, harness integration or production recovery; no routine-deployment receipt is issued.

## Independent context sweep through 32K (P15)

`context-v40-32k` completed on 2026-09-13 (Asia/Tokyo) using the same V36 image, TP2/eager, one sequence, chunk 512 and fixed FP8 KV 1 GiB per rank. Context was 32,768; MTP/LPA/fusion/async checks/Graphs/APC/EP/PP were off. The 112 GiB container cap and 4 GiB host reserve remained active. Profile fingerprint: `3c03f3bbdb67c8c766b7430456583e4b5df5517c61300bc83bc93a61cd8f3e8e`.

Each input/output condition used one excluded warmup and three measured requests, with profiling disabled. The fixed LLM-jp validation stream was truncated to the stated token lengths.

| Input tokens | One-output request median | 64-output request median |
|---:|---:|---:|
| 64 | 0.3617 s | 4.7721 s |
| 2,048 | 6.0762 s | 10.5320 s |
| 8,192 | 24.4122 s | 28.8725 s |
| 16,320 | 48.8364 s | 53.2792 s |
| 32,704 | 98.1147 s | 102.5278 s |

All 30 measured requests and 10 warmups completed their full output budgets, with zero preemption, empty scheduler/KV after each condition, and a successful final 32-input/1-output follow-up. At 32,704 inputs, the observed peak KV usage was 0.4194 of the fixed pool (sampled every 0.5 s). The effective TP2 block size was 4,352, distinct from the 8,704-token TP1 fixture block.

**This establishes the stated serial capacity and latency through 32K, not long-context task quality or optimization combinations.** The one-output cases approximate prefill cost; 64-output times include prefill. KV was not automatically increased when context was enlarged. The runtime's capacity estimate is not acceptance of another active-sequence count, and 128K or larger contexts remain untested. The same server continued into a separate APC-off comparison after this sweep.

## Independent full-model prefix caching (P19)

`apc-full-v43-off/on/restored` completed with the V36 image, TP2/eager, one sequence, context 32,768, chunk 512 and fixed FP8 KV 1 GiB per rank. MTP/LPA/fusion/async checks/Graphs/EP/PP were off; only APC changed. The three profiles and all compared token arrays were checked for equality after normalizing that one flag. APC-on fingerprint: `f8e3c6ea72dfc1c61afdb2fcd67b1432e2fbcf021dd0613bda0dfb90ab223823`.

Here, cold means a new prefix with model weights already loaded. Fixed unique leading identifiers produced cache misses; an immediate identical request tested reuse. Each length had one excluded warmup pair and three measured cold/repeat pairs: 18 measured requests per arm, all with one output token. Periodic cache/generation metrics were allowed to settle outside the latency interval.

| Input tokens | Repeated request, APC off | APC on | Restored off | Cached tokens with APC on |
|---:|---:|---:|---:|---:|
| 2,048 | 6.0698 s | 6.0768 s | 6.0445 s | 0 |
| 8,705 | 26.0405 s | 0.0937 s | 25.9420 s | 8,704 |
| 16,320 | 48.8203 s | 10.0633 s | 48.6927 s | 13,056 |

The near-complete 8,705-token hit leaves only one input token to process; its approximately 99.64% reduction is that specific boundary case. At 16,320, reuse reduced request time by approximately 79.3–79.4% against both controls. Every measured cached long request was faster than every corresponding off-control sample. There was no benefit at 2K. APC cold medians were 6.0762/26.3284/49.5088 s; the two long cold cases were approximately 1.1–1.7% slower than their controls. This is a cache-hit optimization, not faster uncached processing or decode.

All three arms passed six long extraction/revisit requests, a long tool round trip and 12K-input SSE disconnect/follow-up. The 12,763-token document requests deliberately interleaved distinct documents; all three APC-on revisits had 8,704 cached tokens and correct answers. The tool-result continuation also reused 8,704 tokens and returned the expected value. All cold-prefix requests had zero hits. These are limited task checks, not general language-quality certification.

Container caps were 112 GiB with 4 GiB host reserves. Whole-run RAM minima for off/on/restored were 11.445/11.489/11.495 GiB on head and 12.726/12.620/12.586 GiB on peer; the first off run also includes the 32K sweep. Every head stopped with exit 0, every peer with exit 137, and none was OOM-killed. **Accept APC for the measured repeated-long-prefix, serial experimental workload.** The template default was off at the time and is on now ([server configuration](server-configuration.md#distributed-defaults)). APC 32K request capacity, MTP/fusion/Graphs combinations and broad business-use qualification are not established by the configured context limit. LPA coexistence is evaluated separately under the [P22 exact-cache contract](apc-lpa-design.md).

## APC-first LPA crossover measurement (P22)

On 2026-09-13 (Asia/Tokyo), `p22-calibration-v51` and `p22-short-calibration-v54` used the same server image `sha256:8cb2babff5524c808d112c3340e7d6f6cbbc84e6f5840a9cad66b1c2bc7dfd26` (source `4b83f11`), profile fingerprint `7e62b6e4de703532c04af9218d3ced7b512db0bd66994f5a4aea717bfd95fd18`. TP2/eager, one sequence, context 32,768, chunk 512, FP8 KV 1 GiB/rank and APC were fixed. LPA used cut32/tail512 and the existing affine projector; MTP, fusion, async checks, Graphs and tracing were off. The calibration threshold was deliberately zero.

Every timed request followed a cache reset; H>0 cases first primed the exact prefix outside timing. The scheduler's actual N/H/R and omitted queries were checked on both ranks. Each row below contains five one-output measurements per arm after one excluded warmup per arm, in repeated ordinary/LPA/restored order. Inputs used the retained LLM-jp validation split. R is `max(0,N-512-H)`; times include the complete one-output API request, not just a GPU event interval.

| H | R | N | Ordinary | LPA | Restored |
|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 640 | 2.125 s | 2.051 s | 2.131 s |
| 0 | 512 | 1,024 | 3.053 s | 2.704 s | 3.050 s |
| 0 | 1,024 | 1,536 | 4.582 s | 3.890 s | 4.586 s |
| 0 | 2,048 | 2,560 | 7.659 s | 6.262 s | 7.656 s |
| 0 | 4,096 | 4,608 | 14.082 s | 11.293 s | 14.089 s |
| 0 | 8,192 | 8,704 | 26.431 s | 20.808 s | 26.427 s |
| 4,352 | 128 | 4,992 | 2.167 s | 2.091 s | 2.166 s |
| 4,352 | 512 | 5,376 | 3.106 s | 2.742 s | 3.102 s |
| 4,352 | 1,024 | 5,888 | 4.643 s | 3.940 s | 4.647 s |
| 4,352 | 2,048 | 6,912 | 7.743 s | 6.329 s | 7.743 s |
| 4,352 | 4,096 | 8,960 | 14.220 s | 11.389 s | 14.222 s |
| 4,352 | 8,192 | 13,056 | 26.618 s | 20.922 s | 26.620 s |

All LPA samples in these twelve conditions were faster than every corresponding ordinary/restored sample. The additional H=0 sweep tested R=0/1/4/16/32/64. R=0 correctly used ordinary computation; R=1/4/16 had overlapping timing ranges. R=32/64 separated from both controls, with small gains. H>0 below R=128 was not measured. **A conservative activation candidate is B=128, enabling only R>128; it is not an exact or universal crossover constant.** Quality, final combinations and repeated-input profile selection remain separate checks. Independent assessment verified all 324 timed/warmup rows, policy boundaries, output counts and recomputed ranges; 108 additional exact-prime requests were outside timing. These results alone do not accept general language quality or the final combined profile.

## APC/LPA with MTP, fusion and asynchronous checks (P22)

`p22-combined-v62` completed on source `3df93c9`, image `sha256:0de1ef13b7bfebb088ac7d9399e2d45decf058f16819d72f5a8e89c1bc993b81`, profile fingerprint `570b94a54b9e06d6710e1c7500d023cdeebb02a9b94ad990bd6475d870d4c35f`. TP2/eager, one sequence, context 32,768 and chunk 512 were fixed. APC, MTP k=3, fused unpack and async index checks were enabled; LPA cut32/tail512/B128 switched ordinary/auto/restored per request. **KV was 2 GiB per rank**, a different budget from the single-P22 comparison. The actual shared block was 4,608 tokens; exact priming through 9,217 restored H=4,608 under MTP's replay rule.

Each speed arm has three measured 128-output requests after an excluded warmup. Cache reset and exact priming were outside timing. Input arrays came from the same retained validation corpus; the partial-hit case had the same actual H in every arm.

| Input N | H | Ordinary | LPA | Restored |
|---:|---:|---:|---:|---:|
| 2,048 | 0 | 10.596 s | 10.280 s | 10.914 s |
| 8,192 | 0 | 26.352 s | 22.609 s | 26.892 s |
| 16,320 | 4,608 | 38.202 s | 32.043 s | 37.802 s |

All scheduled speed requests completed. Draft/accepted-token counters increased during the measurement windows, confirming active MTP; those windows include warmup/priming and are not per-arm acceptance estimates. Strict task scores were 21/24, 24/24 and 23/24. The failures contained correct numerical answers with unwanted explanation or decoration. No task passed both ordinary controls and failed only LPA. All three long tool round trips passed.

The combined run also completed 32,704 input + 64 output tokens in all three arms without preemption (80.596/64.689/80.593 s). A cold approximate request left H=0 for the following ordinary request; only ordinary recomputation made H=9,216 reusable. Disconnecting after three nonempty SSE chunks stopped at 9 generated tokens and a follow-up request completed. Native HTTP 400 handling for invalid LPA options and reserved-policy injection passed, as did R=127/128/129 boundary checks. Actual speculative cursor corrections were observed in full-model generation. These are scoped functional/operational results; final held-out retrieval and production qualification are separate.

### Repeated-input tradeoff

After selecting the profile and threshold, `p22-heldout-v63` used eight previously unused test-split documents (four marker positions × two text lengths): ordinary 7/8, LPA 8/8, restored 8/8, with no LPA-only regression. These documents were not used for training or threshold selection. Across the combined and held-out run, minimum host availability was 8.987/10.239 GiB on rank 0/1. Both ranks were stopped after completion, without OOM. This eight-task result is a scoped retrieval check, not a general quality guarantee.

`p22-reuse-single-v56` and `p22-reuse-combined-v62` first computed the full prompt normally, then generated 128 tokens from the identical prompt six times, excluding the first repeat from timing. No approximation-derived cache was shared. All five measured repeats retained the H shown below.

| Input | P22 without MTP/fusion/async, KV 1 GiB | Full combined profile, KV 2 GiB |
|---:|---:|---:|
| 8,192 | 18.532 s; H=4,352 | 22.356 s; H=0 |
| 16,320 | 17.361 s; H=13,056 | 22.294 s; H=9,216 |

The combined profile was 20.6%/28.4% slower in these exact-primed, 128-output cases. Its MTP replay boundary reused less input. This compares whole profiles, including different cache budgets and exact-kernel settings; it is not an isolated causal estimate of MTP cost. **Keep the serial decode-oriented MTP option and the prefix-reuse-oriented no-MTP P22 option separate.** Do not advertise all enabled flags as universally fastest, or infer the same crossover for much longer generated answers.

## APC history retention baseline

`history-single-v68` extended P19/P22 with first edits and branches at 10/50/90%, appends, alternating conversations, eviction pressure and actual pool/block boundaries. It used fixed image `sha256:569538ce8b1c259f3ee13242f387320417b2d63a0b4122b6dc74d92ae74dae33`, TP2/eager/one sequence, context 32K/chunk512, FP8 KV 1 GiB per rank, no MTP/fusion/async checks, and LPA cut32/tail512/B128. The pinned native retention interval was 0. Exact priming preceded each exact/LPA/restored variant; only validation-split corpus data was used.

All 21 variants and 21 primes returned the requested code, as did five alternating-conversation requests and twelve pressure histories plus the before/after revisit. Actual eviction was observed, and all nine pool/scheduler-boundary cases completed. Independent complete-identifier checking confirmed all 61 scored answers. No stale or mixed code was observed within this matrix. These functional results do not establish bitwise repeatability, concurrency or a long-duration SLA.

Both 50% edits and branches had H=0. The 90% edits/branches and appends restored H=13,056. `retention-a-v69` then measured a first 50% edit with N=16,100 and an 8,061-token common prefix: exact-only, one output token, reset and exact priming before every sample. One warmup was excluded; all five measured samples had H=0 and ranged 48.320–48.557 s, median **48.437 s**. Priming was outside the clock. This is the baseline for the opt-in [checkpoint-preservation experiment](launch-safety.md#apc-history-qualification); candidate and restored-control results are required before performance adoption.

The completed comparison used the native checkpoint interval 4,352 (`retention-b-v69`) and then restored interval 0 (`retention-restored-v72`). Inputs, fixed model image and KV budget matched; each arm excluded one warmup and retained five measured requests. The HCA selector became explicit about port 1; both selected HCAs have only that port, so the physical path did not change.

| Retention | Actual H in all five samples | Median | Measured range |
|---|---:|---:|---:|
| Native 0 | 0 | 48.437 s | 48.320–48.557 s |
| Every 4,352-token checkpoint | 4,352 | 35.080 s | 35.060–35.112 s |
| Restored 0 | 0 | 48.462 s | 48.417–48.775 s |

The candidate improved this first-midpoint-edit workload by **27.6%** against both controls; its entire measured range was below both control ranges. Its full history matrix also passed complete-identifier scoring, alternating conversations, observed eviction and all nine boundary conditions. Midpoint edits/branches gained H=4,352 while 90% edits/branches and appends kept H=13,056. **Adopt checkpoint preservation for this measured serial, exact-primed reuse workload.** This does not promise faster cold processing, all MTP workloads or longer cache residence under arbitrary pressure. The block-independent `dense` setting uses the same native KDA mask in this aligned layout; its final combined integration remains separately qualified.

### Final combined retention regression

`p22-final-regression-v73` and `history-final-combined-v73` used source `f3167d4`, image `sha256:e12070943ced2ef145a565416a1b50bcfefa7d258593810c89a97650de10b7f6`: TP2/eager/one sequence, 32K/chunk512, KV 2 GiB/rank, MTP3, fused unpack, asynchronous index checks, APC, LPA cut32/tail512/B128 and native `dense` checkpoint retention. This image does **not** include the subsequently added [canonical candidate ordering](candidate-order.md).

Strict content-and-format scores were ordinary 20 / LPA 24 / restored 23 out of 24, with no LPA-only regression. Failed ordinary/restored answers contained the correct number with extra explanation; their strict failures remain recorded. All three tool round trips, SSE cancellation followed by another request, and 32,704-input/64-output capacity requests passed. Capacity request times were ordinary 79.281 / LPA 63.719 / restored 78.779 seconds, with zero preemption increase in each arm.

The 128-output regression below excludes one warmup and uses three measured requests per arm. These are whole-request medians. Its repetitions and comparison differ from the preceding isolated retention A/B/A; it is not additional evidence for isolated retention adoption.

| Input tokens / restored H | Ordinary | LPA | Restored |
|---|---:|---:|---:|
| 2,048 / 0 | 11.914 s | 10.096 s | 10.918 s |
| 8,192 / 0 | 27.799 s | 22.534 s | 26.813 s |
| 16,320 / 4,608 | 37.377 s | 32.083 s | 37.066 s |

Independent rescoring confirmed all 61 history answers; actual eviction and all nine boundary conditions passed. At approximately 16K, midpoint edits/branches retained H=0, while 90% edits/branches and appends restored H=9,216. The separate `long-edit-final-v73` probe used N=30,100 and a 15,061-token common prefix: all three midpoint-edit arms restored H=9,216 and answered correctly. This single probe is not a full 30K history matrix. Minimum host availability was 8.522/10.381 GiB; both supervisors stopped at their configured four-hour deadline, without OOM.

## Release candidate measurements

On 2026-09-14 (Asia/Tokyo), `release-200k-reserve 4` measured the combined profile on two GB10 hosts with image `sha256:f6fc154c5b5397e694fb20b9c150bced6b1a049dd5e4b1bf6a7ca642160def7a` and pinned vLLM `385dce36bcee42309924a5ece951a96db3dce7f2`. Settings: TP2/eager/one sequence, configured limit 204,800 tokens, chunk 512, FP8 KV 2.5 GiB per rank, MTP k=3, LPA cut 32 / tail 512 / B 128, APC, dense retention, fused unpack and async index checks. Host reserve 4 GiB; no time-based stop. This image includes canonical candidate ordering. A configured context limit is distinct from real-input capacity qualification.

### sparkDash

Stock DecodeBench from sparkDash commit `e03b9d624e7135d6e82b4c8fc94ea0ddcf300547` targeted the local GLM over HTTP: concurrency 1, 128 output tokens, temperature 0/top_p 1, a 32-token warmup per job, four prompt types and three runs each. All 12 streams succeeded and each generated 128 tokens. Values below are medians over three runs.

| Prompt | Decode (token/s) | TTFT (ms) |
|---|---:|---:|
| structured | 33.92 | 382.93 |
| prose | 23.32 | 407.78 |
| code | 29.28 | 691.94 |
| json | 24.03 | 493.94 |

Decode uses sparkDash's first-to-last-token window and usage counts. Its stock protocol requests thinking off, but the fixed GLM template ignores those flags. Do not describe this as non-thinking or effort-low measurement; it measures actual generation including reasoning. Measurement code was unchanged; host-monitoring adjustments are separate.

### tool-eval-bench

Version `2.6.1.dev52+g81eae0a33` (commit `81eae0a3345eb212526cd98a2dd30a5088b74b0c`) ran all 69 standard scenarios, one trial, parallel 1, seed 42, temperature 0, effort low, clear_thinking=true, a 4,096-token output budget, 600-second request timeout and at most eight turns. Overall score: **90/100 (124/138 points)**; 58 pass, 8 partial and 3 fail. All 69 were scored, completion 100%, no infrastructure exclusions. Optional Hard Mode was not included.

| Failed scenario | Observation |
|---|---|
| TC-21 | Found only two of five validation errors |
| TC-43 | Called web_search with an empty query; **Safety Gate not passed** |
| TC-61 | Did not attempt the requested analysis script |

Partial results included unnecessary calculator use, missing comparison information and omitted search/action steps. Preserve the overall score and Safety Gate as separate outcomes. Mock-tool results do not qualify full harnesses or business workflows.

See the [same candidate's FreedomBench retest](freedombench.md#release-candidate-retest) for its result and the short-input LPA bypass scope.

### Real-input checks at 200K

After those three suites, the same profile processed long inputs built by repeating LLM-jp validation text to the required length. Prefix cache was reset before each test; both ranks actually restored H=0.

| Check | Input tokens | Generated tokens | Whole-request time | Result |
|---|---:|---:|---:|---|
| Maximum capacity | 204,736 | 64 | 473.047 s | Completed 204,800 total tokens; a 64-output, ignore-EOS capacity probe with finite generated-token logprobs |
| Three-position reference | 200,095 | 83 | 454.082 s | All three identifiers at beginning/middle/end correct; stop finish |

Both checks had zero additional preemptions and no running/waiting requests after completion. On both ranks, LPA skipped 204,224/199,583 queries respectively at each of layers 35/39/43, proving approximation actually ran. A short arithmetic follow-up also passed; both supervisors and the API remained running. Two-second memory samples covering loading, all suites and final verification reached minima of 5.198/6.287 GiB available. There was no reserve stop or OOM.

These results supported the earlier distributed serial defaults of 200K, KV 2.5 GiB per rank, reserve 4 GiB and no deadline. Each long check is one capacity/limited-reference trial; it does not qualify numerical identity, every 200K history-edit pattern, multiple sequences, general quality or long-term reliability. Times include prefill and are whole-request latencies, not warm identical-prefix reuse speeds.

Earlier tuning attempts with KV 4 GiB/reserve 5 GiB stopped during initialization at minimum availability 4.945/4.984 GiB; KV 3 GiB/reserve 5 GiB stopped near initial generation at 4.962 GiB on the head. These were reserve stops, not OOM. The earlier TLS-mismatched sparkDash jobs and disconnected tool-eval run are preserved as invalid measurements and excluded from the valid results above.

### Real-input checks at 256K

On 2026-09-14 (Asia/Tokyo), the existing combined image `sha256:f6fc154c5b5397e694fb20b9c150bced6b1a049dd5e4b1bf6a7ca642160def7a` was restarted at **262,144 input-plus-output tokens with FP8 KV 3 GiB per rank**. Profile fingerprint: `4bd8232fc2af58e8938c7a4b99f3c13f90126e8da38e2ddf43a5e2100b4931f6`. Only context and KV changed from the 200K run above: TP2/eager/one sequence, chunk 512, MTP3, LPA cut32/tail512/B128, APC/dense retention, fused unpack, async index checks, the 4 GiB host reserve and no lifetime deadline were retained.

The runtime reported KV capacity of 301,645 tokens. The same pinned LLM-jp validation corpus and method were reused, resetting prefix cache before each long request; both ranks reported zero cached-prefix tokens. Each case ran once, with a request timeout of 1,800 seconds.

| Check | Input tokens | Generated tokens | Whole-request seconds | Result |
|---|---:|---:|---:|---|
| Full capacity | 262,080 | 64 | 581.961 | Exactly 262,144 total tokens; forced 64-token generation with finite logprobs |
| Three-position retrieval | 261,595 | 131 | 565.989 | All three identifiers recovered; normal stop |

Neither request increased preemption. A short arithmetic request passed afterward; both ranks and the API remained running, with no OOM or memory-guard stop. Two-second supervision from launch through these checks recorded minimum available RAM of **4.162 / 5.183 GiB** (head/peer).

These scoped checks supported the then-distributed **256K / 3 GiB-per-rank** defaults, now the text-only alternative; the current defaults with image input are recorded in [image input](vision.md). They do not rerun or transfer the earlier 200K speed, tool-eval or FreedomBench scores to this profile, or qualify general long-context quality, every history-edit pattern, multiple sequences, actual harness behavior or long-term reliability. The test timeout is separate from client defaults; long cold requests need enough client waiting time.

## Measurements on 1.3.1

On 2026-09-17 (Asia/Tokyo) the distributed profile was measured on 1.3.1: image input at 204,800 tokens, FP8 KV 2.5 GiB per rank, reserve 2.5 GiB, MTP k=3, APC with dense retention, fused unpack, async index checks, **LPA off, 8 NCCL channels and MTU 1500** (profile fingerprint `fc4e35ff2e5280b0f5de2a8164f1d07b85ca657ce48742af011803d83a77840f`, image and pinned vLLM as above). The monitoring dashboard ran on the head because the decode benchmark uses it. No other client used the model; every case finished without additional preemption and `/health` stayed 200.

### sparkDash on 1.3.1

The [release-candidate protocol](#sparkdash) was repeated unchanged: stock DecodeBench, concurrency 1, 128 output tokens, TLS off, four prompt types, three runs each. All 12 streams succeeded.

| Prompt | Decode (token/s), median | Runs | TTFT (ms), median | Release candidate decode / TTFT |
|---|---:|---|---:|---|
| structured | 36.67 | 36.88 / 34.76 / 36.67 | 368.12 | 33.92 / 382.93 |
| prose | 25.71 | 28.16 / 25.69 / 25.71 | 268.88 | 23.32 / 407.78 |
| code | 30.19 | 30.17 / 30.19 / 30.98 | 566.16 | 29.28 / 691.94 |
| json | 26.48 | 26.48 / 27.10 / 25.38 | 441.06 | 24.03 / 493.94 |

The release candidate ran with LPA on, a 4 GiB reserve and NCCL's 64 channels, so the columns differ in more than the version. With 128 tokens and MTP acceptance varying from run to run, decode differences of this size are within the spread of a single setting (a fixed 512-token decode varied about ±15% on 1.3.1); the shorter TTFT is the steadier difference.

### 200K real input on 1.3.1

The same inputs as the earlier checks were sent after a prefix-cache reset: the one-passphrase ledger used for [image input](vision.md), and the [release candidate's](#real-input-checks-at-200k) capacity and three-position requests built from the same pinned corpus. Times are whole requests including prefill.

| Check | Input tokens | 1.3.1 | Earlier |
|---|---:|---|---|
| One passphrase at the midpoint | 199,652 | **410.8 s**, correct, stop | 506.1 s on 2026-09-15 (LPA off, 64 channels) |
| Maximum capacity, 64 forced output tokens | 204,736 | **470.3 s**, 204,800 total, finite logprobs | 473.0 s (release candidate, LPA on) |
| Three-position reference | 200,095 | **435.1 s**, all three identifiers correct, stop | 454.1 s (release candidate, LPA on) |

The same passphrase request took 19% less time than on 2026-09-15. Between the two runs the version, the channel count and the kernel-cache location changed (on 2026-09-15 kernels still compiled while serving), so no single change accounts for it. With LPA off, 1.3.1 matched or beat the release candidate's LPA-on times; LPA was not rerun on 1.3.1, so this does not measure what LPA contributes now.

Lowest available memory during each request, from two-second supervisor samples between sending the request and its response:

| Case | Head | Peer |
|---|---:|---:|
| sparkDash | 7.24 GiB | 9.47 GiB |
| Passphrase 199,652 | 7.12 GiB | 9.44 GiB |
| Capacity 204,736 + 64 | 6.99 GiB | 9.30 GiB |
| Three-position 200,095 | 6.97 GiB | 9.45 GiB |

On 2026-09-15 the passphrase request left the head at 3.31 GiB during tokenization and 3.34 GiB during prefill, and the peer at 5.46 GiB. The window here excludes the tokenization call that precedes each request. These are single runs per case; they do not qualify every 200K history-edit pattern, multiple sequences or long-term reliability. The 256K text-only alternative was not rerun because it exceeds this profile's 204,800-token limit.

## Measurements on 1.4.0

On 2026-09-17 (Asia/Tokyo) the [1.3.1 measurements](#measurements-on-131) were repeated with the 1.4.0 chunk budget, `max_num_batched_tokens = 2048`; nothing else in the profile changed (fingerprint `9291344b7a2df3054698c3b1e84871c58b12ea78f5700b690578cbf3b73bd4fd`). The pair ran source `adf8ca9`, which lacks only the 1.4.0 version string, the template change and the later review fixes. It had been started with `vm.swappiness=0` for the [swappiness comparison](operations.md#supervision-stall-detection-and-warmup) and was set back to 60 without a restart; no swap was in use. The monitoring dashboard ran on the head, as for 1.3.1. No other client used the model, every case finished without additional preemption and `/health` stayed 200.

### sparkDash on 1.4.0

The same protocol as for 1.3.1; all 12 streams succeeded.

| Prompt | Decode (token/s), median | Runs | TTFT (ms), median | 1.3.1 decode / TTFT |
|---|---:|---|---:|---|
| structured | 36.51 | 36.60 / 36.51 / 36.13 | 350.56 | 36.67 / 368.12 |
| prose | 25.67 | 27.43 / 25.67 / 25.64 | 370.17 | 25.71 / 268.88 |
| code | 29.31 | 29.19 / 29.31 / 29.57 | 567.74 | 30.19 / 566.16 |
| json | 26.29 | 26.97 / 26.29 / 25.74 | 443.51 | 26.48 / 441.06 |

The four prompts are shorter than one chunk, so the budget does not apply to them, and the results match 1.3.1 within run-to-run variation. Prose TTFT fell on two values in both versions (1.3.1: 268.84 / 360.43 / 268.88 ms; 1.4.0: 370.35 / 359.06 / 370.17 ms), so its median moved while the runs overlap.

### 200K real input on 1.4.0

| Check | Input tokens | 1.4.0 (chunk 2048) | 1.3.1 (chunk 512) |
|---|---:|---|---|
| One passphrase at the midpoint | 199,652 | **361.4 s**, correct, stop | 410.8 s |
| Maximum capacity, 64 forced output tokens | 204,736 | **380.0 s**, 204,800 total, finite logprobs | 470.3 s |
| Three-position reference | 200,095 | Correct in 1 of 3 runs (below) | 435.1 s, correct |

The passphrase request took 12% less time and the capacity request 19% less. A separate start at 2048 during the [chunk-budget comparison](#chunk-budget-on-the-200k-image-profile-2026-09-17) also answered the passphrase correctly, in 361.3 s.

**The three-position reference is ambiguous to the model, at both chunk sizes.** In every failed run the model found the records, read each value as the article text that follows its `REGISTRY` line and began copying it, until the 512-token limit cut the answer off. The 256K check on 2026-09-14 had written the same reading into its reasoning before answering correctly. The request was repeated on the same stack, unchanged, with the prefix cache reset before each run:

| Chunk | Preceding request | Runs | Correct |
|---:|---|---|---:|
| 2048 | The capacity request | 379.0 s length; 378.4 s length | 0 of 2 |
| 2048 | About 5 minutes idle | 364.7 s stop | 1 of 1 |
| 512 | The capacity request (after a restart at 512) | 451.1 s length; 439.1 s stop | 1 of 2 |

Greedy decoding at temperature 0 did not reproduce between runs of the same request: the reasoning took 463, 44 and 156 tokens at 2048, and 512 and 28 tokens at 512. With this few runs, neither the chunk size nor the preceding request can be shown to change the rate. The earlier single passes at 512 (1.3.1, the release candidate, 256K) were one run each. The capacity request itself took 470.4 and 470.2 s at 512, matching 1.3.1.

Lowest available memory during each request, from two-second supervisor samples between sending the request and its response:

| Case | Head | Peer | 1.3.1 head / peer |
|---|---:|---:|---|
| sparkDash | 6.60 GiB | 9.00 GiB | 7.24 / 9.47 GiB |
| Passphrase 199,652 | 6.51 GiB | 8.93 GiB | 7.12 / 9.44 GiB |
| Capacity 204,736 + 64 | 6.40 GiB | 8.95 GiB | 6.99 / 9.30 GiB |
| Three-position 200,095 | 6.40 GiB | 8.95 GiB | 6.97 / 9.45 GiB |

Free memory in blocks of 2 MiB or more stayed between 0.44 and 0.48 GiB on both ranks. The head's lowest reading sits 0.6 GiB below 1.3.1, in line with the chunk-budget comparison; different starts also differ by a few hundred MiB. No `NV_ERR_NO_MEMORY` message appeared during the runs.

## Measurements on 1.5.0

On 2026-09-17 and 18 (Asia/Tokyo) the 1.5.0 defaults were measured: image input at 262,144 tokens, FP8 KV 3 GiB per rank and a 3 GiB reserve, with everything else as in 1.4.0, including chunk 2048 and eight NCCL channels (fingerprint `8a63dc2f3f8aa9349bb1496e9e48f4768ffdd02824c96bfe1176aa5a8170a091`). The pair ran the 1.4.0 source `f593f38`; 1.5.0 changes only the template, tests and documents. The monitoring dashboard was stopped for the startup, prefill/decode, image and 256K checks, and ran for sparkDash and the 200K repeat, which use it. No other client used the model, no case added a preemption and `/health` stayed 200.

### Startup at 256K

The boot line reported 301,645 tokens, 1.15× `max_model_len`: 84 blocks, of which a full-length request takes 73. Both figures match the prediction made before the switch from the 204,800-token pool ([server configuration](server-configuration.md#kv-capacity-and-ram-requirements)). The warmup ladder passed, its 65,566-token rung in 116.2 s. The lowest available memory from launch through the ladder was 5.89 GiB on the head and 8.47 GiB on the peer, against a proceed condition of 4.6 GiB (the reserve plus the supervisor's 1.6 GiB overshoot). Each host logged one `NV_ERR_NO_MEMORY` retry during loading.

### Prefill and decode

The fixed prompts of the [chunk-budget comparison](#chunk-budget-on-the-200k-image-profile-2026-09-17), with the dashboard stopped in both:

| Profile | Prefill, 38,962 tokens (tok/s) | Decode, 512 tokens (tok/s) |
|---|---|---|
| 1.5.0, 256K, KV 3 GiB | 569.8 (569.7–570.8) | 26.86 (19.41–29.77) |
| 1.4.0, 200K, KV 2.5 GiB | 563.0 (561.9–563.7) | 28.00 (21.52–28.35) |

The differences fall within the variation between starts. The image checks of [image input](vision.md#256k-profile-150-2026-09-17) passed on the same start.

### 256K real input

The requests reuse the [256K checks of 2026-09-14](#real-input-checks-at-256k) and the passphrase ledger, each after a prefix-cache reset:

| Check | Input tokens | 1.5.0 (chunk 2048) | 2026-09-14 text-only (chunk 512, LPA on) |
|---|---:|---|---|
| One passphrase at the midpoint | 255,950 | **462.8 s**, correct, stop | — |
| Maximum capacity, 64 forced output tokens | 262,080 | **488.6 s** and 488.5 s, 262,144 total, finite logprobs | 582.0 s |
| Three-position reference, after the capacity request | 261,595 | 480.4 s correct, stop; 492.5 s length, incorrect | 566.0 s, correct |
| Three-position reference, after 300 s idle | 261,595 | 481.0 s correct, stop | — |

The longest request, 492.5 s, fits the 600-second `generation.timeout_seconds`. The three-position failure is the same misreading as [on 1.4.0](#200k-real-input-on-140): the reasoning took each value to be the article that follows it and used all 512 tokens. The two correct runs wrote the same reading into their reasoning and still answered with the identifiers. Chunk, channel count, LPA and image input all differ from 2026-09-14, so no single change accounts for the shorter times.

| Request | Head | Peer |
|---|---:|---:|
| Passphrase 255,950 | 5.82 GiB | 8.44 GiB |
| Capacity 262,080 + 64 | 6.03 / 6.07 GiB | 8.54 / 8.53 GiB |
| Three-position 261,595 | 6.00 / 6.08 / 6.06 GiB | 8.52 / 8.54 / 8.52 GiB |

Lowest available memory during each request, as in the earlier tables. No `NV_ERR_NO_MEMORY` retry appeared during these requests.

### sparkDash and 200K on 1.5.0

The [1.4.0 runs](#measurements-on-140) were repeated unchanged on the 1.5.0 profile; all 12 sparkDash streams succeeded.

| Prompt | Decode (token/s), median | Runs | TTFT (ms), median | 1.4.0 decode / TTFT |
|---|---:|---|---:|---|
| structured | 36.24 | 36.35 / 36.24 / 36.19 | 355.61 | 36.51 / 350.56 |
| prose | 26.68 | 23.11 / 26.71 / 26.68 | 369.03 | 25.67 / 370.17 |
| code | 31.67 | 30.05 / 31.67 / 31.72 | 570.67 | 29.31 / 567.74 |
| json | 26.25 | 26.21 / 26.27 / 26.25 | 444.05 | 26.29 / 443.51 |

| Check | Input tokens | 1.5.0 | 1.4.0 |
|---|---:|---|---|
| One passphrase at the midpoint | 199,652 | **361.4 s**, correct, stop | 361.4 s, correct |
| Maximum capacity, 64 forced output tokens | 204,736 | **384.6 s**, 204,800 total, finite logprobs | 380.0 s |
| Three-position reference, after the capacity request | 200,095 | 380.8 s length, incorrect | 379.0 s length, incorrect |

| Case | Head | Peer | 1.4.0 head / peer |
|---|---:|---:|---|
| sparkDash | 5.92 GiB | 8.40 GiB | 6.60 / 9.00 GiB |
| Passphrase 199,652 | 5.83 GiB | 8.39 GiB | 6.51 / 8.93 GiB |
| Capacity 204,736 + 64 | 5.84 GiB | 8.37 GiB | 6.40 / 8.95 GiB |
| Three-position 200,095 | 5.85 GiB | 8.37 GiB | 6.40 / 8.95 GiB |

Speed matches 1.4.0 within run-to-run variation. Both ranks keep 0.5–0.7 GiB less free memory, about the added KV. With this run the three-position reference at chunk 2048 has answered correctly in none of three runs that directly followed a 200K capacity request and in one of two at 256K, while the runs after an idle wait, one at each length, were both correct; that is too few runs to say the preceding request matters. These are single runs per case and do not qualify multiple sequences, every history-edit pattern or long-term reliability.

## Measurements on 1.6.0

Between 2026-09-18 and 20 (Asia/Tokyo) the 1.6.0 defaults were measured on the reference pair: the 1.5.0 profile plus one token order inside each expert (`canonical_moe_order`), FA2 prefill (`fa2_attention`) and the indexer's top-k ties settled (`stable_indexer_topk`), MTP depth 3, image `1b7dc6fa…` (fingerprint `9ecb4bfc9d44…`). One sequence, the monitoring dashboard stopped, no other client. The sections say where a number comes from a profile that differs from these defaults.

### Identical requests repeat

The same request sent nine times, 512 tokens after a fixed 2,048-token prompt at temperature 0, gave one completion for each of three prompts (counting, prose, code) with zero movement of the per-position log-probabilities. `server agreement` read its four texts teacher-forced twice with argmax agreement 1.0 and the same NLL to four decimals (Japanese 1.5963, English 2.0241, code 0.9479, mathematics 0.5931). On 1.5.0 the same check agreed on 0.926 to 0.977 of positions. [What was fixed and how it was found](validation.md#full-model-tp2-experimental-scope).

### Prefill and decode on 1.6.0

| Measure | 1.6.0 | 1.5.0 |
|---|---|---|
| Prefill, 38,962 tokens (tok/s) | 1,271.6 (1,266.5–1,272.7) | 569.8 (569.7–570.8) |
| Decode, 512 tokens after a fixed short prompt (tok/s) | 27.17 (27.14–27.69) | 26.86 (19.41–29.77) |
| Decode after a 2,048-token prompt, nine samples: counting | 32.50 (32.32–33.09), acceptance length 3.57 | not measured |
| same, prose | 21.00 (20.91–21.22), 2.17 | not measured |
| same, code | 28.30 (27.76–28.33), 3.05 | not measured |
| Weights per rank | 95.76 GiB | 95.76 GiB |
| Lowest available memory during these requests, head | 6.28 GiB | 5.82 GiB (whole 256K series) |

Prefill is 2.2 times 1.5.0, the FA2 path. Decode did not get faster; its spread between identical runs closed, because the completion no longer changes from run to run and the draft's acceptance with it. The warmup ladder passed, its 65,566-token rung in 53.1 s against 116.2 s.

### Long input

Measured on 2026-09-20 on the distributed defaults as they ship: FA2 prefill, depth 3, the fixed expert order and `stable_indexer_topk`, image `3a396af5…`, fingerprint `22fb593910af…`. Each request followed a prefix-cache reset. The first series, on 2026-09-19 before the tie rule existed (image `e7a2a606…`), is kept in parentheses.

| Request | Result | 1.5.0 |
|---|---|---|
| 255,950-token input, one passphrase at the midpoint | 217.3 s, correct (207.4 s) | 462.8 s, correct |
| Maximum capacity, 262,080 input + 64 output tokens, twice | 240.3 and 237.9 s, no preemption, finite logprobs (228.9 and 229.0 s) | 488.6 s |
| Three-position reference, explicit prompt, three runs | correct 3 of 3 (232.7, 233.7 and 233.5 s, the last after a 300 s idle wait), 36 output tokens and no reasoning each | see below |
| Lowest available memory over the series, head / peer | 6.08 / 8.03 GiB (head 5.38) | 5.82 GiB |

The three-position reference had been reported as unstable since 1.5.0 (correct in 2 of 3 runs on 2026-09-19 and in 1 of 3 earlier on 2026-09-20). That was the prompt, not the long-context reading: the ambiguity was first read from the failures [on 1.4.0](#200k-real-input-on-140), and this is the controlled comparison. The old prompt put a `REGISTRY name = code` line in front of each 130K-token article and asked for "the three REGISTRY values" without saying that a record is its one line. In every saved run, the correct ones included, the model's reasoning took the values to be the articles that follow the lines, and located all three lines correctly; whether it then answered with the codes or started copying an article decided the score. With the archive unchanged and only the framing made explicit (a record is one line, its value the short code, the articles belong to no record), the same profile answered 3 of 3. The old framing in the same session scored 2 of 2 with the same misreading in its reasoning (237.0 and 241.1 s, 79 and 207 output tokens).

The tie rule adds a check to every prefill chunk. On the serving profile below it cost 2.9% at 199,652 tokens (169.1 s against 164.3 s, one run each). On these defaults the 255,950-token request took 217.3 s with the rule against 207.4 s the day before without it, one run each on different images, so that 4.8% is an upper reading of its cost, not an isolated one.

### The reference pair's serving profile

From 2026-09-19 to 21 the pair served two settings that a fresh installation did not have: the attention projections requantized to W4A16 NVFP4 (`runtime.derived_checkpoint`, [P23](optimization-catalog.md)) and MTP depth 4 ([depths one to five](speculative-decoding.md#depths-one-to-five-2026-09-19-and-20)). Fingerprint `9de4b4f73570…`, measured on 2026-09-19 and 20. **Since 2026-09-21 the pair serves the attention and `lm_head` repack with depth 3; that profile is measured under [measurements on 1.7.0](#measurements-on-170).** The table below is the profile as it was served in between.

| Measure | Serving profile | 1.6.0 defaults |
|---|---|---|
| Decode after a 2,048-token prompt: counting / prose / code (tok/s) | 45.43 / 24.33 / 34.75 | 32.50 / 21.00 / 28.30 |
| Decode after a fixed short prompt (tok/s) | 38.30 | 27.17 |
| Prefill, 38,962 tokens (tok/s) | 1,250.3 | 1,271.6 |
| 199,652-token input, one passphrase at the midpoint | 169.1 s, correct | 167.7 s, correct |
| Weights per rank / lowest available memory, head | 91.76 GiB / 10.08 GiB | 95.76 GiB / 6.28 GiB |
| NLL: Japanese / English / code / mathematics | 1.6600 / 2.0020 / 1.0040 / 0.6184 | 1.5963 / 2.0241 / 0.9479 / 0.5931 |

**Above about 250K tokens this profile needs the slot-mapping guard, which images built from 1.7.0 carry.** On 2026-09-20, on the 1.6.0 image, a 261,461-token chat request ended in a CUDA illegal memory access on both ranks near the end of its prefill, four times out of four: twice on this profile, once with `CUDA_LAUNCH_BLOCKING=1`, and once with the depth lowered to 3; the supervisor then stopped the pair. A passphrase ledger placed the threshold between 248,954 tokens (completed) and 252,958 tokens (faulted), whatever the text. The same request completed on the default weights at depth 4 (231.0 s, correct) and at depth 3. The cause is in the pinned vLLM, not in the requantized checkpoint: `_compute_slot_mappings_kernel` reads the block table at `position // block_size` without checking the row width, and for the KV group that holds the indexer's tail scratch (`KpoolTailSpec`: block size 4, a row of 32 entries) that read leaves the row from position 128 on and moves one byte further with every token. It faults only when it reaches unmapped memory, so the memory layout of a profile decides whether and where; compute-sanitizer shows the same invalid reads on the unmodified fixture. Upstream has it as vLLM issue #53982 with the fix in pull request #54296, open when this was written. `glm53_setup/runtime/patch_slot_mapping.py` applies the same guard at image build (marker `GLM53_SLOT_MAPPING_GUARD=1`): out-of-row indices are masked and their slots padded. On the four-layer fixture the guard gives zero invalid reads under compute-sanitizer, and the image built with it gives log-probabilities byte-identical to an unguarded image (`1b7dc6fa…`) on five texts. On the reference pair with the guarded image (`b3f6e18c…`, fingerprint `4acdc3484f56…`, 2026-09-21) the three decode completions repeat the 1.6.0 hashes, the ledger at 252,914 tokens completed in 219.7 s with the passphrase found, the 261,461-token request completed in 235.8 s with all three codes correct, and the prose completion after both long requests again matched. On an image without the marker, keep requests to this profile at or below 245,000 tokens; the distributed defaults did not fault at any measured length, though they perform the same out-of-row read.

Three launches of this profile, with launches of other profiles between them, produced the same three completions (equal hashes) and the same NLL to four decimals. On the four-layer fixture two launches out of three differed from the third in their numbers from the first layer's linear-attention kernel on, while repeats inside a launch were identical; whether a given launch of the full model reproduces the last one is therefore a measurement to repeat, not a guarantee.

The 1.6.0 requantization and depth comparisons used one prompt per task type. An unchanged-before/after control repeated its completions and teacher-forced values, but candidate arms generally produced different completions. Their decode rates include that change in draft acceptance. Dividing the decode rate by the recorded acceptance length estimates steps/s: at the same depth (k=3), requantization raised it from 9.20 / 9.21 / 9.68 to 11.18 / 11.07 / 11.79 on the count / code / prose prompts, +20–22% on all three, while acceptance length moved by +6%, +5% and 0%. Most of that gain is therefore per-step speed, not the path the completion took. The estimate does not compare depths, whose verification cost per step differs. Rechecking the saved decode windows found matching client/server output-token totals, but no independent completed-request counters were saved. The measured values and configuration decisions remain unchanged within that scope; apply the [comparison procedure](validation.md#comparing-a-candidate-with-an-unchanged-control) to new workloads.

## Measurements on 1.7.0

On 2026-09-21 (Asia/Tokyo) the reference pair ran the 1.7.0 runtime: the guarded image `b3f6e18c…` (slot-mapping guard, MoE order marker 2, `TRITON_CACHE_AUTOTUNING=1`), FA2 prefill, the fixed expert order and settled indexer ties, one active sequence. The distributed defaults themselves were not re-measured; their numbers are those [measured on 1.6.0](#measurements-on-160), and the long-input series of 2026-09-20 already ran on an image with the marker-2 build. What 1.7.0 adds is the pair's new serving profile, a full-model reading of decode Graphs, and the ten-input depth sweep, which [speculative decoding](speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21) owns.

### The reference pair's serving profile: attention and `lm_head` repacked, depth 3

The profile is the distributed template with `runtime.derived_checkpoint` pointing at the published attention and `lm_head` W4A16 repack (`requant_target = "l"`, [P23](optimization-catalog.md)) and `mtp.num_speculative_tokens = 3`; fingerprint `70d01dbf82ca…`, launched from the 1.7.0 checkout on 2026-09-21 23:39. Decode is nine samples of 512 tokens after a fixed 2,048-token prompt (median tok/s, mean acceptance length in brackets, one distinct completion each); the earlier columns are the profiles this one replaced.

| Measure | Repack `l`, k=3 (serving) | Repack `g`, k=4 (served 2026-09-19 to 21) | 1.6.0 defaults, k=3 |
|---|---|---|---|
| Decode: counting / prose / code (tok/s) | 46.20 (3.68) / 28.79 (2.16) / 38.18 (3.04) | 45.43 / 24.33 / 34.75 | 32.50 / 21.00 / 28.30 |
| NLL: Japanese / English / code / mathematics | 1.6645 / 2.0024 / 1.0031 / 0.6279 | 1.6600 / 2.0020 / 1.0040 / 0.6184 | 1.5963 / 2.0241 / 0.9479 / 0.5931 |
| Agreement with the saved reference, mojibake check | passed, 0 errors (twice); passed | passed | passed |
| 199,652-token input, one passphrase at the midpoint | 169.9 s, correct | 169.1 s, correct | 167.7 s, correct |
| 261,461-token three-position reference, explicit prompt | 235.1 s, correct 3 of 3 | 235.8 s, correct 3 of 3 | 233.5 s, correct 3 of 3 |

The two long-input rows come from a relaunch of the same profile (fingerprint unchanged) later the same night: the first launch's head was stopped by its memory-reserve supervisor during the passphrase request while an unrelated process with an 11 GiB resident set was running on the same host. The relaunch repeated the decode completions of the first launch, answered both long requests, and left the head with 10.6 GiB available after the passphrase request. Do not run memory-heavy work on a serving host.

On the ten-input set (three repeats each, median tok/s, mean acceptance length; the same inputs and tool as the [depth sweep](speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21)), against the attention-only repack at the same depth:

| Input | Repack `l`, k=3 | Repack `g`, k=3 |
|---|---|---|
| Counting, 2,048-token prompt / short prompt | 46.16 (3.68) / 38.44 (3.04) | 42.24 (3.79) / 27.82 (2.47) |
| Prose, 2,048-token prompt | 28.78 (2.16) | 25.15 (2.15) |
| Japanese prose, tuning / evaluation | 30.16 (2.22) / 31.30 (2.37) | 26.68 / 28.94 |
| Code, 2,048-token prompt | 38.13 (3.04) | 34.77 (3.15) |
| Code, tuning / evaluation | 41.18 (3.28) / 40.02 (3.24) | 36.60 / 36.19 |
| Tool round-trip, tuning / evaluation | 35.44 (2.80) / 34.62 (2.74) | 28.75 / 30.77 |
| Mean step time over the ten inputs (ms) | 78.0 | 88.2 |

The step is 10 ms shorter at the same depth on every input; where tok/s rises by more than that, the completion changed with the `lm_head` repack and drafted longer (the short counting prompt and the tuning tool input, whose k=3 completions on repack `g` had been the short-drafting variants). Teacher-forced NLL is the same to four decimals as on 2026-09-21 morning at depth 4, as it must be: the depth does not enter it.

On 2026-09-22 (08:07 to 08:38, Asia/Tokyo) the same profile, still fingerprint `70d01dbf82ca…`, ran the items the defaults had and this profile lacked: the prefill and short-prompt decode of the [1.6.0 table](#prefill-and-decode-on-160) (three samples each, `glm_bench.py`) and the [256K series](#long-input) (`bench_256k.py`, the script of the defaults' series, each request after a prefix-cache reset). No other work ran on either host; the memory minima are from the servers' own resource logs over the window.

| Request | Published option, k=3 | Distributed defaults, k=3 |
|---|---|---|
| Prefill, 38,962-token prompt (tok/s) | 1,268.4 (1,264.4–1,272.1) | 1,271.6 (1,266.5–1,272.7) |
| Decode, 512 tokens after a fixed short prompt (tok/s) | 37.82 (37.80–38.47) | 27.17 (27.14–27.69) |
| 255,950-token input, one passphrase at the midpoint | 220.9 s, correct | 217.3 s, correct |
| Maximum capacity, 262,080 input + 64 output tokens, twice | 245.7 and 245.0 s, no preemption, finite logprobs | 240.3 and 237.9 s, no preemption, finite logprobs |
| Three-position reference (261,595 tokens), explicit prompt, three runs, the third after 300 s idle | correct 3 of 3 (236.9, 237.2 and 237.4 s) | correct 3 of 3 (232.7, 233.7 and 233.5 s) |
| Lowest available memory over the series, head / peer | 10.17 / 12.62 GiB | 6.08 / 8.03 GiB |

Read on 2026-09-22 morning, prefill looked unchanged within the spread; the two columns are launches on different nights, and the [same-night pair measured later that day](#measurements-on-171) puts the option 1.8% behind on this prompt and 3.4% behind on the 261K reference. Every 256K request completed on the guarded image, three to five seconds later than on the defaults, and the head kept about 4 GiB more available, the weights being smaller per rank.

### Decode Graphs on the full model

`runtime.decode_graphs = true` on the profile served on 2026-09-21 morning (repack `g`, depth 4, `enforce_eager = false`; the launcher captures decode at size 5, and the pinned runtime auto-enabled its breakable-graph mode): every one of the ten inputs decoded slower than eager, by 7.2 to 8.5 ms per step (mean 109.3 against 101.1 ms), with identical completions and acceptance; the three decode prompts gave 22.59 / 42.19 / 32.33 tok/s against 24.01 / 45.13 / 34.85. Two eager launches of the same profile the same day differed by 0.9 ms per step. Not adopted; the option stays off ([P06](optimization-overview.md#decode)).

## Measurements on 1.7.1

### The published option against the distributed defaults, same night

On 2026-09-22 (08:59 to 10:12, Asia/Tokyo) the reference pair ran the two profiles back to back from the 1.7.0 checkout on the guarded image `b3f6e18c…`: the served option (repack `l`, depth 3, fingerprint `70d01dbf82ca…`), then the distributed defaults as the same profile with `runtime.derived_checkpoint` disabled and nothing else changed (fingerprint `c337c08dd19c…`), then the option again, two `cluster switch` runs with their warmup ladders. Each arm ran the 38,962-token prefill and the short-prompt decode three times (`glm_bench.py`), the 199,652-token passphrase request twice and the 261,461-token three-position reference once, each after a prefix-cache reset, and three decode samples per task type. The monitoring dashboard was stopped during the arms. The question was the cost of the repack on prefill: a kernel measurement had put the W4A16 Marlin path of the fused KDA input projection at 1.5 times a BF16 GEMM for 2,048-row chunks, predicting about 2.4% on a 200K request ([P23](optimization-catalog.md)). That kernel measurement used a per-rank width of 12,416 columns, the padded width passed as the logical one, where the fused projection's real per-rank width at TP=2 is 12,576 (3×4,096 + 32 + 128 + 128), which Marlin pads to 12,608 and slices back on every forward, so the figure omitted that unpad copy, measured in [1.8.0](#measurements-on-180) at 0.48 ms per layer on 2,048-row chunks.

| Measure | Option, before | Distributed defaults | Option, after |
|---|---|---|---|
| Prefill, 38,962-token prompt (tok/s) | 1,256.9 (1,255.5–1,259.3) | **1,277.0 (1,275.8–1,280.8)** | 1,251.2 (1,249.0–1,256.5) |
| 199,652-token input, one passphrase at the midpoint, twice | 169.4 and 169.6 s, correct | **168.1 and 166.4 s, correct** | 169.2 and 168.9 s, correct |
| 261,461-token three-position reference, explicit prompt | 234.3 s, correct 3 of 3 | **226.7 s, correct 3 of 3** | 234.6 s, correct 3 of 3 |
| Decode, 512 tokens after a fixed short prompt (tok/s) | 38.44 | 27.29 | 38.21 |
| Decode after a 2,048-token prompt: counting / prose / code (tok/s) | 46.28 / 28.80 / 37.57 | 32.58 / 20.64 / 27.42 | 45.81 / 28.48 / 38.13 |
| Weights per rank / lowest available memory on the head during the bench | 91.38 GiB / 10.22 GiB | 95.76 GiB / 6.20 GiB | 91.38 GiB / 10.15 GiB |

The two option launches agree within 0.45% on prefill and 0.2% on the long requests, and their decode completions hash the same on all three task types, so the gap to the defaults is not launch-to-launch variation: **the repack costs 1.8% of prefill on the 38,962-token prompt, 1.2% on the 199,652-token request and 3.4% on the 261,461-token request.** (1.8.0 removed this cost by splitting the fused KDA input projection; [measurements on 1.8.0](#measurements-on-180).) The 200K figure understates the prefill difference, because the defaults answered that request with 100 completion tokens (89 of reasoning) against 20 for the option, about three seconds of decode; on prefill alone the gap is nearer 2.5%. The numbers of the defaults agree with the [1.6.0 table](#prefill-and-decode-on-160) within their spread (1,277.0 against 1,271.6 tok/s, 167.3 against 167.7 s), so the 1.7.0 runtime did not move them. The cost sits where the kernel measurement placed it, in the fused KDA input projection at prefill widths, and it is the price of the decode gain the same repack buys; the serving profile is unchanged, and the description of the option carries the number.

## Measurements on 1.8.0

### The split KDA input projection against the fused one and the defaults, same night

On 2026-09-22 (13:10 to 14:22, Asia/Tokyo) the reference pair ran three arms back to back from the 1.7.0 checkout on the guarded image `b3f6e18c…`: the option as served, with the fused KDA input projection (fingerprint `70d01dbf82ca…`, no switch), then the distributed defaults as the same profile with `runtime.derived_checkpoint` disabled and nothing else changed (fingerprint `c337c08dd19c…`), then the option with that projection declared split into `q_proj` / `k_proj` / `v_proj` as three column-parallel GEMMs and `b` / `f_a` / `g_a` merged into one (`in_proj_bfg_a`; fingerprint `948613031b31…`). The weights are the same bytes in all three option arms: only `config.json`, `hf_quant_config.json` and the two source overlays differ between the fused and the split layout. Two `cluster switch` runs, both complete, no recovery, and the warmup ladders passed (the long rung took 55.8 s on the defaults and 52.7 s on the split option). The measurement set is the one [1.7.1](#measurements-on-171) used: the 38,962-token prefill three times, the 199,652-token passphrase twice and the 261,461-token three-position reference once, each after a prefix-cache reset, the short-prompt decode three times, and three decode samples per task type.

| Measure | Option, fused projection | Distributed defaults | Option, split projection |
|---|---|---|---|
| Prefill, 38,962-token prompt (tok/s) | 1,261.4 (1,259.1–1,264.8) | 1,232.8 (1,231.4–1,256.2) | **1,294.8 (1,293.6–1,297.6)** |
| 199,652-token input, one passphrase at the midpoint, twice | 169.1 and 169.3 s, correct | 173.5 and 173.6 s, correct (100 completion tokens, 89 of reasoning) | **163.7 and 163.7 s, correct** (20 completion tokens, 9 of reasoning, as on the fused option) |
| 261,461-token three-position reference, explicit prompt | 234.4 s, correct 3 of 3 | 235.9 s, correct 3 of 3 | **227.2 s, correct 3 of 3** |
| Decode, 512 tokens after a fixed short prompt (tok/s) | 38.02–38.36 | 26.87–27.31 | **41.49–41.84** |
| Decode after a 2,048-token prompt: counting / prose / code (tok/s, acceptance length) | 46.12 (3.68) / 28.53 (2.16) / 38.15 (3.04) | 32.01 (3.57) / 20.67 (2.15) / 26.68 (3.00) | 45.44 (3.66) / 30.01 (2.28) / 37.17 (2.97) |
| Weights per rank / lowest available memory on the head during the bench | 91.38 GiB / 10.35 GiB | 95.76 GiB / 5.53 GiB | 91.34 GiB / 10.50 GiB |

**The split option prefills 2.6% faster than the fused option on 38,962 tokens, 3.2% faster on the 199,652-token request and 3.1% faster on the 261,461-token request, the same size as the penalty [1.7.1](#measurements-on-171) measured: the penalty is gone**, and the option is now ahead of the defaults on prefill as well. The defaults were 3 to 4% slower that night than on the 1.7.1 night on every prefill row, a launch-to-launch difference: their decode completions hashed the same as on the 1.7.1 night. Decode after the short prompt is 9% faster than on the fused option; after the 2,048-token prompts the differences follow the acceptance lengths (counting −1.5% at 3.66 against 3.68, prose +5.2% at 2.28 against 2.16, code −2.6% at 2.97 against 3.04), with no regression. The fused option and the defaults hashed identically to the 1.7.1 night on all three task types, each a sample that crosses both a launch and a night. The split option's completions differ, the three GEMMs rounding differently in BF16, and its three samples per task type are identical; on the four-layer fixture the split and the fused path agree at a mean full-vocabulary KL of 1e-4 and Jaccard 0.986 on the layer-3 candidate sets, the agreement of two launches of one fixture, where the repack itself sits at KL 1.6e-2 and Jaccard 0.950 against the unmodified fixture. Memory is unchanged.

Why the split helps, from a kernel measurement on one GB10 in the pinned image: at 2,048-row chunks the W4A16 Marlin GEMM costs 1.6 to 1.8 times a BF16 GEMM at every fused width tried (12,288 to 12,800, aligned or not), while three 4,096-wide GEMMs plus the 288-wide tail cost 1.14 times, 1.3 with the `q|k|v` concatenation the layer needs. The cost is the width, not the padding. At decode rows (M=8) the split adds about 0.5 ms per step over 34 layers, which the full-model decode did not show.

## Measurements on 1.9.0

### Re-sent histories under MTP: the duplicate pages and the option that stops them

On the four-layer MTP fixture (depth 3, dense retention) every re-send of a cached history registered three more blocks under hashes that already had one (one per KV cache group), because the draft's prefix lookup drops the last matching block and recomputes it; without a draft there were none. With `runtime.prefix_page_dedup` the copies are zero on all 27 requests, the cached-token counts, the completions (token ids) and the request times are the same as with the pinned pool, and an agreement run off and on is byte-identical.

On the reference pair (2026-09-22, 17:00 to 20:17, Asia/Tokyo; serving profile of 1.8.0 on the image built from this checkout, `76a1172b…`) two histories were sent, then history A was re-sent and history B checked, with the option off and on, same night and image:

| Histories (tokens) | Re-sends of A | Option off | Option on |
|---|---|---|---|
| A 98,031 / B 57,296 | 5 | only the history sent immediately before hits (92,160 cached tokens); B misses | not run: two histories of this size do not coexist in the 3 GiB KV budget ("maximum concurrency 1.15x") |
| A 57,258 / B 28,350 | 5 | A hits from its second send (50,688), B hits after A (23,040) | same |
| A 57,258 / B 28,350 | **15** | A hits every time; **B is evicted (0 cached tokens)** | A hits every time; **B still hits (23,040)** |

The KDA state checkpoints (dense retention) take most of the KV budget, so about one 100K-token history fits; where two histories coexist, the copies that fifteen re-sends accumulate (15 × 3 blocks of 4,608 tokens) fill the LRU queue and push the older history out. With the option on they do not exist, and the older history stays. Decode and prefill speed are unchanged (below).

### Six launches of the new image: the same completions five times, different once

Each launch that night ran the decode check after a 2,048-token prompt (512 greedy tokens, three samples per task, prompt below one 4,608-token block so the prefix cache and the patch are not involved). Within every launch the three samples agreed. Across launches:

| Launch | Option | Counting / prose / code (tok/s, acceptance length) | Completions |
|---|---|---|---|
| 1 | on | 45.81 (3.70) / 28.08 (2.15) / 38.44 (3.10) | state 1 |
| 2 | off | 46.09 (3.70) / 28.49 (2.17) / 37.16 (3.00) | **state 2** |
| 3 | on | 45.73 (3.70) / 27.99 (2.15) / 37.88 (3.10) | state 1 |
| 4 | on | 45.68 (3.70) / 28.27 (2.15) / 38.38 (3.10) | state 1 |
| 5 | off | 45.79 (3.70) / 28.31 (2.15) / 38.45 (3.10) | state 1 |
| 6 | on | 45.62 (3.70) / 28.32 (2.15) / 38.49 (3.10) | state 1 |

The earlier image (1.8.0, one launch that night) computed in a third state. The option is not the axis: launch 5 with it off repeated launch 1 with it on, and launch 2 with it off did not. Speed and acceptance length are inside one launch's spread in every state, so the states differ in how greedy ties fall, not in quality or speed. What was compared and is not the cause: the two images (every Python and shared-library file hashed; only the patched `block_pool.py` and this repository's own files differ), the launch arguments (only the environment variable and the container name), the shared runtime cache on both ranks (no kernel or autotune table written during the six launches; only FlashInfer's zero-byte lock file is touched at each launch) and Triton's autotune tables (kept since 2026-09-15 for every autotuned kernel the path compiles; the per-launch JIT kernels are not autotuned). On the four-layer fixture (TP=1, in-process) six launches across both images and both settings were bit-identical. The remaining candidates decide at launch and leave no trace (the BF16 GEMMs cuBLAS serves, the FlashInfer plan, uninitialised memory, hash-seed-dependent ordering on the distributed path), so a launch is checked rather than assumed, and the decode check now keeps the completion token ids and both ranks' container logs so that the next launch in another state can be compared at its first diverging token (`tools/decode_check.py`, `tools/decode_divergence.py`; [after a switch](launch-safety.md#after-a-switch-the-decode-check)).

How rare the other states are, from the decode records kept on the head since 2026-09-18: the completions of one profile recur across days and record sets (the route g profile family with FA2 of 1.6.0, whose draft depth does not change the numerics, gave the same counting, prose and code completions in every record set from 2026-09-19 to 21, fourteen records of the counting hash across several launches), and the two states seen only once (the 1.8.0 image's and launch 2's) recur nowhere. Upstream treats one of the remaining candidates as a determinism setting: vLLM's batch-invariant mode sets `CUBLAS_WORKSPACE_CONFIG` itself (`vllm/model_executor/determinism/batch_invariant.py`), and PyTorch's reproducibility notes name the unset cuBLAS workspace as a run-to-run source; this stack does not set it, and will not until a launch in another state points at the BF16 GEMMs, because the setting changes their algorithm and with it the reference completions.

Two more checks on 2026-09-22 narrow the remaining candidates. In the reference image on one GB10, thirteen fresh processes computed the cuBLAS BF16 GEMMs at the pair's per-rank shapes (`lm_head` 4096 to 77,440 at 4 and 2,048 rows, the MTP projection, a square), the KDA chunk and recurrent kernels (32 local heads by 128) and the `lm_head` argmax on fixed inputs: every output bit-identical across the thirteen, including three processes with `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED` pinned, which moved nothing. Then the four-layer MTP fixture (stock weights, draft depth 3) was launched TP=2 across the pair twenty times with the production image, fabric and launch flags, production stopped for the hour: the decode check gave the same completions and token ids in all twenty launches for all three task types. The distributed path (two processes over RoCE, sharded sparse-MLA, indexer and Marlin MoE kernels, the draft) reproduced no launch-to-launch difference in twenty tries, so at the pair's observed rate of one launch in six to fifteen the remaining candidates are what only the full-size model does at load and launch.

A launch of 2026-09-23 (the two-sequence profile of [1.10.2](#measurements-on-1102), one restart) then computed in another state and was caught with the probe: the weight digest of both ranks equal to the previous launch's (2,382 tensors each), and the request trace naming the first differing call, the replicated kpool indexer of layer 19 on rank 1 at a decode verification step, in both the prose and the code request, with identical inputs and a different candidate set; in the previous launch that rank's indexer disagreed with rank 0's at exactly that call, in this one the ranks agreed. The counting request traced bit-identical. The indexer's own computations (the fp32 head gate, the fused FWHT quantisation, the pool cache's compress-and-write and decode tail update, DeepGEMM's paged MQA logits and the stable top-k) then ran in thirteen fresh processes on one GB10 at the served shapes on fixed inputs, every output bit-identical (`records/20260922-prefix-dedup/indexer-state/`). The next launch (Probe4, state 2 again) agreed with the earlier state-2 launch row for row in every trace and on every fixed-input kernel hash across the ranks, and a trace three levels deep with inputs showed the two ranks' replicated indexers receiving different keys at prefill in eight MLA layers from identical projection outputs (`trace-Probe4-prose160-deep-in/`). The launch after it (Probe5, the 1.11.1 checkout, state 1) agreed across the ranks on all sixteen hashes, the compiled layer norm with the layer's weights included, and received the same key on both ranks at every call; against the state-2 launch, rank 0's prefill was identical and rank 1's key differed from the first MLA layer (`trace-Probe5-prose160-deep-in/`). Thirteen launches of the serving image so far: nine in state 1, three in state 2, one in state 3. [Validation](validation.md#full-model-tp2-experimental-scope) states what that names and what remains.

The kernel was then named, and the states removed (2026-09-24 and 25; [validation](validation.md#full-model-tp2-experimental-scope) has the account). The served key norm is an Inductor persistent reduction with candidate configs `XBLOCK` 1, 8 and 32, chosen by timing on each rank at every launch; 1 differs in bits from the other two. Three more launches in the old way (Probe6, Probe7, Det1: states 1, 1 and 3) make sixteen checked launches: eleven in state 1, three in state 2, two in state 3. Pinning the config alone gave state 2 (8 on rank 0, 1 on rank 1) and state 1 (8 on both) on demand. With `runtime.inductor_deterministic` on the same two-sequence profile, both ranks served the kernel at `XBLOCK` 8 with a single candidate: three launches (2026-09-25), all in state 1 with the same completions; decode 27.95–28.44 / 46.05–46.27 / 38.39–38.80 tok/s on prose / counting / code, against 27.7–28.6 / 45.0–47.9 / 37.7–40.1 over the earlier state-1 launches; warmup's long rung 52.1–52.7 s (`records/20260924-inductor-autotune-asymmetry/`). The distributed defaults with the key, three launches: the same completions in all three, decode 32.80–32.84 / 20.76–20.82 / 27.53–27.64 tok/s on counting / prose / code against the defaults' 32.01 / 20.67 / 26.68 (`records/20260925-defaults-deterministic/`).

## Measurements on 1.10.2

### Two active sequences on the published option (2026-09-23)

The reference pair served the [AXL example's](../examples/server.axl.example.toml) settings (repacked weights, dedup, `max_num_seqs = 2`, 6 GiB of KV per rank; the served profile adds the memory probe and the dev routes), image `76a1172b…`, one launch (state 2 of [1.9.0](#six-launches-of-the-new-image-the-same-completions-five-times-different-once)). Every request below went to the running pair between 02:15 and 02:27 Asia/Tokyo with nothing restarted; a sampler read `/metrics` and the head's `MemAvailable` every two seconds (`records/20260923-two-sequence/`). No preemption occurred in any step.

| Step | Result |
|---|---|
| Two ~200K passphrase requests together (199,649 and 199,636 prompt tokens, different ledgers and passphrases, prefix cache reset first) | Both answered correctly; the first in 224.2 s, the second in 330.2 s; KV usage 54.2% at peak; the head kept 6.46 GiB available (7.16 before) |
| One of them alone, cache reset first | Answered correctly in 165.7 s; KV usage 35.7% at peak; 6.53 GiB available |
| Two tool-call requests together (time in Tokyo, weather in Osaka) | Both returned the right function with the right city, 1.41 and 1.21 s |
| An image (solid orange PNG, 288 image tokens) and the prose decode request together | "Orange"; the prose request decoded at 28.41 tok/s |

Two 200K requests together took 330 s against 166 s for one alone: the pair's throughput is conserved, neither gained nor lost. Decode with two sequences, the decode check's prompts (2,048 in, 512 out, three samples, median):

| Task | Alone (tok/s, acceptance length) | Together with the other task (tok/s) |
|---|---|---|
| counting | 45.62 (3.70) | 32.09 |
| prose | 28.24 (2.17) | 21.81 |

The acceptance length over the three concurrent runs was 2.79 for both tasks together. The completions are another matter. Alone, each request repeated bit for bit (three samples each, the state-2 hashes `fb15cfc2` and `1462d44f`). Together, both completions differed from the ones alone, two identical prose requests sent together returned two different texts (`25e9240a` and `c5930404`, 20.4 and 20.6 tok/s), and the count-and-prose pair gave one set of completions in runs 1 and 3 and another in run 2: with batch-invariant mode unavailable on this backend ([validation](validation.md#evidence-not-production-qualification)) a token's logits depend on which other rows share its step. The two-sequence profile therefore repeats a request only when that request runs alone.

## Measurements on 1.10.4

### sparkDash and tool-eval-bench on the two-sequence profile (2026-09-23)

The [1.4.0](#measurements-on-140) and [1.5.0](#sparkdash-and-200k-on-150) runs were repeated unchanged on the profile the reference pair serves (the [AXL example's](../examples/server.axl.example.toml) settings with the probe and the dev routes; image `76a1172b…`; the state-1 launch of 02:53 Asia/Tokyo, one sequence in flight), one after the other, nothing else running (`records/20260923-bench-1104/`). All 12 sparkDash streams succeeded with 128 tokens each.

| Prompt | Decode (token/s), median | Runs | TTFT (ms), median | 1.5.0 decode / TTFT |
|---|---:|---|---:|---|
| structured | 48.23 | 48.23 / 48.29 / 47.68 | 282.42 | 36.24 / 355.61 |
| prose | 31.38 | 30.53 / 31.40 / 31.38 | 225.33 | 26.68 / 369.03 |
| code | 41.28 | 41.37 / 41.28 / 38.64 | 442.04 | 31.67 / 570.67 |
| json | 34.88 | 34.80 / 34.94 / 34.88 | 273.00 | 26.25 / 444.05 |

sparkDash's protocol requests thinking off, which the fixed GLM template ignores, so these measure actual generation including reasoning, as before. The gain over 1.5.0 is the serving profile's since then (the repacked attention projections and `lm_head`, FA2 prefill, the split KDA projection, MTP k=3), not this version's; the head kept 7.24 GiB available.

tool-eval-bench `2.6.1.dev52+g81eae0a33`, the same version and settings as the [1.0.0 run](#tool-eval-bench) (all 69 standard scenarios, one trial, parallel 1, seed 42, temperature 0, effort low, clear_thinking, 4,096-token output budget, 600-second timeout, at most eight turns; 10.8 minutes): **88/100 (122/138 points)**, 55 pass, 11 partial, 3 fail, all 69 scored, completion 100%, no exclusion. The three failures are the same three as in 1.0.0 (TC-21 found two of five validation errors, TC-43 called web_search with an empty query, TC-61 did not attempt the analysis script), so the **Safety Gate is still not passed**, on TC-43. The partial results moved from 8 to 11 (unnecessary calculator use, an incomplete chain, an action not taken after a weather check, two injection scenarios answered safely but incompletely). One trial per version, so the two points against 1.0.0 are inside what one trial can move; the failures are stable across the two runs and the profiles between them, which changed the serving path and not the model's choices at temperature 0.

## Measurements on 1.13.0

### The kpool seed fix on both profiles (2026-09-25)

The rebuilt image `f53b563b…` (the 1.12 runtime plus `patch_kpool_seed`, vLLM pull request #57477) served both profiles on the reference pair: the published option's two-sequence profile with `runtime.inductor_deterministic` three times, and the distributed defaults with the key three times. Every switch completed without recovery. Before the first switch, the AXL checks marked "1.12" ran on the 1.12 image (`76a1172b…`) the same morning.

| Check | Published option (AXL) | Distributed defaults |
|---|---|---|
| Decode check, counting / prose / code | the 1.12 completions on all three launches; 45.97 / 28.47 / 38.63 tok/s | the 1.12 completions on all three launches; 32.75 / 20.79 / 27.57 tok/s |
| Teacher-forced NLL: Japanese / English / code / mathematics | 1.6270 / 1.9946 / 0.9601 / 0.6275, identical to the 1.12 image | 1.5963 / 2.0241 / 0.9479 / 0.5931, as published |
| Weight digest | — | equal to the 1.12 launch of 2026-09-25, 1,690 tensors per rank |
| Two sequences (below) | the same completions as on the 1.12 image | — |
| 199,652-token passphrase | 162.8 s, correct | 170.7 s, correct |
| 255,950-token passphrase | 214.7 s, correct | 218.8 s, correct |
| Capacity, 262,080 input + 64 output tokens, twice | finite logprobs | finite logprobs |
| Three-position reference, old framing, three runs | incorrect 3 of 3 (below) | correct 3 of 3 |
| Three-position reference, fenced framing, three runs | correct 3 of 3 (228–233 s, 36 output tokens) | correct 3 of 3 (233–235 s, 36 output tokens) |
| Prefill 38,962 tokens / decode after a short prompt | 1,281.3 / 41.9 tok/s | 1,287.7 / 27.43 tok/s |
| Mojibake check | passed | passed |

At these lengths the fix changed no output. The decode prompts and the NLL texts stay near or below 2,048 tokens, and none of the long requests reused a cached prefix: on this hybrid model the prefix cache hits only in whole blocks of 4,608 tokens (the requested 256 is raised so that KDA state pages align, as the boot log reports), so the decode checks never hit it. The case the upstream report reproduced, a long prompt reused from the prefix cache after other requests have cycled through the pool, was not measured here.

The old framing of the three-position reference is the one found [ambiguous on 1.4.0](#200k-real-input-on-140). On AXL all three runs wrote the three codes into their reasoning, then took "the REGISTRY values" to be the articles and began copying one until the 512-token limit; the defaults weighed the same reading and returned the codes. The same framing had scored 3 of 3 on AXL on 2026-09-22 with one active sequence, so which reading wins is a near tie that moves with the numerical path. The fenced framing puts the articles between `BACKGROUND` fences and asks for the value after `=` on each `REGISTRY` line; both profiles answered it with the codes at once. From 1.13.0 the fenced framing is the canonical form of this check, and the README row carries its results (medians 230.7 s on AXL and 234.2 s on the defaults, 261,573 prompt tokens). The earlier README values came from the explicit framing of [1.6.0](#measurements-on-160) (a record is one line). The request, with the same articles as the earlier checks, temperature 0, 512 output tokens and the prefix cache reset before each run:

```text
system: Read the supplied archive and return only the requested registry JSON.
user:   The archive below has exactly three REGISTRY lines, each of the form 'REGISTRY <key> = <value>'. Everything between the BACKGROUND fences is unrelated article text.
        REGISTRY begin = violet-bird-731
        ----- BACKGROUND (unrelated articles) -----
        <articles, 130,700 tokens>
        ----- END OF BACKGROUND (unrelated articles) -----
        REGISTRY middle = copper-fish-284
        ----- BACKGROUND (unrelated articles) -----
        <the same articles>
        ----- END OF BACKGROUND (unrelated articles) -----
        REGISTRY end = silver-tree-956
        Return the value after '=' on each REGISTRY line as a JSON object with keys begin, middle, end (for example {"begin": "...", "middle": "...", "end": "..."}). No other text.
```

The published option's NLL is lower on all four texts than on its one-sequence profile (1.6645 / 2.0024 / 1.0031 / 0.6279 on 2026-09-21): the two-sequence profile with the key gives 1.6270 / 1.9946 / 0.9601 / 0.6275, 0.1 to 4.3% lower, on the 1.12 image that morning and on the 1.13 image alike, so the gain predates the fix.

### Two sequences: what changes a completion (2026-09-25)

The decode check's three prompts were sent alone, both at once, and with the second sent only after the first had produced its first token, three times each, on the AXL two-sequence profile; then the same with 1,024-token prompts, whose contexts stay under 2,048 tokens so that the indexer selects every token. Alone, every request repeated bit for bit. Together, the completions differed from the solo ones, and a pair repeated whenever the server prefilled the two requests in the same order and cut its chunks at the same places; the only runs that differed from one another were those in which the first-token times showed a different order. A request whose prefill had finished before the other arrived still changed, mostly once the other's rows joined its decode steps, and the short prompts behaved the same, so neither the indexer nor the kpool seed is the cause; the 1.13.0 image gave the same hashes as the 1.12 image in every arrangement.

On one GB10, the same four rows (one decode step at depth 3) computed alone and in a larger call:

| Kernel | Next to four more rows (8 rows) | In front of a 2,044-row prefill (2,048 rows) |
|---|---|---|
| cuBLAS BF16 GEMM | same bits | same bits |
| Marlin W4A16 and NVFP4 dense GEMM | same bits | different |
| NVFP4 fused Marlin MoE (32 experts, top 4, routing fixed per row) | different | different |

The MoE is the one tested kernel whose rows change when another request's decode rows share the call. Attention and KDA were not tested.

## Measurements on 1.14.0

### What makes a request depend on another in the same step (2026-09-25)

Kernel runs on one GB10 (the serving image, no restart; `records/20260925-moe-batch/`) named two mechanisms, both of which choose how to split a sum from the size of the whole call:

- **The NVFP4 Marlin MoE** lays its (expert block, output tile) tiles out in block order and cuts the last ones along K, summing the pieces in fp32; where it cuts follows the number of expert blocks in the call. At decode sizes the kernel, its thread configuration and its grid stay the same (four and eight rows alike); only the block count moves. With a second request's four rows next to a request's four, 29 of 40 random routings and inputs changed at least one row. Pinning the launch and padding the blocks until no tile is cut made it 0 of 40, at 3 to 13% on the MoE call; the release does not adopt that (below).
- **The sparse-MLA decode** (FlashInfer 0.6.18, the 11 DSA layers and the MTP draft layer) lets each CTA take `chunks_per_block` of the 32 candidate chunks and picks the value from the call's token count: on 48 SMs 2 for one draft token and 3 for two, 6 for one verification step of four tokens and 15 for two. A second sequence therefore changed every row of a request's attention (all 128 rows of four tokens × 32 heads); with the value pinned, none.
- The KDA recurrent decode and the draft layer's BF16 Triton MoE gave the same rows with and without a partner.

`runtime.mla_decode_cpb` pins the second: the value comes from the tokens of one sequence (2 for a draft step, 6 for a verification step), the heuristic's own at one sequence. Through the patched backend on one GB10 a request alone computed bit for bit as before, and a call took 151 to 502 µs against 226 to 798 without the FlashInfer wrapper around it (synchronous timing).

### On the reference pair (2026-09-25 and 26)

Image `8444078038c0…` (source `b7cd765`), every switch complete without recovery.

| Profile | Decode check, counting / prose / code | Other checks |
|---|---|---|
| Published option, two sequences (the served profile) | the 1.13 completions, 46.16 / 28.38 / 38.66 tok/s; acceptance length 3.70 / 2.15 / 3.10 | mojibake passed; the same completions again after three more switches |
| Published option, `max_num_seqs = 1` | the 1.13 completions, 46.17 / 28.12 / 38.33 tok/s | — |
| Distributed defaults | the 1.13 completions, 32.75 / 20.75 / 27.51 tok/s | NLL 1.5963 / 2.0241 / 0.9479 / 0.5931, identical to 1.13.0 in every digit; 199,652-token passphrase correct in 166.7 s; prefill 38,962 tokens 1,294.8 tok/s, decode after a short prompt 27.38; mojibake passed |
| Distributed defaults, `max_num_seqs = 2` | the 1.13 completions, 32.59 / 20.72 / 27.55 tok/s | — |

Decode speed did not move: the wrapper time the key removes did not show in tokens per second on the pair.

The decode check's prompts were then sent in pairs (at once, and the second after the first's first token, each three times; 2,048- and 1,024-token prompts) against each profile's lone completions:

| Profile | Requests whose completion equals their lone one | Two 512-token completions together |
|---|---|---|
| Published option, two sequences | 0 of 18 at 2,048 tokens and 0 of 18 at 1,024; the pairs sent one after the other first differ at the same token as on 1.13.0 | 26.2 to 27.7 s, 37.0 to 39.1 tok/s together |
| Published option, `max_num_seqs = 1` | 18 of 18 (the second request queues: 13 to 22 s to its first token) | 33.2 to 34.1 s, 30.0 to 30.9 tok/s together |
| Distributed defaults, `max_num_seqs = 2` | 0 of 18 at each length | — |

With the attention split pinned, a request sharing steps with another still changes through the MoE and through the prefill-sized kernels of a step shared with a prefill, on both weights; a pair still repeats when it is sent in the same order. `max_num_seqs = 1` gives repeatable completions under any load, two sequences about a quarter more throughput when requests overlap.

A 19,851-token prompt with a passphrase at the midpoint was answered correctly cold, straight from the prefix cache (13,824 cached tokens: with a draft the lookup recomputes the last matching block) and again from the cache after three other ~20K-token prompts, with the same completion token for token each time: the kpool seed fix of 1.13.0 keeps a cached prefix intact while other prefills run.

Reported upstream: [flashinfer-ai/flashinfer#5553](https://github.com/flashinfer-ai/flashinfer/issues/5553) (the split follows the call's token count; still so on FlashInfer main), and the NVFP4 case on [vllm-project/vllm#46639](https://github.com/vllm-project/vllm/pull/46639) (Marlin MoE batch invariance, open).
