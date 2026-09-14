# TP=2 benchmark method

[日本語](benchmarks.ja.md) · [Validation](validation.md)

This page records the MTP-off baseline and independent batching measurements. See the [MTP k=1 comparison](speculative-decoding.md#measured-k1-results) for the optional speculative profile.

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

Check the result independently; a zero CLI exit code can accompany zero successful requests:

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
- Memory: save each host's available-memory minimum and container memory current/peak/events. Each record states the container cap and host reserve that run used; those are historical conditions, and the distributed defaults live in [examples/startup.example.toml](../examples/startup.example.toml). A cgroup peak is not a model-memory counter and may omit GPU allocations; combine it with runtime and host observations.
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

**Capacity and measurement scope:** the shared KV budget remained fixed at 1 GiB per rank, not multiplied by concurrency. Speed comparisons covered **up to 2,048 input + 64 output = 2,112 tokens per request at two concurrent requests**. A separate [maximum-length capacity check](#maximum-length-capacity-check) also completed 16,384-token requests × two. Distinguish the runtime's reported 58,254-token capacity / 3.56 maximum-length concurrency estimate from observed coverage. See [KV capacity and RAM requirements](startup-configuration.md#kv-capacity-and-ram-requirements).

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

## Task grouping order comparison (P14)

On 2026-09-13 (Asia/Tokyo), the P13 two-sequence, chunk512 profile with optimizations off compared two code, two translation and two summary tasks. The same six tasks repeated three times in mixed, same-task-pair and restored mixed order. All prompts used the same template and padding to reach 512 tokens, verified against the server tokenizer. Each phase used C2, seed42, temperature0, six warmup requests and 128 fixed output tokens. This is a synthetic throughput test using padding and `ignore_eos`; full task-completion quality was not scored.

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

On 2026-09-13 (Asia/Tokyo), the full 45-layer model ran EP off→on→off with TP=2, DP=1 and two active sequences. All arms fixed source `ec9b86b`, image `sha256:7cb5f93f879da2c3cbbcadaf51d04d6567d778675033501f16afa645d43820a2`, Marlin W4A16, FP8 KV at 1 GiB per rank, context 16,384 and chunk512. MTP/LPA/fusion/Graphs/APC were off. Actual placement was inspected by RPC before timing; no layer-hash hooks or active profiler ran during benchmarks.

The fixed vLLM random benchmark used seed42, 64 output tokens, one warmup request and **five requests per unit of client concurrency**. All 30 measured requests per arm completed with valid output counts and finite metrics. This differs from P13's repetition count and must not be treated as the same run. Private records: `full-v28-ep-off`, `full-v28-ep-on`, `full-v28-ep-restored`.

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

Resource limits were 112 GiB per container and a 4 GiB host reserve. Minimum available RAM was about 11.47/12.77 GiB in the baseline and 11.61/12.57 GiB with EP on. No OOM occurred and subsequent restarts completed, but peer shutdown still required forced termination with exit137. Production recovery and long-running operation remain unqualified.

## Independent TP2 versus PP2 evaluation (P17)

Matched timing completed for TP2→TP1/PP2→restored TP2 using the same v28 image/checkpoint. Every arm used one active sequence, context16,384, chunk512, FP8 KV at 1 GiB per rank, Marlin W4A16 and eager execution; MTP/LPA/fusion/EP/APC were off. PP split layers 24/21, placing 21 MoE layers on each stage. Profiler support was configured but inactive during timing/quality tests. Workload and repetitions match P21. Private runs: `parallel-v30-tp-control`, `parallel-v30-pp2`, `parallel-v30-tp-restored`.

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

Before profiling, minimum head available memory under PP was about 4.27 GiB, including load/warmup, close to the 4 GiB experiment guard. It fell further after the failure. Keep normal TP2 and the standard 8 GiB reserve unchanged; the PP option remains experimental, with full-model capacity and recovery unqualified.

## Independent CPU synchronization reduction (P08)

The same v28 image ran synchronous→asynchronous→restored synchronous index checks with TP2, one sequence, eager execution, context16,384, chunk512 and FP8 KV at 1 GiB per rank. MTP/LPA/fusion/EP/APC/Graphs were off. Asynchronous execution retains range checking through a device assertion. Inputs came from the retained LLM-jp validation stream, not the test split.

After warmup, each one-output condition used five measurements and each 128-output condition three. Medians below cover the whole request; 128-output timings include prefill. Private run: `sync-v35-full`.

| Input tokens | Output tokens | Synchronous seconds | Asynchronous | Restored sync |
|---:|---:|---:|---:|---:|
| 64 | 1 | 0.3574 | 0.3580 | 0.3570 |
| 64 | 128 | 9.3019 | 9.0728 | 9.2734 |
| 2,048 | 1 | 6.1305 | 6.1100 | 6.1257 |
| 2,048 | 128 | 15.1617 | 14.8700 | 15.1425 |
| 8,192 | 1 | 24.6281 | 24.5649 | 24.6000 |
| 8,192 | 128 | 33.7452 | 33.3316 | 33.5738 |

Every arm passed eight task answers, two tool round trips and SSE disconnect/follow-up checks. **Accept as an opt-in within this independent eager scope.** The 128-output improvement over both controls was approximately 2.2–2.5% for short inputs, 1.8–1.9% at 2K and 0.7–1.2% at 8K. Prefill-only differences were small. Keep `index_checks="auto"` as the default (synchronous in eager); The serial MTP/LPA/fusion combination is evaluated separately under P18 below.

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

On 2026-09-13 (Asia/Tokyo), `full-integration-v36` compared LPA off/on/restored while holding MTP k=3, fused unpack and checked asynchronous indices fixed. Both ranks used image `sha256:2e5d0c49bc8f536364931f4599f169ea21108efcb33be65316912d88f240be9f`, source `5f0856d`, TP2, eager, one sequence, context16,384, chunk512 and FP8 KV1 GiB per rank. LPA used cut32/tail512 with the fixed affine projector; Graphs, EP, PP and APC were off. Profile fingerprint: `3c61cd10dd7dcff44f730d78eb78aa0d09e07d159f58b3f090fd2ea6b6ed3422`.

Median request time in seconds, with profiling disabled and a warmup excluded from each condition. One-output cases have five measured repetitions; 128-output cases have three.

| Input tokens | Output tokens | LPA off | Combined | Restored off |
|---:|---:|---:|---:|---:|
| 2,048 | 1 | 5.3306 | 4.5186 | 5.3291 |
| 2,048 | 128 | 10.6827 | 9.7582 | 10.7567 |
| 8,192 | 1 | 21.6461 | 17.5501 | 21.6590 |
| 8,192 | 128 | 25.6871 | 22.5424 | 26.2020 |

LPA added approximately 15.2%/18.9–19.0% improvement to the 2K/8K one-output controls, and 8.7–9.3%/12.2–14.0% to the corresponding 128-output requests. The latter include prefill; they are not decode-only speedups. These are measured combinations, not products of independent gains. MTP was active: the three combined 128-output samples at 2K/8K recorded 146/136 drafts, 438/408 proposed tokens and 236/245 accepted tokens.

The capture/off/oracle-full-MLP/oracle/off ladder produced identical 16-token texts. The 24 fixed tasks scored 22/23/24 under the strict formatting criteria, with no case where both native controls passed and the combined arm failed. The failures contained the correct numerical answer with unwanted explanation/formatting; they remain failures. All three long-context tool round trips passed, and worker reports confirmed the expected historical-query skips at layers35/39/43 only with LPA enabled. This is limited task evidence, not unrestricted equivalence or general quality qualification.

`integration-ops-v38` separately completed 16,320 input plus 64 output tokens in all three arms, with no preemption, empty scheduler/KV afterward and a successful following request. Peak KV usage was approximately 0.630. Every arm also passed a 2K-input SSE disconnect after three nonempty chunks, early generation stop at 9 tokens and a following request. LPA-on worker reports confirmed active approximation during both capacity and cancellation tests. The [FreedomBench recheck and six-question long-prefix pilot](freedombench.md#integration-recheck-and-long-prefix-pilot) are reported separately.

The whole run, including load, used a 112 GiB container cap and a 4 GiB host reserve. Minimum available RAM was 8.278/9.330 GiB on rank0/1. Both containers stopped without OOM; head exited0 and peer137 after explicit stop. **Accept this serial combination for the measured experimental scope.** Default settings and the normal 8 GiB reserve remain unchanged. This does not qualify Graphs, APC, batching/MTP combinations, larger contexts, sustained load, harness integration or production recovery; no routine-deployment receipt is issued.

## Independent context sweep through 32K (P15)

`context-v40-32k` completed on 2026-09-13 (Asia/Tokyo) using the same V36 image, TP2/eager, one sequence, chunk512 and fixed FP8 KV1 GiB per rank. Context was 32,768; MTP/LPA/fusion/async checks/Graphs/APC/EP/PP were off. The 112 GiB container cap and 4 GiB host reserve remained active. Profile fingerprint: `3c03f3bbdb67c8c766b7430456583e4b5df5517c61300bc83bc93a61cd8f3e8e`.

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

`apc-full-v43-off/on/restored` completed with the V36 image, TP2/eager, one sequence, context 32,768, chunk512 and fixed FP8 KV1 GiB per rank. MTP/LPA/fusion/async checks/Graphs/EP/PP were off; only APC changed. The three profiles and all compared token arrays were checked for equality after normalizing that one flag. APC-on fingerprint: `f8e3c6ea72dfc1c61afdb2fcd67b1432e2fbcf021dd0613bda0dfb90ab223823`.

Here, cold means a new prefix with model weights already loaded. Fixed unique leading identifiers produced cache misses; an immediate identical request tested reuse. Each length had one excluded warmup pair and three measured cold/repeat pairs: 18 measured requests per arm, all with one output token. Periodic cache/generation metrics were allowed to settle outside the latency interval.

| Input tokens | Repeated request, APC off | APC on | Restored off | Cached tokens with APC on |
|---:|---:|---:|---:|---:|
| 2,048 | 6.0698 s | 6.0768 s | 6.0445 s | 0 |
| 8,705 | 26.0405 s | 0.0937 s | 25.9420 s | 8,704 |
| 16,320 | 48.8203 s | 10.0633 s | 48.6927 s | 13,056 |

The near-complete 8,705-token hit leaves only one input token to process; its approximately 99.64% reduction is that specific boundary case. At 16,320, reuse reduced request time by approximately 79.3–79.4% against both controls. Every measured cached long request was faster than every corresponding off-control sample. There was no benefit at 2K. APC cold medians were 6.0762/26.3284/49.5088 s; the two long cold cases were approximately 1.1–1.7% slower than their controls. This is a cache-hit optimization, not faster uncached processing or decode.

All three arms passed six long extraction/revisit requests, a long tool round trip and 12K-input SSE disconnect/follow-up. The 12,763-token document requests deliberately interleaved distinct documents; all three APC-on revisits had 8,704 cached tokens and correct answers. The tool-result continuation also reused 8,704 tokens and returned the expected value. All cold-prefix requests had zero hits. These are limited task checks, not general language-quality certification.

Container caps were 112 GiB with 4 GiB host reserves. Whole-run RAM minima for off/on/restored were 11.445/11.489/11.495 GiB on head and 12.726/12.620/12.586 GiB on peer; the first off run also includes the 32K sweep. Every head stopped with exit 0, every peer with exit 137, and none was OOM-killed. **Accept APC for the measured repeated-long-prefix, serial experimental workload; default remains off.** APC 32K request capacity, MTP/fusion/Graphs combinations and broad business-use qualification are not established by the configured context limit. LPA coexistence is evaluated separately under the [P22 exact-cache contract](apc-lpa-design.md).

## APC-first LPA crossover measurement (P22)

On 2026-09-13 (Asia/Tokyo), `p22-calibration-v51` and `p22-short-calibration-v54` used the same server image `sha256:8cb2babff5524c808d112c3340e7d6f6cbbc84e6f5840a9cad66b1c2bc7dfd26` (source `4b83f11`), profile fingerprint `7e62b6e4de703532c04af9218d3ced7b512db0bd66994f5a4aea717bfd95fd18`. TP2/eager, one sequence, context32,768, chunk512, FP8 KV1 GiB/rank and APC were fixed. LPA used cut32/tail512 and the existing affine projector; MTP, fusion, async checks, Graphs and tracing were off. The calibration threshold was deliberately zero.

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

The combined run also completed 32,704 input + 64 output tokens in all three arms without preemption (80.596/64.689/80.593 s). A cold approximate request left H=0 for the following ordinary request; only ordinary recomputation made H=9,216 reusable. Disconnecting after three nonempty SSE chunks stopped at 9 generated tokens and a follow-up request completed. Native HTTP400 handling for invalid LPA options and reserved-policy injection passed, as did R=127/128/129 boundary checks. Actual speculative cursor corrections were observed in full-model generation. These are scoped functional/operational results; final held-out retrieval and production qualification are separate.

### Repeated-input tradeoff

After selecting the profile and threshold, `p22-heldout-v63` used eight previously unused test-split documents (four marker positions × two text lengths): ordinary 7/8, LPA 8/8, restored 8/8, with no LPA-only regression. These documents were not used for training or threshold selection. Across the combined and held-out run, minimum host availability was 8.987/10.239 GiB on rank 0/1. Both ranks were stopped after completion, without OOM. This eight-task result is a scoped retrieval check, not a general quality guarantee.

`p22-reuse-single-v56` and `p22-reuse-combined-v62` first computed the full prompt normally, then generated 128 tokens from the identical prompt six times, excluding the first repeat from timing. No approximation-derived cache was shared. All five measured repeats retained the H shown below.

| Input | P22 without MTP/fusion/async, KV 1 GiB | Full combined profile, KV 2 GiB |
|---:|---:|---:|
| 8,192 | 18.532 s; H=4,352 | 22.356 s; H=0 |
| 16,320 | 17.361 s; H=13,056 | 22.294 s; H=9,216 |

The combined profile was 20.6%/28.4% slower in these exact-primed, 128-output cases. Its MTP replay boundary reused less input. This compares whole profiles, including different cache budgets and exact-kernel settings; it is not an isolated causal estimate of MTP cost. **Keep the serial decode-oriented MTP option and the prefix-reuse-oriented no-MTP P22 option separate.** Do not advertise all enabled flags as universally fastest, or infer the same crossover for much longer generated answers.

## APC history retention baseline

`history-single-v68` extended P19/P22 with first edits and branches at 10/50/90%, appends, alternating conversations, eviction pressure and actual pool/block boundaries. It used fixed image `sha256:569538ce8b1c259f3ee13242f387320417b2d63a0b4122b6dc74d92ae74dae33`, TP2/eager/one sequence, context32K/chunk512, FP8 KV1 GiB per rank, no MTP/fusion/async checks, and LPA cut32/tail512/B128. The pinned native retention interval was 0. Exact priming preceded each exact/LPA/restored variant; only validation-split corpus data was used.

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

`p22-final-regression-v73` and `history-final-combined-v73` used source `f3167d4`, image `sha256:e12070943ced2ef145a565416a1b50bcfefa7d258593810c89a97650de10b7f6`: TP2/eager/one sequence, 32K/chunk512, KV2 GiB/rank, MTP3, fused unpack, asynchronous index checks, APC, LPA cut32/tail512/B128 and native `dense` checkpoint retention. This image does **not** include the subsequently added [canonical candidate ordering](candidate-order.md).

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

Stock DecodeBench from sparkDash commit `e03b9d624e7135d6e82b4c8fc94ea0ddcf300547` targeted the local GLM over HTTP: concurrency1, 128 output tokens, temperature 0/top_p 1, a 32-token warmup per job, four prompt types and three runs each. All 12 streams succeeded and each generated 128 tokens. Values below are medians over three runs.

| Prompt | Decode (token/s) | TTFT (ms) |
|---|---:|---:|
| structured | 33.92 | 382.93 |
| prose | 23.32 | 407.78 |
| code | 29.28 | 691.94 |
| json | 24.03 | 493.94 |

Decode uses sparkDash's first-to-last-token window and usage counts. Its stock protocol requests thinking off, but the fixed GLM template ignores those flags. Do not describe this as non-thinking or effort-low measurement; it measures actual generation including reasoning. Measurement code was unchanged; host-monitoring adjustments are separate.

### tool-eval-bench

Version `2.6.1.dev52+g81eae0a33` (commit `81eae0a3345eb212526cd98a2dd30a5088b74b0c`) ran all 69 standard scenarios, one trial, parallel 1, seed 42, temperature 0, effort low, clear_thinking=true, a 4,096-token output budget, 600-second request timeout and at most eight turns. Overall score: **90/100 (124/138 points)**; 58 pass, 8 partial and 3 fail. All 69 were scored, completion100%, no infrastructure exclusions. Optional Hard Mode was not included.

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
| Maximum capacity | 204,736 | 64 | 473.047 s | Completed204,800 total tokens; a 64-output, ignore-EOS capacity probe with finite generated-token logprobs |
| Three-position reference | 200,095 | 83 | 454.082 s | All three identifiers at beginning/middle/end correct; stop finish |

Both checks had zero additional preemptions and no running/waiting requests after completion. On both ranks, LPA skipped 204,224/199,583 queries respectively at each of layers35/39/43, proving approximation actually ran. A short arithmetic follow-up also passed; both supervisors and the API remained running. Two-second memory samples covering loading, all suites and final verification reached minima of 5.198/6.287 GiB available. There was no reserve stop or OOM.

These results supported the earlier distributed serial defaults of 200K, KV 2.5 GiB per rank, reserve 4 GiB and no deadline. Each long check is one capacity/limited-reference trial; it does not qualify numerical identity, every 200K history-edit pattern, multiple sequences, general quality or long-term reliability. Times include prefill and are whole-request latencies, not warm identical-prefix reuse speeds.

Earlier tuning attempts with KV 4 GiB/reserve 5 GiB stopped during initialization at minimum availability4.945/4.984 GiB; KV 3 GiB/reserve 5 GiB stopped near initial generation at4.962 GiB on the head. These were reserve stops, not OOM. The earlier TLS-mismatched sparkDash jobs and disconnected tool-eval run are preserved as invalid measurements and excluded from the valid results above.

### Real-input checks at 256K

On 2026-09-14 (Asia/Tokyo), the existing combined image `sha256:f6fc154c5b5397e694fb20b9c150bced6b1a049dd5e4b1bf6a7ca642160def7a` was restarted at **262,144 input-plus-output tokens with FP8 KV 3 GiB per rank**. Profile fingerprint: `4bd8232fc2af58e8938c7a4b99f3c13f90126e8da38e2ddf43a5e2100b4931f6`. Only context and KV changed from the 200K run above: TP2/eager/one sequence, chunk512, MTP3, LPA cut32/tail512/B128, APC/dense retention, fused unpack, async index checks, the 4 GiB host reserve and no lifetime deadline were retained.

The runtime reported KV capacity of 301,645 tokens. The same pinned LLM-jp validation corpus and method were reused, resetting prefix cache before each long request; both ranks reported zero cached-prefix tokens. Each case ran once, with a request timeout of 1,800 seconds.

| Check | Input tokens | Generated tokens | Whole-request seconds | Result |
|---|---:|---:|---:|---|
| Full capacity | 262,080 | 64 | 581.961 | Exactly 262,144 total tokens; forced 64-token generation with finite logprobs |
| Three-position retrieval | 261,595 | 131 | 565.989 | All three identifiers recovered; normal stop |

Neither request increased preemption. A short arithmetic request passed afterward; both ranks and the API remained running, with no OOM or memory-guard stop. Two-second supervision from launch through these checks recorded minimum available RAM of **4.162 / 5.183 GiB** (head/peer).

These scoped checks supported the then-distributed **256K / 3 GiB-per-rank** defaults, now the text-only alternative; the current defaults with image input are recorded in [image input at 200K](vision.md). They do not rerun or transfer the earlier 200K speed, tool-eval or FreedomBench scores to this profile, or qualify general long-context quality, every history-edit pattern, multiple sequences, actual harness behavior or long-term reliability. The test timeout is separate from client defaults; long cold requests need enough client waiting time.
