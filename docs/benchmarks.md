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
- Memory: save each host's available-memory minimum and container memory current/peak/events. A cgroup peak is not a model-memory counter and may omit GPU allocations; combine it with runtime and host observations.
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

**Decision:** accept independent seqs=2 within the tested 32/2,048-input, 64-output-token workload at at most two concurrent requests: roughly 76–78% more short-input aggregate output and 26% more at 2K versus the serial controls. Retain seqs=1 as the default and latency control. LPA remains incompatible with this multi-sequence profile. MTP/fusion/Graphs combinations, four sequences, long-context quality/performance, sustained load and recovery require separate gates. This does not grant enterprise or routine-service qualification.

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
