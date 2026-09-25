# Performance investigation

[日本語](performance-investigation.ja.md)

The [performance and quality catalog](optimization-catalog.md) owns the initiative list, dated baseline and next-version comparison fields. This document owns the detailed investigation procedures.

See the [performance and capacity Q&A](optimization-overview.md#performance-and-capacity-qa) for KV budgets, 256K/1M contexts, waiting times and concurrent serving.

These are measurement hypotheses, not diagnosed bottlenecks or promised speedups. Keep prefill latency, per-request decode latency and aggregate throughput separate. Nominal LPDDR bandwidth alone does not establish the cost of quantized MoE routing, sparse attention, scheduling or communication.

Measured component/full-target results and their limits are recorded in [component validation](component-validation.md).

## Lessons adopted from the A100 case study

[shi3z's case study](https://note.com/shi3zblog/n/nd5fc5341b342) motivates measuring launch overhead, conversion traffic, speculative execution and workload grouping. It concerns DeepSeek on A100; its throughput numbers are not GLM/GB10 targets. The decisions below are our application to this repository, not reproductions of that experiment.

| Idea | GLM/Spark decision and evidence needed |
|---|---|
| Reduce fine-grained launches | Already instrumented for investigation. Rank kernels by count and summed time; separate API launch overhead, NCCL, copies and actual kernel work. Validate any fusion against the reference before measuring unprofiled latency. |
| Avoid expanded intermediate weights | Inspect actual Marlin/dense traces for repeated unpack/dequantize/allocation/copy work. Preserve block/global scales and the current arithmetic contract. Repacking at load time is a candidate only if repeated work is measured. |
| MTP improves useful work per step | Compare off/k=1/k=3 with accepted tokens, verify time and per-request latency. A single sequence can already verify multiple speculative rows; MTP does not require max_num_seqs > 1 to gain GEMM-like work. Depths 1–5 measured; see [speculative decoding](speculative-decoding.md#depths-one-to-five-2026-09-19-and-20). |
| Group similar tasks | Ran as P14 ([task grouping experiment](#task-grouping-experiment)): not adopted. |
| CPU-offloaded experts / x86 integer kernels | Do not transplant the x86 VNNI/AMX path to this ARM/unified-memory system. CPU/GPU contention and actual copies need separate evidence. |
| Split into independent replicas | Excluded for the current two-node full checkpoint: one node cannot hold the model. PP is a distinct candidate with the implementation prerequisites below. |

For units and topology use [NVIDIA's hardware specification](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) and [network guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html): the advertised port rate is 200 **Gb/s** (25 GB/s before overhead), not 200 GB/s. The local measured collective rate is a different quantity. E2M1 code conversion alone is not an end-to-end equivalence proof for scaled NVFP4 operations.

## Kernel launches and synchronization

Set `profiling.enabled=true` in a dedicated server profile. The launcher enables the pinned vLLM Torch profiler and mounts a fresh per-container output directory under `records/profiles/`. Invoke the server's on-demand profiler around one warmed, exclusive request and collect both ranks' traces. Record the exact prompt and number of completed output tokens; keep cold JIT, prefill and speculative decode separate.

`python -m glm53_setup profile-assess <trace.json.gz>` counts GPU kernel events, CUDA launch API events and NCCL kernel events separately. Duration sums may overlap and are not end-to-end latency. Compare a one-output-token prefill control against the same prompt with longer fixed output to estimate additional launches per generated token. With MTP, also record acceptance and engine step counts; draft tokens are not accepted output tokens. Measure final latency with tracing disabled.

The report also ranks accumulated time by kernel name and separates memcpy events, known transferred bytes and copies lacking byte metadata. These are evidence of observed events only; they do not prove that a conversion buffer is unnecessary or that every allocation/read appears in the trace. Distinguish JIT compilation from repeated runtime launches.

Host synchronization calls are counted separately as well. With `runtime.index_checks = "sync"` the reference attention performs GPU-to-host index-range checks, and LPA reads positions on the host. Attribute their actual cost in traces before changing them; removing safety checks without an equivalent validated contract is not an optimization acceptance criterion.

Use traces to prioritize kernel work. CUDA events in LPA's layer profiler do not count launches. The reference attention uses Python-driven query chunks, and LPA requires eager mode; decode Graphs (P06) and a fused NoPE kernel (P04) were measured and not adopted ([catalog](optimization-catalog.md#performance-initiatives)).

## Throughput versus determinism

The first implemented launch-reduction candidate is `cache.fused_unpack`: one Triton kernel replaces intermediate FP8 copies, FP32 conversion, scale copies and multiplication when unpacking gathered MLA cache records. It leaves attention candidates and FP32 attention arithmetic unchanged. Its template default is owned by [server configuration](server-configuration.md#distributed-defaults) (on since the serial combination was accepted). Exhaustive FP8-code and scale tests are a component gate; fixture state comparisons and full-model unprofiled A/B runs remain separate acceptance gates.

For an isolated component measurement on the GPU image with current source mounted, run `python -m glm53_setup.validation.benchmark_unpack --output /path/to/new-record`. It checks exact output, excludes warmup, records five timing batches and profiles each path separately. On the tested GB10, 2,176 and 17,408 records each reduced from four kernels to one; observed median component times were approximately 0.045 to 0.011 ms and 0.630 to 0.212 ms respectively. These synthetic component sizes correspond to one and eight full candidate rows; they do not establish model speedup. Keep the full trace/result JSON with the image and source identity.

Use a separate no-LPA profile for `context.max_num_seqs > 1`; the LPA hook requires one contiguous sequence. A request that shares steps with another can get a different completion, for reasons that are known and scoped in [concurrency scope](validation.md#concurrency-scope); judge content, tool arguments, finish reasons, cross-request isolation, cancellation and resource safety independently of exact text. Report aggregate throughput and each request's latency/quality; do not compare aggregate rates to single-request decode.

## Task grouping experiment

**Ran as P14 and not adopted for this workload** ([result](benchmarks.md#task-grouping-order-comparison-p14)).

How it was compared: a fixed, pre-labeled workload of code, translation and summary tasks in a deterministic mixed ordering against same-task groups, with request IDs, input/output token budgets, concurrent sequence count, seed, cache policy, offered load and length distributions matched, on the no-LPA throughput profile, with single-request measurements as the latency control. Attributing a gain to shared weights would further need routed expert IDs counted at the real selection boundary in a separate diagnostic run; logits or task labels alone do not establish expert reuse.

## Expert Parallel (P21)

**Not adopted for the tested workload; default off (2026-09-13)** ([result](benchmarks.md#independent-expert-parallel-evaluation-p21)).

How it was compared: EP changed only the expert-layer partitioning; TP=2, DP=1, the two GPUs and the KV reservation stayed. The pinned FusedMoE config maps TP=2/DP=1 plus EP to two partitions of complete experts, and with DP/PCP/SP all one `use_all2all_kernels` is false, so the experiment neither invokes DeepEP nor removes all collectives ([vLLM EP description](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/#layer-behavior-with-ep-enabled)). The tenhkspark recipe's launch example motivated it; its other settings do not establish EP's benefit here.

A default-off [startup option](server-configuration.md) with an image marker was checked on a two-rank fixture (expert ownership, tensor shapes, output, restoration to EP off), then compared A/B/A on the full model against the two-sequence TP profile with everything else pinned (image, checkpoint, FP8 KV 1 GiB per rank, 16K, chunk 512, MTP/LPA/fusion/Graphs/APC off; short/2K input, 64 output, five repetitions per condition), followed by the task, tool, cancellation, restart and 16K×2 capacity checks. Adoption required a gain beyond the observed variation with quality, capacity, memory reserve and recovery intact, reported as aggregate throughput beside per-request TTFT/ITL/TPOT.

## TP versus PP

**Not adopted for the tested generation workload; default TP2 (2026-09-13)** ([results and limitations](benchmarks.md#independent-tp2-versus-pp2-evaluation-p17)).

The unmodified [fixed model source](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py) gates PP because `make_empty_intermediate_tensors` is missing, and its partial PP branch omits deferred mHC `post`/`comb` state. The repository's pinned patch implements those contracts, with an eight-layer fixture checking all four transferred tensors before full-model evaluation. The [server configuration](server-configuration.md) exposes an experimental PP2 profile with explicit layer partitioning. It requires eager execution, one sequence and EP/LPA/MTP/fusion/APC disabled.

For future PP comparisons, preserve identical precision, prompts and memory budgets; check layer allocation, KDA/MLA state, mHC boundaries, rank memory and failure recovery. Count actual communication events rather than assuming a layers-times-two collective count. MTP/LPA support and broader capacity qualification remain separate work.
