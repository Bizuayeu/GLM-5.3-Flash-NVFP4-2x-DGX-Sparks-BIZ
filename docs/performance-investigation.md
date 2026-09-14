# Performance investigation

[日本語](performance-investigation.ja.md)

The [performance and quality catalog](optimization-catalog.md) ([日本語](optimization-catalog.ja.md)) owns the initiative list, dated baseline and next-version comparison fields. This document owns the detailed investigation procedures.

See the [performance and capacity Q&A](optimization-overview.md#performance-and-capacity-qa) for KV budgets, 200K/1M contexts, waiting times and concurrent serving.

These are measurement hypotheses, not diagnosed bottlenecks or promised speedups. Keep prefill latency, per-request decode latency and aggregate throughput separate. Nominal LPDDR bandwidth alone does not establish the cost of quantized MoE routing, sparse attention, scheduling or communication.

Measured component/full-target results and their limits are recorded in [component validation](component-validation.md).

## Lessons adopted from the A100 case study

[shi3z's case study](https://note.com/shi3zblog/n/nd5fc5341b342) motivates measuring launch overhead, conversion traffic, speculative execution and workload grouping. It concerns DeepSeek on A100; its throughput numbers are not GLM/GB10 targets. The decisions below are our application to this repository, not reproductions of that experiment.

| Idea | GLM/Spark decision and evidence needed |
|---|---|
| Reduce fine-grained launches | Already instrumented for investigation. Rank kernels by count and summed time; separate API launch overhead, NCCL, copies and actual kernel work. Validate any fusion against the reference before measuring unprofiled latency. |
| Avoid expanded intermediate weights | Inspect actual Marlin/dense traces for repeated unpack/dequantize/allocation/copy work. Preserve block/global scales and the current arithmetic contract. Repacking at load time is a candidate only if repeated work is measured. |
| MTP improves useful work per step | Compare off/k=1/k=3 with accepted tokens, verify time and per-request latency. A single sequence can already verify multiple speculative rows; MTP does not require max_num_seqs > 1 to gain GEMM-like work. Higher depth is a new experiment, not an assumed improvement. |
| Group similar tasks | Add the controlled homogeneous-versus-mixed batch experiment below before implementing a semantic scheduler. |
| CPU-offloaded experts / x86 integer kernels | Do not transplant the x86 VNNI/AMX path to this ARM/unified-memory system. CPU/GPU contention and actual copies need separate evidence. |
| Split into independent replicas | Excluded for the current two-node full checkpoint: one node cannot hold the model. PP is a distinct candidate with the implementation prerequisites below. |

For units and topology use [NVIDIA's hardware specification](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) and [network guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html): the advertised port rate is 200 **Gb/s** (25 GB/s before overhead), not 200 GB/s. The local measured collective rate is a different quantity. E2M1 code conversion alone is not an end-to-end equivalence proof for scaled NVFP4 operations.

## Kernel launches and synchronization

Set `profiling.enabled=true` in a dedicated startup profile. The launcher enables the pinned vLLM Torch profiler and mounts a fresh per-container output directory under `records/profiles/`. Invoke the server's on-demand profiler around one warmed, exclusive request and collect both ranks' traces. Record the exact prompt and number of completed output tokens; keep cold JIT, prefill and speculative decode separate.

`python -m glm53_setup profile-assess <trace.json.gz>` counts GPU kernel events, CUDA launch API events and NCCL kernel events separately. Duration sums may overlap and are not end-to-end latency. Compare a one-output-token prefill control against the same prompt with longer fixed output to estimate additional launches per generated token. With MTP, also record acceptance and engine step counts; draft tokens are not accepted output tokens. Measure final latency with tracing disabled.

The report also ranks accumulated time by kernel name and separates memcpy events, known transferred bytes and copies lacking byte metadata. These are evidence of observed events only; they do not prove that a conversion buffer is unnecessary or that every allocation/read appears in the trace. Distinguish JIT compilation from repeated runtime launches.

Host synchronization calls are counted separately as well. The current reference attention performs GPU-to-host index-range checks, and LPA reads positions on the host. Attribute their actual cost in traces before changing them; removing safety checks without an equivalent validated contract is not an optimization acceptance criterion.

Use traces to prioritize kernel work. CUDA events in LPA's layer profiler do not count launches. The reference attention uses Python-driven query chunks, and LPA requires eager mode; CUDA graphs or a fused NoPE kernel are separate correctness projects requiring full-candidate, cache/state and quality checks.

## Throughput versus determinism

The first implemented launch-reduction candidate is `cache.fused_unpack`: one Triton kernel replaces intermediate FP8 copies, FP32 conversion, scale copies and multiplication when unpacking gathered MLA cache records. It leaves attention candidates and FP32 attention arithmetic unchanged. The default is off. Exhaustive FP8-code and scale tests are a component gate; fixture state comparisons and full-model unprofiled A/B runs remain separate acceptance gates.

For an isolated component measurement on the GPU image with current source mounted, run `python -m glm53_setup.validation.benchmark_unpack --output /path/to/new-record`. It checks exact output, excludes warmup, records five timing batches and profiles each path separately. On the tested GB10, 2,176 and 17,408 records each reduced from four kernels to one; observed median component times were approximately 0.045 to 0.011 ms and 0.630 to 0.212 ms respectively. These synthetic component sizes correspond to one and eight full candidate rows; they do not establish model speedup. Keep the full trace/result JSON with the image and source identity.

Use a separate no-LPA profile for `context.max_num_seqs > 1`; the LPA hook requires one contiguous sequence. Start with two and then four active sequences only if resource and task checks pass. A near-tie greedy divergence is a numerical/reproducibility diagnostic, not automatically a task-quality failure. Preserve it while judging content, tool arguments, finish reasons, cross-request isolation, cancellation and resource safety independently. Report aggregate throughput and each request's latency/quality; do not compare aggregate rates to single-request decode.

## Task grouping experiment

Use a fixed, pre-labeled workload of code, translation and summary tasks. Compare a deterministic mixed ordering with same-task groups, preserving request IDs, input/output token budgets, concurrent sequence count, seed, cache policy and offered load. Match length distributions so padding or short-input bias cannot masquerade as semantic reuse. Begin with the no-LPA throughput profile; single-request measurements remain the latency control.

Report per-request quality and queue delay, aggregate throughput, completion coverage and actual batch overlap. In a separate diagnostic run, count routed expert IDs at the real selection boundary if the pinned runtime exposes them; logits or task labels alone do not establish expert reuse. Compare distinct experts per layer/batch and repeat the workload ordering before attributing a speedup to shared weights. A future grouping scheduler must also bound starvation and preserve cancellation, tenant isolation and latency requirements. Do not add semantic routing or prompt rewriting before this experiment establishes a benefit.

## Expert Parallel (P21)

**Status: full-model independent A/B/A completed; EP not adopted for the tested workload, default off (2026-09-13).** See [measurements and scoped quality/capacity results](benchmarks.md#independent-expert-parallel-evaluation-p21). This is separate from the accepted [two-active-sequence batching experiment](benchmarks.md#independent-active-batching) and from distribution-only startup acceptance on another node pair. The procedure below records how the comparison is made.

The pinned FusedMoE parallel config maps TP=2/DP=1 plus EP to two partitions of complete experts. Marlin's declared parallel support accepts this configuration, and its execution path accepts an expert map. With DP/PCP/SP all one, `use_all2all_kernels` is false even with EP enabled; do not assume this experiment invokes DeepEP or removes all collectives. These are source-level compatibility findings, not proof of successful loading, ownership or numerical behavior. The [startup option](startup-configuration.md) requires a matching image marker and rejects untested optimization combinations.

Keep TP=2 and DP=1 on the same two GPUs. The intended change is expert-layer partitioning, not an additional model replica or a larger KV reservation. The [vLLM EP description](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/#layer-behavior-with-ep-enabled) explains this topology; actual support must be checked in the pinned runtime. The [tenhkspark launch example](https://github.com/tenhkspark/glm53-flash-nvfp4-dgx-spark/blob/main/serve/start-head.sh) motivates the experiment, but its native attention, other settings and aggregate throughput do not establish EP's independent benefit here.

1. **Compatibility and implementation:** inspect the pinned GLM/FusedMoE loader, Marlin NVFP4 EP support, expert placement and communication backend on ARM64/SM121. If unsupported, retain the error and assess the smallest compatible change separately; do not silently switch precision, attention candidates, executor or dependencies. Implement an explicit default-off startup setting, both-rank arguments, configuration fingerprint and preflight checks using the existing startup configuration. CPU contracts must verify off preserves the current command, on selects the intended EP topology, and incompatible settings fail. Do not add redundant experts/EPLB or DP replicas to this experiment.
2. **Component gate:** use a small two-rank fixture to verify expected expert ownership, compatible loaded tensor shapes/bytes, output/state behavior, real transport and restoration to EP off. A startup flag or import is not proof that EP ran. Record per-rank persistent memory and temporary/communication buffer allocations.
3. **Full-model A/B/A:** A is the qualified two-sequence TP profile, B enables EP only, then restore A. Pin the same candidate image, checkpoint, Marlin W4A16, all-candidate NoPE attention, FP8 KV at 1 GiB per rank, 16K context, chunk 512 and two active sequences. Keep MTP/LPA/fusion/Graphs/APC off. Reuse the existing short/2K-input, 64-output workload with one/two clients; use the plan's five measured repetitions per condition after warmup. Keep the same prompts, sampling, reasoning effort, output budgets and offered load. Preserve all repetitions, errors and real concurrency observations.
4. **Quality, capacity and recovery:** repeat the existing text/extraction and distinct-ID tool round trips, then SSE/cancellation and a following request, controlled stop/restart and the separate 16K×2 capacity check. Record correct values, format, termination and numerical diagnostics separately. Compare each rank's minimum available RAM, peak allocation, swap activity, KV usage/preemption and communication cost. A fixed KV pool does not bound every other allocation.
5. **Decision:** adopt for the measured throughput workload only if the gain is repeatable beyond observed variation and quality, capacity, memory reserve and recovery criteria still pass. Report aggregate throughput together with per-request TTFT/ITL/TPOT; a throughput gain can worsen individual latency. Record the permitted additional-memory budget and latency tradeoff before the full comparison, using the actual pair's baseline and configured reserve rather than invented limits. If gains remain within noise or require unacceptable memory/latency, retain batching with EP off. Broader concurrency and MTP/LPA integration remain separate experiments; do not multiply their individual gains.

Raw evidence belongs in a fresh private `records/<run-id>/`; publish only reviewed summaries and update P21 with the actual result. The linked benchmark report owns the EP result; this procedure does not qualify other workloads or combinations.

## TP versus PP

**Status: independent full-model timing A/B/A completed; PP not adopted for the tested generation workload, default TP2 (2026-09-13).** PP improved prefill but slowed decode. Limited task, tool and disconnect checks passed; PP capacity and decode tracing remain incomplete after a profiler restart failure. See the [results and limitations](benchmarks.md#independent-tp2-versus-pp2-evaluation-p17).

The unmodified [fixed model source](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py) gates PP because `make_empty_intermediate_tensors` is missing, and its partial PP branch omits deferred mHC `post`/`comb` state. The repository's pinned patch implements those contracts, with an eight-layer fixture checking all four transferred tensors before full-model evaluation. The [startup configuration](startup-configuration.md) exposes an experimental PP2 profile with explicit layer partitioning. It requires eager execution, one sequence and EP/LPA/MTP/fusion/APC disabled.

For future PP comparisons, preserve identical precision, prompts and memory budgets; check layer allocation, KDA/MLA state, mHC boundaries, rank memory and failure recovery. Count actual communication events rather than assuming a layers-times-two collective count. MTP/LPA support and broader capacity qualification remain separate work.

[Indexer reuse/reindex](indexer-reuse.md) is a separate staged candidate: cost and overlap first, then request-scoped selection reuse with cache updates preserved. It must be evaluated jointly with LPA before claiming cumulative gains.
