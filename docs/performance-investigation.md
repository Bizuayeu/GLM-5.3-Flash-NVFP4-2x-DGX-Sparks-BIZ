# Performance investigation

These are measurement hypotheses, not diagnosed bottlenecks or promised speedups. Keep prefill latency, per-request decode latency and aggregate throughput separate. Nominal LPDDR bandwidth alone does not establish the cost of quantized MoE routing, sparse attention, scheduling or communication.

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

## TP versus PP

TP=2 versus PP=2 is relevant for two memory-constrained nodes. It is **not currently a supported launch-option toggle** in this pinned GLM. The [fixed model source](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py) states that PP is gated off because `make_empty_intermediate_tensors` is missing. Its partial PP branch also omits deferred mHC `post`/`comb` state. LPA expects all language layers in one local module tree, adding a cross-stage representation issue.

Before adding a PP profile, implement/validate intermediate-state transport on a small fixture with LPA/MTP disabled. Check layer allocation, KDA/MLA state, mHC boundaries, rank memory and failure recovery. Only then compare the full model against TP with identical precision, prompts and memory budgets. MTP/LPA are later PP extensions. Count actual communication events rather than assuming a layers-times-two collective count.

No PP result, graph speedup or fused-kernel qualification is claimed by this plan.

[Indexer reuse/reindex](indexer-reuse.md) is a separate staged candidate: cost and overlap first, then request-scoped selection reuse with cache updates preserved. It must be evaluated jointly with LPA before claiming cumulative gains.
