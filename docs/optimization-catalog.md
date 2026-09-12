# Enterprise performance and quality catalog

[日本語](optimization-catalog.ja.md) · [Project purpose](../README.md#enterprise-use-objectives) · [Measurement procedures](performance-investigation.md)

This catalog records what to investigate, what has been measured, and why a candidate was retained or deferred, so the next GLM version can be evaluated against an explicit baseline. It owns initiative IDs and reevaluation criteria; linked documents own measurements, detailed test contracts and runtime settings. Listing a candidate does not mean it is implemented or available as a startup option.

## Baseline and source ownership

**Initial baseline: 2026-09-12 (Asia/Tokyo), documents at commit `9997ccc`. Status rows include subsequent validation through 2026-09-13.** Ongoing integration runs and uncommitted prototypes are not passing results. Follow later results through the linked sources and Git history; retain the document commit and run ID when comparing versions.

| Source | Owns |
|---|---|
| [README](../README.md) | Enterprise purpose and entry points |
| This catalog and its Japanese counterpart | Stable IDs, intent, dated status and reevaluation criteria |
| [Performance investigation](performance-investigation.md) | Launch/synchronization, task grouping and TP/PP measurement procedures |
| [Startup configuration](startup-configuration.md) / [runtime lock](../config/runtime.lock.json) | Setting semantics, supported scope and fixed assets; configurable limits are not qualified limits |
| [Baseline benchmarks](benchmarks.md) / [MTP](speculative-decoding.md) / [LPA](lpa.md) / [CUDA and indexer measurements](component-validation.md) | Numerical results, experimental conditions and limitations |
| [FreedomBench](freedombench.md) / [validation](validation.md) / [harnesses](harnesses.md) | Quality, functionality and enterprise acceptance contracts |
| Private `records/<run-id>/` | Raw responses, traces, all repetitions, failures and site details; public documents contain verified summaries |

The initial reference is two GB10 nodes, TP=2, Marlin W4A16, FP8 KV, candidate-preserving NoPE reference attention, eager execution and one active sequence, with MTP/LPA/fusion/APC off. **Later experiments do not necessarily share the same image, inputs or resource budgets: do not add or multiply gains from different runs.** The tested W4A16 arithmetic is distinct from the original W4A4 recipe.

The observed division of benefit is primarily decode for MTP, long-input prefill for LPA, and prefill for current CUDA unpack fusion. Fusion is a first step at KV reconstruction, not completion of attention-body fusion or CUDA Graphs. CSA2 components are retained while model-serving integration is deferred.

**P06 update after the baseline (2026-09-12):** Independent Graph startup settings and a candidate image are implemented. A truncated fixture without fusion/LPA/MTP demonstrated capture/replay but diverged from eager output tokens at 2K input. Numerical acceptance and full-model progression are paused; eager remains the default. See [observations, pinned assets and resumption criteria](component-validation.md#independent-decode-graph-fixture). This update is separate from the dated baseline rows below.

**P13 update:** Independent 1→2→1-sequence comparisons established throughput improvement and limited task agreement for the short/2K-input, 64-output-token workload with two concurrent requests. A separate capacity check completed two concurrent 16,320-input/64-output requests. **Keep performance, quality and capacity scopes distinct; changing configured limits does not automatically enlarge KV reservation.** See [measurements and capacity conditions](benchmarks.md#independent-active-batching).

**P05 decision: not adopted.** Candidate width 2176 was unsupported for decode, numerical checks failed, and the prefill-only probe rejected its first shape. Close this backend-selection attempt and move to other initiatives; reopen after relevant upstream support changes. No candidates were truncated to fit it. See the [component probe](component-validation.md#direct-padded-native-attention-probe).

## Performance initiatives

Expected effects are hypotheses. A measured result applies to its documented conditions, not automatically to production, language quality or combined configurations. Preserve IDs across model versions, including unsuccessful candidates.

| Initiative | Work | Expected effect / metric | Baseline status / evidence |
|---|---|---|---|
| P01 Standard MTP, three speculative tokens | Use the checkpoint's BF16 draft without an external draft model; compare off/k=1/k=3 | Fewer target steps per useful token; decode gains versus draft/verify time, acceptance and extra memory | **k=3 selected from measurements.** Enabling MTP is opt-in; other depths and multiple active sequences unqualified. [MTP](speculative-decoding.md) |
| P02 LPA | Predict later attention inputs; skip past-token MLP and unused queries, retaining an exact tail and all-layer decode | Lower long-input prefill/TTFT; assess quality effects of approximate state, not an assumed decode gain | **Measured experimental path.** Limited long-task checks; broader quality and integration are separate gates. [LPA](lpa.md) |
| P03 CUDA fusion: KV reconstruction | Fuse FP8 MLA-cache copies, FP32 conversion and scale multiplication with Triton | Fewer launches/intermediates, primarily faster prefill | **Exact components and full-target A/B/A measured.** About 17% at the 8K one-output control; little short-context decode change. Default off. [Measurements](component-validation.md) |
| P04 NoPE attention-body kernels/fusion | Replace Python query loops and multiple operations while retaining candidates, scales and causal masks | Lower prefill/decode attention time and launch count | **Reject tested larger query chunks and FP32 fusion.** Fewer launches/intermediates did not improve speed; no serving integration. Reopen with evidence for a different strategy. [Components](component-validation.md#nope-attention-fusion-and-query-batching-p04) |
| P05 SM121 attention backend selection | First verify fixed-GLM NoPE/sparse-MLA shape/cache support in candidates such as FlashInfer/TRT or Triton, then A/B matched workloads | Reliable startup/warmup and effective kernel performance; detect unintended fallback | **Not adopted.** Numerical failure and required candidate-width incompatibility; reopen after relevant upstream changes. [Components](component-validation.md#direct-padded-native-attention-probe) |
| P06 CUDA Graphs | Resolve host decisions, dynamic operations and buffer lifetimes for capture/replay of supported shapes | Lower CPU step/launch overhead, especially steady decode | **Full model unqualified.** LPA's eager restriction needs its own validation; component capture is not server support. [Procedure](performance-investigation.md#kernel-launches-and-synchronization) |
| P07 Launch and synchronization measurement | Separate each rank's kernels, CUDA launch APIs, NCCL and host sync; estimate per-token deltas against a prefill control | Locate bottlenecks and make before/after comparisons reproducible | **Measured.** P03 reduced launches, not NCCL count. Duration sums are not wall-clock time. [Measurements](component-validation.md) |
| P08 Further sync/copy/conversion reduction | Use P07 to locate transfers, dtype changes and host round trips; consider load-time weight repacking only after observing repeated conversion | Less avoidable latency, UMA traffic and temporary storage | **Instrumentation available; further adoption pending.** Include dense GEMV/MoE cost; dropping checks alone is not acceptance. [Procedure](performance-investigation.md) |
| P09 Independent FP8 KV evaluation | Hold weight precision fixed; compare supported FP8/BF16 cache paths, scales, capacity and readback accuracy | More context/concurrency from smaller caches; measure speed direction and quality | **FP8 in use; dtype A/B pending.** Current launch path fixes FP8. BF16 needs compatible cache layout/runtime, not just a setting. [Configuration](startup-configuration.md) |
| P10 UMA and memory operation | Control KV pinning, memory utilization, host reserve, container limits, swap and cache state | Reduce OOM/swap-related latency; establish reproducibility and capacity | **Guards/settings implemented; systematic tuning pending.** Restrict `drop_caches` to justified cold-load comparisons; it is not a steady-decode optimization. [Configuration](startup-configuration.md) / [benchmarks](benchmarks.md) |
| P11 Prefill chunk × Kpool/indexer | Vary chunk budgets; compare split/native processing at and around pool, tail and cache boundaries | Improve long prefill and mixed-load ITL while preserving state/candidates | **Reject 128; retain 1024 as a throughput candidate.** Report 2K gains with worse maximum ITL; default 512. Separate 16K x two capacity passed at 1024. [Measurements](benchmarks.md#independent-prefill-chunk-evaluation-p11) |
| P12 Concurrent-stream measurement | Record client concurrency 1/2/4 separately from server active sequences; tabulate TTFT, queue, individual ITL and aggregate | Expose serial/parallel tradeoffs and stalls during long-input arrival | **Queueing and real batching measured separately.** P13 observed two active requests and multiple decode rows. [Measurements](benchmarks.md#independent-active-batching) |
| P13 Standard batching | With LPA off, validate seqs=2 then 4 and actual overlap; separate determinism from task acceptance | More aggregate throughput; quantify per-request latency/memory tradeoffs | **Accept two sequences for the limited throughput workload.** Task/tool and 16K x two capacity checks are separately scoped. Default one; four sequences and combinations unqualified. [Measurements](benchmarks.md#independent-active-batching) |
| P14 Similar-task batching | Compare code/translation/summary groups with mixed order, matching length, concurrency and offered load | Possible changes in expert overlap/speculative acceptance and aggregate throughput | **Not adopted for this workload.** One small matched order comparison gained only 0.7-0.9%; no expert-route attribution or complete-task quality evidence. Retain standard batching. [Measurements](benchmarks.md#task-grouping-order-comparison-p14) |
| P15 Context-length sweep | Vary input length and KV budget independently; consider 8K/32K/128K/262,144 only after capacity/support checks | Map prefill/decode/memory breakpoints and usable limits | **Systematic sweep pending.** Separate accepted configuration from completed real requests; expand from tested lengths. [Configuration](startup-configuration.md) / [benchmarks](benchmarks.md) |
| P16 CSA2: candidate Reuse/Reindex | Compare cross-layer selection reuse, restricted rescoring and shared pools, preserving per-layer KV and mandatory writes | Potential reduction in removable indexer work, judged by coverage and full latency | **Components and observation tested; serving deferred.** Small measured op share and slower shared-pool work at the 8K-equivalent size. Reopen only with useful full cost, coverage and quality at a larger context. [Components](indexer-reuse.md) / [measurements](component-validation.md#indexer-observation) |
| P17 TP=2 versus PP=2 | Implement/fixture-test intermediate tensors and mHC post/comb transport before matched A/B | Possible communication-wait savings versus stage serialization/imbalance | **Components and PP1 control complete; PP2 not accepted.** Fixed initial layout rejection; head management access was lost during retry, leaving root cause/final state unconfirmed. Full-model A/B not run. [Evidence](component-validation.md#pipeline-fixture-preparation-and-startup-p17) |
| P18 MTP + LPA + fusion, optionally Graphs | Compare individual/combined paths on identical assets/tasks with restored controls and observed LPA/speculation/capture activity | Determine interaction, quality/memory regressions and recovery | **Integration under evaluation; no settled result.** Do not infer combined gains from individual percentages. Graphs is a later supported condition. [Remaining integration](component-validation.md#reproduction-and-remaining-integration) |
| P19 Prefix caching (APC) | Compare cold/warm repeated system/tools/history and different prefixes; check hybrid state, boundaries and isolation | Lower TTFT and repeated prefill in conversations | **Unqualified; incompatible with current LPA.** Distinguish cache hits from faster cold processing. [LPA scope](lpa.md) |
| P20 Indexer workspace sizing | Measure peak requirements by shape/chunk/MTP depth; shrink only proven excess reservation | More host/KV headroom and fewer unnecessary allocations | **Conditional candidate; independent benefit unqualified.** Distinct from P16 selection changes; omit patches superseded upstream. [Investigation](performance-investigation.md) |

**Additional initiative (2026-09-13):**

| Initiative | Work | Expected effect / metric | Current status / procedure |
|---|---|---|---|
| P21 Expert Parallel | Independently switch expert-layer partitioning from TP to EP while retaining TP=2, DP=1, two active sequences and the fixed per-rank KV budget. Check pinned GB10/Marlin support before implementing default-off configuration, launch and validation paths | Expert compute efficiency and aggregate throughput, assessed with extra memory, communication, individual TTFT/ITL and quality | **Default-off settings and CPU contracts implemented; GPU untested.** Pinned Marlin declares support and accepts expert maps. Actual ownership, numerical behavior, speed and resources must pass before adoption. [Procedure](performance-investigation.md#expert-parallel-p21) |

P12 is a measurement axis, P13 a scheduler setting, and P14 a workload-order experiment: do not count one gain three times. MTP can already verify multiple speculative rows within one sequence; GEMM-like work does not necessarily require `max_num_seqs > 1`.

## Enterprise quality and adoption gates

Evaluate performance together with source-faithful answers, correct tools and manageable licensing. The objective is neither to steer the model toward a particular ideology nor to remove all appropriate safety refusals.

| Initiative | Work | Expected benefit / evidence | Baseline status / source |
|---|---|---|---|
| E01 License and provenance selection | Separate model, original code, patches, containers and harnesses; pin versions/notices under the MIT/Apache-centered selection policy | Make usage, modification and distribution boundaries reviewable by enterprises | Policy and inventory available; containers and harnesses do not acquire one blanket license. [Licensing](licensing.md) / [notices](../THIRD_PARTY_NOTICES.md) |
| E02 FreedomBench and political context | Separate original English, Japanese and long business-context tests across baseline/MTP/LPA/combined profiles; distinguish refusal, wrong answer, format and transport failures | Reveal responses to tested topics, unsupported political insertions, source infidelity and changes caused by optimization | Local adapter and required contracts available; full-original/all-profile and extension qualification pending. [FB-01–05](freedombench.md) |
| E03 Business functionality and operation | Test text, tool round trips, SSE, cancellation, isolation, ZCode/Claude Code, sustained load and two-rank recovery | Determine whether a fast benchmark profile is usable in practice; preserve rollback | Limited basic API results; harness/production reliability unqualified. [Validation](validation.md) / [harnesses](harnesses.md) / [operations](operations.md) |

FreedomBench measures agreement with its authors' answers on a limited set of questions. **A high score cannot prove the absence of all ideological bias.** Provide scope, conditions, results and untested areas so enterprises can judge suitability for their own use. Do not mix translated/extended scores with the original, or count short questions on which LPA did not activate as evidence for approximation quality. Reevaluate with the effective fusion/graph settings when adding those paths. The FreedomBench documents alone own question and scoring contracts.

## Repeatable process for the next version

1. **Establish a new native reference.** Check model/revision, licenses, runtime, SM support, quantization and cache layouts. Reassess old patches; mark upstream-resolved work unnecessary rather than automatically porting it.
2. **Move from components to the full model.** Separate exact arithmetic tests from approximation quality. Validate fixture state, then matched full-model A/B/A; separate tracing from final timing.
3. **Decide on each component, then combine at the end.** Evaluate each initiative independently through an adopt/defer/reject decision. Hold prerequisite components fixed in both arms and change one initiative per comparison. Once candidates are selected, use P18 for intended operating combinations and necessary off controls. Do not repeat the entire integration matrix after every component; retain existing combined runs as preliminary evidence. Do not simultaneously change batching, long-context capacity or PP.
4. **Separate functional acceptance from performance adoption.** Infrastructure such as Graphs may be accepted with little speed gain once correctness, resource limits and recovery are verified within its supported scope. Adoption as an acceleration requires a reproducible gain. Decide default enablement and final integration qualification separately. Distinguish exact text, task/tool correctness, refusals and incomplete execution; establish workload/thresholds before results. E02/E03 precede enterprise qualification.
5. **Retain negative decisions.** Record noise-level effects, slowdowns, capacity/quality failures or upstream obsolescence with reopening criteria. Preserve old results in Git and create new runs instead of overwriting them.

### Functional acceptance and defaults

Record **functional acceptance with its tested scope / performance adoption / default on or off / integration pending or qualified** separately. “Functionally accepted; speed difference within noise; default off; integration pending” is a valid outcome. Component or truncated-fixture success does not qualify the full model.

For P06, check memory retained for graphs, startup capture time, supported shapes and buffer lifetimes, and recovery from failures. Graphs is not a side-effect-free setting. Check LPA prefill and MTP speculative paths in P18; speed alone does not justify graphing every path. See PyTorch's [CUDA Graph constraints and memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs).

### Shared comparison record

Use these fields in the next version's experiment report; this is not a new configuration format or automated scoring system.

| Field | Record |
|---|---|
| Target and decision | Initiative ID, old/new version, intended workload/metric, predeclared criteria, adopt/defer/reject and reason; separate functional scope, performance adoption, defaults and integration qualification |
| Fixed assets | Model revision, source commit, both rank images, runtime/backend, arithmetic, template/tokenizer, projector/view and license inventory reference |
| Effective configuration | Profile fingerprint, TP/PP, active sequences, chunk, KV dtype/capacity, context, actual MTP/LPA/fusion/graph/APC state |
| Workload | Corpus/task/question hash and split, request IDs, actual input/output tokens, effort/sampling, order/concurrency/actual overlap and other load |
| Timing | Cold startup, warmup, TTFT, prefill control, per-request TPOT/ITL/queue/E2E, aggregate, repetition count and distribution; small-sample medians are not an SLA |
| Attribution and resources | Per-rank kernel/launch/NCCL/sync/copy, MTP acceptance/steps, minimum host reserve, container peak, swap and temperature; do not sum overlapping intervals |
| Quality and completeness | Planned/completed/missing/error counts, refusals, truncation, content/tools/source fidelity, numerical/state diagnostics and restored controls; identify unqualified areas |
| Reproduction and source | All A/B/A trials, profiler-off timing, raw run ID, reference document commit, public summary links and rollback settings |

When tokenizers or input lengths change, separate same-text business comparisons from same-token-budget processing comparisons. Keep raw logs private and maintain links from this catalog to the authoritative result documents.
