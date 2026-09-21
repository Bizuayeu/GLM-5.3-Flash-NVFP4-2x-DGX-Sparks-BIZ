# CUDA and indexer component validation

[日本語](component-validation.ja.md)

The unpack/indexer measurements below use the pinned two-GB10 target, runtime source `74548eeac4ca5abe9f0036b3ac89f77555a31ec7` and image `sha256:32394330800422a71df89c89d399b8bd17d2dbe90806572ea4583f15ad46f09a` on both ranks. The later Graph fixture has its own scope and source identities below. These are experimental measurements, not production or language-quality qualification.

## Conditions

TP=2, Marlin W4A16, FP8 KV, eager, one sequence, no prefix caching, **LPA/MTP disabled**. The component worker changes only unpack mode between exclusive requests. Input uses the validation split of the retained LLM-jp corpus (SHA256 `fe1a7ffbf8fe36e9db925c25459863ce0cd432c6fd4ebe0585482dcd41da0ddb`); no held-out test documents were used. Each path has a warmup, followed by five one-output measurements or three 128-output measurements, with temperature=0, seed=42 and fixed output counts. Tracing is inactive during timing.

## Full-target unpack A/B/A

Median client request seconds, including prefill and output generation:

| Input tokens | Output tokens | Unpack off | Fused unpack | Restored off |
|---:|---:|---:|---:|---:|
| 64 | 1 | 0.357 | 0.326 | 0.359 |
| 2,048 | 1 | 6.162 | 5.076 | 6.133 |
| 8,192 | 1 | 24.513 | 20.392 | 24.578 |
| 64 | 128 | 9.361 | 9.349 | 9.370 |
| 2,048 | 128 | 15.231 | 14.068 | 15.171 |
| 8,192 | 128 | 33.535 | 29.285 | 33.418 |

The 8K one-output control improves by about 17%. The 128-output rows include prefill and are **not decode-only rates**. Short-context generation shows little change. These small samples do not establish production tail latency.

Paired traces of the same 64-token input with 1/33 outputs show **1,743→1,710 GPU kernels per additional token** on each rank; CUDA launch API counts match. NCCL remains 92 events per additional token. This reduction of 33 kernels/token is much smaller than the prefill benefit; it must not be described as eliminating the overall decode bottleneck. The separate unpack microbenchmark reduced four kernels to one at full candidate-row sizes.

## Numerical scope

GPU component checks matched Torch unpack exactly across all FP8 byte codes and selected FP32 scales, including signed zero, and matched attention output with padding and empty candidates. All full-target one-output A/B/A cases had equal token IDs.

The 128-output cases varied even between native/off repetitions. A separate 32-output logprob diagnostic found ties and changes in candidate probabilities on common prefixes in both off and fused runs. Shared-token logprob differences reached approximately 1.31 in the native repetitions; these are not evidence of uniformly tiny perturbations. Full-target deterministic equivalence and language-quality acceptance remain unresolved. Keep this baseline issue separate from the exact component arithmetic result; do not infer either fusion-specific damage or production qualification from it.

## Indexer observation

The observer attached to all 11 `SparseAttnIndexerKpool` operations and captured logical candidate IDs before shared-buffer overwrite. The following medians sum the CUDA intervals within each rank; **do not add the two ranks together**.

| Input tokens | Rank 0 op sum | Rank 1 op sum | One-output request | Rank 0 fraction |
|---:|---:|---:|---:|---:|
| 2,048 | 5.04 ms | 4.94 ms | 6.103 s | 0.083% |
| 8,192 | 34.84 ms | 34.03 ms | 24.562 s | 0.142% |

These intervals include the kpool op's scoring, selection and cache updates. They **exclude upstream indexer projections, normalization, query quantization and gate computation** in the enclosing `Indexer.forward`. They are not an upper bound on every possible indexer optimization. Reindex retains target query work, and mandatory cache writes cannot simply be counted as savings.

At the final query of this 8K validation input, adjacent sparse-layer Jaccard ranged from 0.349 to 0.693, with target-candidate coverage from 0.518 to 0.818. Both ranks agreed on the captured sets. The 2K selections covered all causal tokens and produced trivial Jaccard=1, which is excluded from evidence of useful sparse reuse. These are sampled queries from one fixed validation input, not a general overlap distribution or a tuned layer schedule.

The [indexer components](indexer-reuse.md) are retained for further evaluation. Reuse/Reindex is not enabled in serving: no layer schedule has passed quality gates, the measured kpool op is a small portion of the tested workload, and the shared-pool scorer was slower than native at the 8K-equivalent component size. Longer contexts require separate capacity, full-cost and candidate-quality measurements.

## Reproduction and remaining integration

Use `validation.component_worker=true` with LPA/MTP off and the verified image in the [server configuration](server-configuration.md). On the head, run `python -m glm53_setup.validation.run_components --config /path/to/profile.toml --corpus /path/to/documents.jsonl --output /new/record`. The driver stores all responses, actual token counts, A/B/A repeatability, timing samples and per-rank overlap. Retain the corresponding profiler traces and resource logs.

The [serial LPA/MTP/fused-unpack/async comparison](benchmarks.md#serial-integration-of-mtp-lpa-fused-unpack-and-async-checks-p18) now has limited paired task, timing, capacity and cancellation evidence. Broader business tasks, the full FreedomBench matrix and harness qualification remain separate gates. The template default for fused unpack was off when these measurements were taken; the current value is in [server configuration](server-configuration.md#distributed-defaults).

## Independent decode Graph fixture

On 2026-09-12 (Asia/Tokyo), a four-layer, single-GB10 fixture compared synchronous eager execution against asynchronous index checks plus uncompiled-prefill/full-decode Graphs. **Fusion, LPA, MTP and APC were off** in both arms. The checkpoint retained its original widths and experts; this is not a language-quality test. Settings were context 16,384, chunk 512, one active sequence, FP8 KV 512 MiB, Marlin W4A16, seed 42 and temperature zero. Each of 64/2,048/8,192 input tokens generated 32 tokens after warmup, with three timed repetitions and tracing afterward.

This used the image named above with the new attention source mounted at both its canonical module and deployed `glm53_reference.py` path. Attention SHA256 was `d044697bae525fac77cbec451475f29df5298a5d63700560b7c532a6729335de`; the exact driver bytes were `0590daed24d583097fe125ceb5f5200081d3d8f60d728e28b2b4393304fffd81`. The driver is retained as `glm53_setup.validation.run_graph_fixture` at source commit `eb30095`. Private run IDs are `graph-component-v13-results/fixture-eager-single` and `fixture-graph-single`.

| Input tokens | Eager median seconds | Graph median seconds | Generated tokens across arms |
|---:|---:|---:|---|
| 64 | 0.518719 | 0.504478 | Equal in all three repetitions |
| 2,048 | 0.982739 | 0.970794 | Differed from output token 12 in all three repetitions |
| 8,192 | 2.480334 | 2.487705 | Equal in all three repetitions |

Each arm was internally token-repeatable. At the first 2K divergence, eager assigned equal top logprobs to token IDs 2090 and 43424 and selected 2090; Graph selected 43424 with a 0.0625 logprob lead. This is consistent with a numerical perturbation at a tie, but does not isolate the cause or establish acceptable quality. The comparison changed both bound-check scheduling and Graph execution; asynchronous eager is the missing attribution control. Speed differences are small and are not grounds for performance adoption.

The Graph trace observed 31 host Graph launches versus zero in eager. These are not single GPU kernels. Capture reported approximately 0.06 GiB of additional memory in this truncated fixture, not the full model. Both runs completed without OOM; invalid-index tests in separate component containers had previously confirmed that asynchronous assertions invalidate the CUDA context on failure.

An additional asynchronous-eager control (`fixture-eager-async-single`, driver SHA256 `f718c322431f1388c9b28ff3f0ee20caa5ed3be7426657349494fd32f5a53115`) was token-repeatable and matched synchronous eager at all three lengths. It differed from Graph only at 2K. This narrows the observed divergence to the Graph execution condition, without identifying the internal operation or proving a quality regression. The control's median seconds were 0.508454 / 0.989053 / 2.485845 at 64 / 2K / 8K.

**Decision: capture/replay demonstrated; numerical acceptance and full-model progression deferred, default off.** The full-model progression was then measured on 2026-09-21 and did not adopt Graphs: on the reference pair's serving profile at depth 4, every one of ten inputs decoded 7 to 9 ms per step slower than eager with identical completions ([P06](optimization-overview.md#decode)). Preserve the mismatch rather than treating exit zero as acceptance. Given the small measured speed differences, prioritize other independent initiatives. Reopen with evidence of useful full-model dispatch savings, then use identical forced token prefixes to localize attention/KDA or other state differences before acceptance. Full-model Graph, default enablement and LPA/MTP integration remain unqualified. A candidate image was rebuilt identically on both nodes from `eb30095` as `sha256:e18f7ae02e96beeb9af954a4e5600ce6ff534d9f91a9fcc2e440a4d91350c494`; it has not run the full-model Graph test or replaced the normal profile.

**Re-read on 2026-09-18 with the expert token order fixed** (image `sha256:e7a2a606…`, source `0957c4e`, the [canonical MoE order](server-configuration.md#commands) on). The same driver on the byte-verified four-layer MTP fixture (MTP k=3, fused unpack, asynchronous index checks, FP8 KV 512 MiB, seed 42, temperature zero), run twice eager and twice with `FULL_DECODE_ONLY` decode Graphs (one captured decode shape of 4 tokens, 122 host Graph launches per run), produced identical 32-token outputs and identical top-10 logprobs at 64 / 2,048 / 8,192 input tokens in every one of the runs and across every pair of runs, including the earlier 2K tie. Four further runs with prefix caching enabled (two adding a 12,288-token input) were identical as well; none of them observed a cache hit, because the MTP block on this fixture is 8,960 tokens and a hit needs priming beyond the 16,384-token context, so the hit path itself is not covered here. The 2026-09-12 divergence is therefore read as the expert-order variability, not as a Graph difference. Consequences: the launcher now accepts Graphs with MTP and prefix caching for one sequence and derives the capture size from the speculative depth ([server configuration](server-configuration.md#feature-combinations-and-limits)); batching, LPA and TP=2 collectives remain unqualified for Graphs, and full-model speed and acceptance are a separate A/B.

## Direct padded native attention probe

Private run `native-attention-v15` tested the public FlashInfer sparse MLA API on the same `e18f7ae…` image after full-model servers had stopped. The probe used real FP8 cache packing, 64 zero RoPE dimensions, GLM `arbitrary_fp32` scales and preserved every candidate. Driver SHA256: `4c41525d85ca5ba148cbd98d8c53241cf5e00817e47cd373c7089dc14f93fef6`; module: `glm53_setup.validation.benchmark_native_attention`.

The direct path **did not pass**. At 32 heads, tested widths 63/64/65/2048 had nonzero empty-row outputs and maximum absolute differences of 3.03125, exceeding the predefined two-BF16-ulp bounds (approximately 0.025–0.031). When 2051 candidates were padded to 2176, dispatch rejected the shape because the fixed SM120 v32/GLM decode table supports widths 128/512/1024/2048, not 2176 or a larger padded width. The tail-sensitivity and timing stages were consequently not run. This was a component validation failure, with container exit 1 and no OOM; no serving backend was changed.

**Decision:** do not adopt this direct backend substitution. Zero RoPE padding alone is insufficient; truncating candidates to 2048 is not acceptable. A separate kernel extension or candidate-preserving composition would require new numerical/state and performance tests. No native-attention speedup is claimed from this run.

A subsequent prefill-only probe (`native-prefill-v16`, driver SHA256 `171b013c59cd5ffb3acc45143236b5ff541c740e33a2ab4cafc3f35f80ad903b`) used 65 query rows and caller-zeroed output, with intended suffix-only valid candidates and empty rows. The library rejected the first configuration: `GLM_NSA`, 32 heads, top-k 128, page size 64. It exited 1 without OOM before any parity case completed. The 2176-candidate prefill case and timing stages were not reached; this does not prove every prefill shape unsupported. Prefill-only substitution remains unqualified, with no serving changes.

**Revisit by row kind (2026-09-17).** `benchmark_native_attention --by-row-kind` at source `adf8ca9` (driver SHA256 `4baea4aa9161658cac09db71ee849de7175b2b0b7b3be0ad744c68ca96320b8a`) reran the probe on an idle GB10 with the serving image `f6fc154c…`, FlashInfer 0.6.18 and width 2048, the widest SM120 v32/GLM decode shape. Errors are split by row kind, and empty rows were either passed to the kernel as they are or pointed at slot 0 with their output zeroed afterwards. That handling is described in Mia's `Dockerfile` (no code adopted) and drowzeys WALLS #4/#5/#7 (no code adopted).

| Heads | Query rows | Rows with every candidate | Rows with 17 candidates | Empty rows passed as they are | Empty rows at slot 0, zeroed | Bound |
|---:|---|---:|---:|---:|---:|---:|
| 32 | 3 (decode kernel) | 0.0078 | 0.0625 | 3.03125 | 0 | 0.031 |
| 32 | 65 (prefill orchestrator) | 0.0083 | 0.0389 | 3.03125 | 0 | 0.027 |
| 64 | 3 | 0.0078 | 0.0547 | 3.03125 | 0 | 0.029 |
| 64 | 65 | 0.0088 | 0.0566 | 3.03125 | 0 | 0.030 |

The v15 difference came from the empty rows and reproduced exactly; zeroing them removes it. Rows holding every candidate stay within the bound, but rows holding 17 candidates (packed first for three rows and last for 65) exceed it by 1.3–2× under either empty-row handling. The 65-row shape at width 2048 was accepted, unlike the top-k 128 shape of v16. With all 2,048 candidates valid the kernel stayed within the bound and took 0.085/0.092/0.413/2.439 ms for 1/8/65/512 rows, against 0.187/1.105/8.831/68.837 ms for the reference under the P04 timing conditions.

Dropping one pool to fit the kernel (2,051 to 2,047 candidates) was measured on synthetic data only: a random packed FP8 cache and queries, 64 rows × 32 heads, pools ranked by their reference attention mass. The largest output change was 0.022 when the lowest-mass pool was dropped, 0.039 for a random pool and 0.151 for the highest-mass pool, against a largest reference output of 0.326 and a bound of 0.016. Synthetic attention is nearly uniform (a pool carries about 1/512 of the mass), so these numbers do not predict the effect under a real indexer; real ranks and attention were not captured.

**Decision after the revisit:** zeroing empty rows removes the v15 failure, but rows with few candidates still exceed the bound and fitting the kernel drops candidates, so the direct substitution stays unadopted. Using 2,047 candidates would need whole-model quality evidence (FreedomBench, tool evaluation, long needle retrieval) and a separate decision.

## SM90 FA2 MLA wrapper probe

On 2026-09-17 (Asia/Tokyo), `benchmark_sm90_attention` at source `adf8ca9` (driver SHA256 `c94af60c1ac31ac1b3184edb749f7c34496a4895a0e6d05f4c6e7ecec4a0c260`) called FlashInfer's `BatchMLAPagedAttentionWrapper` the way the pinned vLLM's `FLASHINFER_MLA_SPARSE_SM90` backend does: NoPE (`head_dim_kpe=0`), page size one with each query row's candidates as its KV pages, exact per-row lengths and `causal=False`. The pinned vLLM selects that backend only on compute capability 9, where it uses `fa3`; the probe used `fa2` on GB10 (capability 12.1). It ran on the same host and image as the revisit above, without model weights and without changing serving. Informed by sfxnz's `docker/Dockerfile.sm121-v8` (no code adopted) and tonyd2wild PR #17 and Issue #20 (no code adopted), which extend that backend to capability 12 with `fa2`.

The first `plan()` built the NoPE module by JIT in 8.0 s; the image's JIT cache holds only the `head_dim_kpe=64` variants. **FP8 KV is unavailable on GB10 with FlashInfer 0.6.18:** `plan()` raises `FP8 kv_data_type for MLA requires an SM90 (Hopper) device, got SM121.` The probe therefore used BF16 KV decoded from the same packed cache as the reference. A BF16 MLA cache takes 1,024 bytes per token per MLA layer, against 656 for `fp8_ds_mla`.

The numerical checks passed the same two-BF16-epsilon bound as P04:

- Widths 17/63/64/65/2048/2051/2176 with contiguous and sliced queries: maximum differences 0.00024–0.0039 (bound 0.047). No width limit applied.
- Empty rows, including width 0: exactly zero.
- A sole tail candidate among 2,051: kernel difference 0.0, and omitting the tail changes the reference output by 15.6.
- 64, 128 and 256 rows of 2,176 candidates: finite, maximum difference 0.00049. The NaN reported for FlashInfer 0.6.17 did not appear.

Timing used the P04 conditions: 32 heads, 2,176 valid candidates, the median of five synchronized batches of ten calls. The backend plans once per step and runs once per layer.

| Query rows | Reference (ms) | FA2 run (ms) | FA2 plan (ms) | Reference / FA2 kernels per call |
|---:|---:|---:|---:|---:|
| 1 | 0.205 | 0.032 | 0.028 | 27 / 1 |
| 8 | 1.193 | 0.073 | 0.036 | 25 / 1 |
| 65 | 9.082 | 0.569 | 0.041 | 195 / 1 |
| 512 | 70.604 | 3.508 | 0.085 | 1,348 / 1 |

**Decision:** the kernel is numerically usable as a component on GB10 and about 20× faster than the reference at 512 rows. This is not evidence for switching the serving backend. A switch would store BF16 KV, which holds fewer tokens per GiB, and the packed-cache features were built for the SM120 path: P03 fused unpack decodes `fp8_ds_mla`, P19 prefix caching and P22 LPA rely on the 4,608-token layout and the reference hook, and [canonical candidate order](candidate-order.md) patches only the SM120 GLM path. Each would need requalification, with full-model quality and capacity checks, in a separate decision.

## NoPE attention fusion and query batching (P04)

On 2026-09-13 (Asia/Tokyo), two independent GB10 component experiments used image `e18f7ae…`, 32 heads, latent width 512, 2,176 candidate entries per query and packed FP8 cache with arbitrary FP32 scales. Neither changes the serving backend. Synchronous index-range checks remained enabled; Graphs and unpack fusion were off in the reference. Each timing is the median of five batches of ten calls after three warmups, synchronized around each batch. Profiling was separate.

**Larger reference query chunks — not adopted.** `benchmark_query_chunks` at source `10d0fc6`, private run `query-components-v17`, compared internal chunks 8/32/64. This differs from the scheduler chunk in P11. Padding, empty rows and single-tail candidates passed the predefined numerical bound; some nine-query results were not bitwise equal. At 512 queries, all results were bitwise equal, but larger chunks were slower and used more temporary memory:

| Internal query chunk | Median call (ms) | Extra peak live allocation (MiB) | GPU kernels per call |
|---:|---:|---:|---:|
| 8 | 70.473 | 133.42 | 1,348 |
| 32 | 80.245 | 488.75 | 340 |
| 64 | 81.382 | 956.39 | 172 |

Keep the internal default at eight. Fewer launches alone did not improve this workload.

**FP32 fused attention — not adopted.** `fused_nope_attention` and `benchmark_query_chunks --fused-attention` at source `99c7427`, private run `fused-nope-components-v18`, combine cache decoding, FP32 score reduction, online softmax and value accumulation without materializing gathered KV. The kernel preserves all candidates and uses 64-bit addressing. Separate GPU tests compared against FP64 for widths 0/17/65/2048/2051/2176, including noncontiguous queries, exact empty-row zero and a sole valid tail candidate. The declared bound was `2 * BF16 epsilon * max(1, max(abs(reference)))`, not bitwise equivalence or a guarantee of two ulps at every output magnitude.

The timing workload below has fully valid candidate entries; empty-row checks were separate. Query-chunk timings above use masked rows and should not be treated as the identical workload.

| Query rows | Reference (ms) | Fused FP32 (ms) | Reference / fused extra live allocation (MiB) | Reference / fused kernels per call |
|---:|---:|---:|---:|---:|
| 1 | 0.192 | 0.446 | 9.93 / 0.031 | 27 / 5 |
| 8 | 1.060 | 1.945 | 79.42 / 0.250 | 25 / 5 |
| 65 | 8.787 | 13.566 | 120.27 / 2.031 | 195 / 5 |
| 512 | 68.154 | 105.006 | 134.24 / 16.000 | 1,348 / 5 |

All outputs were finite and within the declared bound; maximum absolute differences ranged from 0.000122 to 0.000488. The five GPU kernels include range checking around the fused attention kernel. The measured synchronization API count remained four for both paths, including benchmark synchronization. These are component-call counts, not full-model kernels per generated token.

A bounded tiling follow-up (`benchmark_fused_tiles`, private run `fused-tiles-v19b`) tested tile/warps 8/4, 16/4, 16/8 and 32/8. All passed the same bound and compiled without register spills. Best one-query time was 0.389 ms; best 512-query time was 105.039 ms. Neither beat the reference measurements. The initial `fused-tiles-v19` attempt failed at container mount setup before GPU execution; its record is retained separately.

The prototypes reduce launches and temporary allocation, but **do not provide a measured speed benefit**. Keep them for reproduction; do not connect them to startup, claim full-model acceptance, or treat the already useful P03 unpack fusion as rejected. Reopen P04 with a materially different execution strategy supported by profiling, rather than further blind tile sweeps.

### Shared-head Tensor Core candidate — not adopted

`runtime.fused_nope_dot.dot_nope_tf32x3` is a separate follow-up: each program shares decoded K/V across 16 query heads and uses explicit `tf32x3` matrix products, with FP32 softmax/accumulation and BF16 output. It is not IEEE-equivalent arithmetic; [Triton's dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) defines the precision selection. The benchmark records PTX hashes and checks that TF32 MMA instructions were emitted. It preserves all supplied candidates and shares the existing shape/device/range checks.

The GPU FP64, empty-row and sole-tail-candidate tests passed in `dot-component-v33-results`. The same image `7cb5f93…` ran the isolated component without model weights; all timed outputs were finite and within the predeclared bounds. PTX contained TF32 MMA instructions. However, compilation reported 1,522/1,530 register spills for 16/32 heads, and the kernel was substantially slower. At 512 query rows, reference versus TF32x3 took approximately 63.75→962.42 ms with 16 heads and 68.95→1,945.97 ms with 32 heads. This implementation is **not adopted**; it has no full-model integration or speedup claim. The register-spill evidence is retained for any future redesign.

`benchmark_query_chunks --tf32x3-attention --heads 16` or `--heads 32` reproduces the component in a fresh output directory. Source archive SHA256: `290dd45b32997ecb101e237db74485fbc0437339e1a0419761209b46cf49077d`. Serving remains unchanged and the earlier SIMT/query-chunk rejections stand.

## Pipeline fixture preparation and startup (P17)

The experimental source-pinned model patch propagates BF16 hidden/residual tensors and FP32 deferred mHC post/comb tensors across PP stages. The four-layer, single-GPU control (`pipeline-v17-control-a`) completed three repetitions at each of 64/2,048/8,192 input tokens with 16 outputs, finite chosen-token logprobs and the expected buffer shapes. Image: `sha256:b9ae526c6369b7bac909c2043853f60172e66eecc1b34e891b1faa607284eb4a`. This is a truncated-model control, not PP qualification or a language-quality result.

The first two-stage attempt (`pipeline-v17-pp2-a`) loaded stage weights but failed before readiness: the KDA-only stage reported six supported KV layouts while the KDA/MLA stage reported only `LBHNC`. The pinned resolver required identical lists even though their intersection was nonempty. Both containers were stopped without OOM.

`patch_pipeline_layout` now selects the common supported list for PP=2 while preserving the existing mixed-cache shape checks and requested-layout validation. An empty intersection remains an error; non-PP selection is unchanged. CPU checks cover heterogeneous lists, preference order and empty/disjoint support. The source utility SHA256 is `63f58dd2b29543045109f40897ddc5f57e299bb4259f0e37a64b024cb1f20ddb`; patched SHA256 is `48222663f4fb1842dd4d93e34f6e356ebaa3d4ebe1c15255d0b5bd670359e4d3`. Both nodes built the same candidate image, `sha256:5052db070ec651c0b16f1c72f523d631d4b96ba7a2e2b5c3adce46b3f2cace79`.

The retry (`pipeline-v17-pp2-b`) is **not accepted**. After temporary SSH unavailability cleared without a reboot, its log confirmed a mismatch between stage-local Mamba cache specs; both containers had exited without OOM. The fixed GLM cache grouping also explicitly requires an MLA layer on every stage that has Mamba layers. A four-layer model split 2+2 is therefore an unsuitable PP fixture. Do not relax those cache contracts to make it run, or infer the cause of the SSH interruption from this model failure.

The fixture builder accepts `--layers 8`, providing one KDA/KDA/KDA/MLA block per stage. Both nodes built and byte-verified all 17,526 selected tensors (25,606,617,384 bytes) against the pinned original weights. The eight-layer comparison uses the original `b9ae526…` PP image on both sides; the layout-intersection patch is unnecessary when both stages declare the same support. Its observer records finite attention outputs and active KDA conv/recurrent state as well as transfer hashes. The [server profile](server-configuration.md) can select PP with a matching image; full-model acceptance remains separate.

### Eight-layer PP observations

Private runs `pipeline-v23-control-a` and `pipeline-v23-pp2-a` completed 64/2,048/8,192 input tokens, 16 output tokens and three repetitions per length. Both used eager TP=1, chunk 512, context 16,384, FP8 KV 512 MiB per rank, no MTP/LPA/fusion/APC and identical image/fixture bytes. The control retained all eight layers on one GPU; PP2 placed layers 0–3 and 4–7 on separate GPUs. Observation-source SHA256: `44fcceb70214a951df0e33739fd7a4b11288f517469d083a848e05db9bf671d4`; driver SHA256: `a2d0410e98af0bdd337653911e733f156d8b875926de4e4222accacb7e9f36c2`.

All **198 stage transfers** matched every byte hash, shape and dtype of hidden/residual/post/comb on sender and receiver. All nine generated token sequences matched the PP1 control, and all observed tensors were finite. At 64 tokens, layer outputs, active KDA state and chosen-token logprobs also matched exactly.

Longer inputs were not bitwise invariant. Across 1,584 layer observations containing 2,376 active KDA tensor observations, 677 layer rows differed between PP1 and PP2. The first difference was at layer 6 of the first 2K prefill chunk, beyond the transfer boundary. Both PP1 and PP2 also showed internal repeat variation despite repeated token sequences:

| Input tokens | Maximum cross-arm chosen-logprob difference | Maximum PP1 repeat difference | Maximum PP2 repeat difference |
|---:|---:|---:|---:|
| 64 | 0 | 0 | 0 |
| 2,048 | 0.06609 | 0.06527 | 0.06480 |
| 8,192 | 0.18685 | 0.18307 | 0.18979 |

This does not establish strict numerical equivalence, negligible error or unrestricted language quality. It also does not identify a transfer defect: the actual transmitted tensors match, while repeat variability exists without PP. Retain the diagnostic and compare matched-prefix numerical differences against the baseline before stronger acceptance claims.

Both runs stopped without OOM. Minimum host available RAM was 71.82 GiB for the 48 GiB-capped PP1 control and 87.69/39.71 GiB for the 32 GiB-capped PP2 ranks; the host reserve was 4 GiB. These minima include loading/warmup, not only steady inference. Head stopped with exit 0; peer still required forced termination (exit 137). Observer hashes introduce synchronization and CPU copies, so this is **not a TP2-versus-PP2 speed benchmark**. Full-model performance, longer-run recovery and optimization combinations remain unqualified.

## Independent expert placement fixture (P21)

The same byte-verified eight-layer model ran TP=2/PP=1 with EP off, on and restored off (`expert-v24-off-a`, `expert-v24-on-a`, `expert-v24-off-b`). Each arm used the same `b9ae526…` image, eager, one sequence, chunk 512, context 16,384, FP8 KV 512 MiB per rank and no MTP/LPA/fusion/APC. Every arm completed three repetitions at 64/2,048/8,192 input tokens with 16 outputs and finite observed tensors. These truncated-model runs do not measure language-task quality.

Actual `RoutedExperts` objects in all five MoE layers confirmed the intended transition. EP off retained 288 tensor-sharded experts per rank; EP on placed 144 complete experts per rank, with disjoint ownership covering all 288. `MarlinExperts` was selected throughout. MoE TP size changed 2→1 and EP size 1→2, while model TP remained 2 and DP/SP remained 1. The fixed runtime's `use_all2all_kernels` was false in this topology; no DeepEP invocation is claimed. Restored off recovered the original ownership. The observer also recorded actual loaded parameter shapes, dtypes and byte counts.

Ready-time Torch allocation was approximately 12.682 GiB per rank and reservation 14.896 GiB in all arms. This is evidence for the small fixture, not a full-model memory guarantee. All containers stopped without OOM; head exited 0 and peer 137.

Generated tokens matched across arms at 64 and 2K input. At 8K, every arm varied between its own repetitions, and both off→on and off→restored pairs diverged after 2/4/2 common output tokens. Maximum shared-prefix logprob differences were:

| Input tokens | Off versus EP on | Off versus restored off |
|---:|---:|---:|
| 64 | 0.11011 | 0.08337 |
| 2,048 | 0.22505 | 0.08900 |
| 8,192 | 0.06813 | 0.06038 |

EP's 2K probability difference is larger than the restored baseline difference, so this is not a claim of numerical equivalence. Keep the 8K baseline variability and the EP delta visible. **This fixture demonstrates placement, loading and bounded execution.** The subsequent [full-model comparison](benchmarks.md#independent-expert-parallel-evaluation-p21) passed its limited task/capacity checks but did not justify performance adoption. The diagnostic hashes synchronize/copy tensors and were disabled for full-model speed measurements.

EP observer SHA256: `38c2e228ab086d179b06a7663a0879b4871950fdefc65632e4768d9a51541642`; validation driver SHA256: `a12c8aa6705e3df670853ff391bddfea2ff039c72e64b36387117a7e8405a0dc`. Full raw responses and state observations remain private.

## Independent prefix-cache fixture (P19)

`apc-fixture-v41-off/on/restored` used the byte-verified four-layer fixture with the V36 image recorded in [serial integration](benchmarks.md#serial-integration-of-mtp-lpa-fused-unpack-and-async-checks-p18). All arms used TP1/eager, one sequence, context 16,384, chunk 512 and FP8 KV 512 MiB. MTP, LPA, fused unpack and async index checks were off. APC was the changed setting. All arms completed 13 input lengths, three cold/warm pairs per length, 16 output tokens per request and an interleaved-input check, then exited 0 without OOM. Host RAM minima were 89.20/98.04/100.48 GiB under a 32 GiB container cap and 4 GiB reserve.

Although the requested block size was 256, the actual common cache block was 8,704 tokens. Warm requests through 8,704 input tokens reported zero cached tokens; 8,705 and 16,319 inputs each reused 8,704 tokens. The shorter no-hit results must not be counted as successful cache reuse or a quality check of that reuse.

| Input tokens | APC-on cold median | APC-on warm median | Warm cached tokens |
|---:|---:|---:|---:|
| 8,705 | 2.5634 s | 0.2518 s | 8,704 |
| 16,319 | 4.5622 s | 2.2750 s | 8,704 |

These fixture request times include 16 output tokens and may include shape-specific JIT in a first sample; they are not full-model performance claims. Through 8,705 inputs, generated tokens matched across all arms. At 16,319, every arm varied at the first token between the same two candidates: an exact tie or a 0.03125 logprob lead. Off/on pairs matched 3/6 times, while off/restored matched 5/6. Maximum within-arm shared-prefix logprob deltas were about 0.0632. Other lengths also showed probability variation without cache hits; no bitwise equivalence or cause attribution is claimed.

**This fixture demonstrates actual cache reuse and bounded execution.** Its interleaved prompts were only 2K and had no hits, so they do not establish isolation of positively cached requests. The subsequent [full-model A/B/A](benchmarks.md#independent-full-model-prefix-caching-p19) supplies limited real-hit long-document, tool, resource/cancellation and restored-control evidence. Driver SHA256: `da967a4928ab88846c33523fb11c0b9c65234d27f88274ac752c78b3bd8bc198`. The template default for APC was off at the time and is on now ([server configuration](server-configuration.md#distributed-defaults)); LPA/APC uses the separate P22 contract below.

## APC-first LPA cache isolation (P22)

The [P22 fixture](apc-lpa-design.md) uses a deliberately non-teacher projector on the byte-verified four-layer checkpoint. It checks the real cache manager's joint hit, actual query omission, GPU hashes of returned exact-prefix blocks, and a following ordinary request. The synthetic projector is a contamination probe, not a trained language-quality candidate.

`p22-fixture-v50` passed with TP=1/eager, one sequence, context 32,768, chunk 512, FP8 KV 1 GiB, no MTP/fusion/async checks, a 32 GiB container cap and 4 GiB host reserve. Effective worker source SHA256 was `695039966835f037286c89f7de2657cc1423795991ce99ec85120a61cb85009d`; pinned GPU-worker patch SHA256 was `53647920093e7c141f0aad7dcbb033925f867970dcaa9e525a21391474281eb0`. The fixture driver SHA256 was `9bafd1e58c0f94f7b9ce9cbb8f76ec0fc09b7a083cb410110afe4ec937c917ab`.

At N=18,432 and H=8,704, the approximate suffix omitted 9,216 queries in layer 3. All eight hashed prefix tensor views were unchanged. The following exact request still had H=8,704, then grew reusable exact cache to 17,408. A cold approximate request had H=0 and left no reusable prefix. All three exact-control 16-token outputs matched; the synthetic approximation differed in its first distribution. The container exited 0 without OOM, with minimum host availability 91.398 GiB.

`p22-fixture-v57` added MTP k=3, fused unpack and async checks on fixed image `sha256:59eeba6c94db9f1536010f6c258992c203458703dcf1e23b0bde7cb53d445716`, source `f7b47fc`. The measured block was 8,960; MTP replay required exact priming through 17,921 tokens to restore H=8,960. At N=27,904, all eight requests completed, shared-prefix hashes stayed unchanged, the approximate suffix did not extend shared reuse, and subsequent ordinary computation grew it to 17,920. The container cap was 48 GiB with a 4 GiB host reserve.

**Its strict token-identity criterion did not pass.** The first exact control and the later two exact controls diverged at output position 9: one top candidate led by 0.0625 in the first control, while four candidates tied in the other path; maximum shared-token logprob difference at that first divergence was 0.061949. The exact request following approximation matched the restored exact control. The failed result and exit 1 are retained, rather than relabeled as a numerical pass.

A separate `native-apc-mtp-v58` diagnostic ran six exact-prime/exact-follow-up pairs with **no LPA owner ever constructed**. It reproduced both complete 16-token paths seen above. This establishes that the observed branching also occurs without LPA. It does not prove bitwise stability or general language quality. Cache-isolation evidence, strict numerical identity and subsequent full-model task/performance acceptance remain distinct results.

`p22-fixture-v62` then exercised the asynchronous scheduler on source `3df93c9`, image `sha256:0de1ef13b7bfebb088ac7d9399e2d45decf058f16819d72f5a8e89c1bc993b81`. The CPU cursor can be ahead of the GPU after rejected MTP drafts; the LPA guard now recognizes this only in generated-token decode and still rejects any mismatch inside the prompt. Both approximate requests observed 13 such corrections. All eight requests, cache-boundary checks and unchanged-prefix hashes passed; the three exact 16-token controls also matched in this run, which exited 0 without OOM. The earlier native-only variability remains a separate retained result. Full-model asynchronous MTP combinations require their own validation.

## Additional history fixtures

`fixture-v66-target` and `fixture-v66-mtp3` used source `441384c`, image `sha256:569538ce8b1c259f3ee13242f387320417b2d63a0b4122b6dc74d92ae74dae33`. Each completed 62 requests, including 18 edit/branch conditions at pool boundaries, scheduler-block boundaries and 10/50/90% positions. Original shared-prefix hashes remained unchanged in every condition, edited requests never restored beyond their changed prefix, and exact revisits retained the original shared states. The runtime difference from `3df93c9` was a read-only cache-layout RPC; forward and cache-publication algorithms were unchanged.

The target fixture passed strict control identity and exited 0. The MTP3/fusion/async fixture **failed strict control identity and exited 1**: its first exact control followed one previously observed path, while trial/restored followed the other. Independent comparison found both complete 16-token paths in the earlier six-pair native-only `v58` diagnostic. Its 18 state-isolation conditions passed; that does not turn its numerical failure into a pass. Neither container was OOM-killed. Full-model task and operational acceptance remain separate from bitwise repeatability.

## Cold-cache FA2 startup (2026-09-20)

An eight-layer, byte-verified stock fixture (23.91 GiB loaded weights) on one GB10 compared an unset JIT parallelism environment with `MAX_JOBS=2` and `FLASHINFER_NVCC_THREADS=1`. Both used the 1.6.0 source, reference image `3a396af5…`, Marlin W4A16, FA2 prefill, one sequence, eager mode, 16,384 context, 512-token chunks and 512 MiB FP8 KV. Each arm had a new empty runtime cache, including the NVIDIA driver cache. Existing caches, OS page cache and swap were left intact. Other GPU services were stopped for the comparison and restored afterwards.

| Observation | Unset | 2 / 1 |
|---|---:|---:|
| Load start to engine ready | 270.36 s | 266.50 s |
| First 2,048-token request after ready | 9.26 s | 8.91 s |
| Lowest host MemAvailable during startup and request | 71.38 GiB | 71.11 GiB |
| Lowest host MemAvailable during that request | 80.12 GiB | 80.94 GiB |
| Maximum observed nvcc / cicc processes | 3 / 1 | 2 / 1 |

Both exited successfully without OOM, invoked FA2 eight times in the measured request and returned the same single token. Fresh FA2 NoPE shared libraries and Triton, TileLang, Inductor and NVIDIA cache files were confirmed. These checks establish execution, not language quality. Host/process sampling was every two seconds and can miss short peaks.

**Decision: keep the existing environment.** The limit can remove at most one compiler process. The FA2 NoPE module is the only FlashInfer module this recipe compiles at run time, and it has three translation units, so an unrestricted Ninja build runs three nvcc at once rather than one per CPU core. The running TP=2 MTP reference pair holds the same single module with three objects on both ranks, and its build log shows the three compiles starting together and ending within 8.1 s. The fixture's overall minimum fell during weight loading, about 70 s before the multi-nvcc window and about 9 GiB below it, so it does not measure JIT parallelism; on the full model the first FA2 launch's lowest reading did fall in the first warmup request, which is this window. Inside the window the unset arm dipped to 80.12 GiB and recovered to 80.61, while the limited arm declined monotonically to 80.94. That is consistent with one compiler process, but four samples per window and one run per arm give no spread. The pinned FlashInfer already defaults NVCC threads to one, so only the Ninja job limit changes. The JIT cache persists across launches, so the exposure is one launch per new cache. Initial OS cached memory differed (42.86 versus 67.29 GiB); overall startup time is not an isolated JIT-speed measurement. No full-model benefit is claimed and no TP=2 restart was warranted by this fixture result.
