# CUDA and indexer component validation

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

Use `validation.component_worker=true` with LPA/MTP off and the verified image in the [startup configuration](startup-configuration.md). On the head, run `python -m glm53_setup.validation.run_components --config /path/to/profile.toml --corpus /path/to/documents.jsonl --output /new/record`. The driver stores all responses, actual token counts, A/B/A repeatability, timing samples and per-rank overlap. Retain the corresponding profiler traces and resource logs.

CUDA fusion with LPA/MTP, enterprise tasks, FreedomBench and harness qualification are separate integration gates. The default fused-unpack setting remains off until the operator selects a tested profile.

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

**Decision: capture/replay demonstrated; numerical acceptance and full-model progression deferred, default off.** Preserve the mismatch rather than treating exit zero as acceptance. Given the small measured speed differences, prioritize other independent initiatives. Reopen with evidence of useful full-model dispatch savings, then use identical forced token prefixes to localize attention/KDA or other state differences before acceptance. Full-model Graph, default enablement and LPA/MTP integration remain unqualified. A candidate image was rebuilt identically on both nodes from `eb30095` as `sha256:e18f7ae02e96beeb9af954a4e5600ce6ff534d9f91a9fcc2e440a4d91350c494`; it has not run the full-model Graph test or replaced the normal profile.
