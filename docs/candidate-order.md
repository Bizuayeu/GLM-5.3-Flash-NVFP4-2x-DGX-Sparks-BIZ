# Canonical sparse candidate order

[日本語](candidate-order.ja.md)

The reference image normalizes GLM sparse-MLA candidates into logical token order before translating them into physical cache indices. This is a shared vLLM/GLM runtime fix, independent of speculative decoding. Ordinary generation, MTP and external proposers consume the same backend boundary.

The observed top-k kernels can return the same selected set in different orders for identical scores and valid ranges. Finite-precision Attention can then produce different outputs. The normalization preserves selection, multiplicity and padding; it does not rescore, truncate or deduplicate candidates. It creates a separate ordered tensor and leaves the shared indexer buffer and KV data unchanged. Physical cache indices are not used as the ordering key.

## Installation and upgrades

Every reference image carries the patch and sets `GLM53_CANONICAL_CANDIDATES=1`; build, verify and inspect images as the [current image contract](server-configuration.md#current-image-contract) describes. For a controlled comparison or rollback, a container environment override `GLM53_CANONICAL_CANDIDATES=0` disables the normalization; the GLM backend rejects other values. Keep the override in the run record; it is not a performance recommendation.

## Scope and validation

The implementation is in `glm53_setup.runtime.candidate_order`, wired by the source-pinned reference patch into the GLM path of `FLASHINFER_MLA_SPARSE_SM120`. It is not a global change to all vLLM top-k operators or all model backends. No Euryale package is required by this runtime code.

CPU contracts cover all permutations, duplicate IDs, padding, empty rows, noncontiguous tensors, integer limits, immutability and logical-before-physical mapping. GPU component checks and fixture evidence remain separate from full-model/TP=2 qualification. This change does not guarantee complete numerical reproducibility of other layers. Tied *membership* at the 512th pool is settled separately by [`runtime.stable_indexer_topk`](server-configuration.md#repeatability-switches). See [validation](validation.md) for the current qualification limits.

### GB10 regression and cost measurements

On 2026-09-13–14 (Asia/Tokyo), the patched image `sha256:2051793f66cfe6e352104bfec38348bf2e75756bc06892d449675bc7759fcd43` passed the GPU candidate/mapping component checks. The source-pinned Docker build completed successfully. The local CPU suite ran 172 tests with 13 pinned-runtime-dependent skips; the new tensor contracts ran, rather than being skipped.

The one-GB10, four-layer, Marlin W4A16 fixture used C1/TP1/eager/APC-off, a 512 MiB KV budget and a 32,768-token model limit. The test harness held its other numerical controls constant: 13-row projection/mHC/logit geometry, 8-query NoPE groups and a 4-warp KDA triangular solver. These are test controls, not additional changes installed by this candidate-order patch. Internal tensor capture and full-state audit hooks were disabled. Speculative checks retained the boundary-repair adapter and scripted proposer, including its control-file overhead.

- Ordinary generation: inputs 3/129/513/2051/2052/16384 with 32 output tokens, plus input 129 with 128 outputs. All seven ON cases repeated identically over six runs each. OFF reproduced the 16K repeat difference; its failed repeat check is retained as a negative control.
- Speculative verification: a capacity-12 engine with effective widths 7/9/12, inputs 129/2051/16384, and oracle/first-rejection/middle-rejection cases. All 27 cases, repeated three times each, matched the new ordinary baseline and exercised the requested first-step acceptance/rejection. This used a scripted proposer, not learned Euryale weights or the standard MTP draft, and is not a draft-quality or MTP-speed result.

Ordinary request-time medians below exclude one warmup and use five timed runs. All times include prefill and generation. Some OFF/ON token sequences differ, and OFF's 16K repetitions diverge; this is a same-workload-shape cost comparison, not an identical-output speedup claim.

| Input / output tokens | OFF | ON |
|---|---:|---:|
| 3 / 32 | 480.52 ms | 479.82 ms |
| 129 / 32 | 509.93 ms | 513.07 ms |
| 513 / 32 | 603.01 ms | 606.46 ms |
| 2051 / 32 | 976.93 ms | 982.52 ms |
| 2052 / 32 | 975.74 ms | 982.08 ms |
| 16384 / 32 | 4443.69 ms | 4466.76 ms |
| 129 / 128 | 1925.63 ms | 1937.21 ms |

The observed median change ranged from -0.15% to +0.65%; the slight negative value is not evidence of a speed gain. In a separate candidate-order microbenchmark using captured IDs and 2,176 columns, CUDA-event time per call was about 0.043–0.047 ms for 1/8/10/13 rows and 0.292 ms for 512 rows (five samples of 100 calls, after ten warmups). Peak extra Torch allocation was 60 KiB for one row and about 21.3 MiB for 512 rows. These are transient candidate tensors, not a second KV cache.

The runtime module was confirmed inside the built image, with no candidate-order injection from the test harness. All jobs completed without OOM. The measurements above apply to the fixture controls. Subsequent full-model regression is recorded separately below.

### Full-model TP=2 combined regression

On 2026-09-14 (Asia/Tokyo), both ranks used rebuilt image `sha256:f6fc154c5b5397e694fb20b9c150bced6b1a049dd5e4b1bf6a7ca642160def7a`: C1, eager, 32K, MTP k=3, fusion, async index checks, dense APC retention and LPA. Both ranks passed the 172-test CPU suite with six skips.

Ordinary / LPA / restored task scores were 22 / 24 / 21 out of 24, with no LPA-only regression. All three tool round trips, cancellation followed by another request, and 32,704 input + 64 output capacity cases passed; capacity preemption deltas were zero. History passed independent rescoring of 61 answers, nine boundaries and eviction. A separate 30,100-token midpoint edit restored H=9,216 and answered correctly in all three arms.

A same-request replay of eight existing heldout cases scored 8/8 in each arm, aggregated from request-ID-audited cases, part of them rerun; this is not a fresh unused test set. Private evidence: `records/20260913-mia-updates/canonical-v78-assessment.json`.

For 128 output tokens, three measured requests after one warmup had median total seconds of 12.215 / 10.366 / 10.969 at 2,048 input; 28.205 / 24.529 / 28.790 at 8,192; and 39.976 / 34.547 / 40.099 at 16,320 with H=4,608. Generated token sequences differ in 34 of 36 paired measurements against the older image, so elapsed differences do not isolate sorting-kernel cost or establish a speed gain. Sustained load, a trained Euryale proposer and numerical identity across all configurations remain unqualified.
