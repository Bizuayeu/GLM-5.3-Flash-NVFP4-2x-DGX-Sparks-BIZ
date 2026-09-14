# LPA: Late-prefill approximation — experimental

[日本語](lpa.ja.md) · [Baseline](benchmarks.md)

This experiment predicts the normalized attention inputs of later GLM layers, builds their cache and recurrent state with the existing attention implementation, and skips historical MLP rows. Generation still executes all language layers. The checkpoint is unchanged; only a small auxiliary projector is fitted.

The feature name is **LPA (Late-prefill approximation)**. Configuration uses `[lpa]`; commands are `lpa-fixture`, `lpa-corpus` and `lpa-train`. Rebuild images against the [current image contract](startup-configuration.md#current-image-contract); old interfaces are not retained.

The starting point was [Kishida's Qwen3 experiment](https://nowokay.hatenablog.com/entry/2026/09/11/120001). The GLM adaptation has its own state and quality tests.

GLM's sparse MLA needs its latent cache, indexer and incomplete-pool tail. KDA also needs convolution and recurrent state. The adapter retains these update paths and processes each token once. Optional `skip_mla_queries=true` omits only unused historical query evaluation in the pinned reference MLA backend; retained queries keep every candidate. Historical mHC and KDA output computation remain.

The projector uses a learned diagonal scale plus a low-rank residual map. Layer scales differ substantially, so an identity residual prior alone is inadequate for the KDA-to-MLA boundary. Artifacts contain a cut, layer count, representation version and tensors; loading validates their shapes, precision and finite values. Use the same pinned checkpoint and arithmetic as the teacher. Cached projector tensors are reused across requests.

## Operating scope

**LPA and prefix-cache reuse are mutually exclusive, so the distributed template ships `lpa.enabled = false` and this is a batch opt-in.** An approximated request publishes nothing to the shared prefix cache, and suppression continues through the exact tail and decode, because blocks computed after an approximated region still depend on approximated history. A workload that resends a growing prompt (any chat or coding harness) therefore never accumulates a reusable prefix while LPA is on: `break_even_tokens` compares one request's prefill cost and cannot see the reuse every later request forfeits. Enable LPA for a long input that is processed once, not for a conversation.

The distributed TOML enables MTP coexistence. The corrected four-layer MTP3/fused-unpack/async fixture passed eight lengths from 3 to 8,192 tokens, including active KDA state comparisons, in `integration-fixture-v36`. The subsequent [serial full-model comparison](benchmarks.md#serial-integration-of-mtp-lpa-fused-unpack-and-async-checks-p18) records its limited task, timing, capacity and cancellation acceptance. Component token agreement is not a general full-model quality claim.

- Eager, text-only, TP=2, one active sequence and one controlling client. Sequence-parallel MoE and concurrent controllers are unsupported. Without APC, MTP k=1/k=3 requires explicit `allow_mtp=true`; the [startup TOML](startup-configuration.md) wires this automatically. APC uses the separate scheduler-integrated P22 path described below.
- A configurable final prompt window is computed normally. Fully protected short prompts use the ordinary path without loading or running a projector.
- The auxiliary model changes historical state. Running all decode layers does not restore exact target-model probabilities.
- This is an experimental worker extension, not qualification of the routine deployment launcher or either coding harness. Keep its development RPC on loopback.

### APC and shared-state provenance

LPA changes cached KV and KDA state. A cache produced with approximation must not enter an LPA-off request or a different projector configuration. The manual `lpa_configure` RPC therefore rejects prefix caching: it cannot establish the scheduler's cache-publication boundary.

P22 implements [APC-first, uncached-suffix LPA](apc-lpa-design.md). The scheduler first restores a jointly usable exact prefix H, then selects the remaining approximation interval from N/H/T and a crossover threshold. Shared publication stops at the first approximation and remains stopped through the exact tail and decode. Requests can explicitly select normal computation to prime common documents. Approximate states remain request-local.

This path requires the matching image marker and startup settings. CPU contracts and four-layer GPU cache-isolation checks have passed; full-model performance, threshold calibration and MTP combinations remain under evaluation. Capture/oracle RPCs are not enabled with APC. Measure combined benefits rather than adding independent gains.

## Reproduce the components

Use the same fixed reference image as the teacher for GPU commands; CLI help and corpus sampling do not require Torch.

```sh
python -m glm53_setup lpa-corpus --output records/corpus-ja --documents 512
python -m glm53_setup lpa-corpus --subset en-wiki --output records/corpus-en --documents 128
python -m glm53_setup lpa-corpus --subset code --shard 300 --output records/corpus-code --documents 128
python -m glm53_setup lpa-fixture --fixture /fixture --output /out/oracle --cut 0 --skip-mla-queries --lengths 3 4 5 127 128 129 511 512 513 8705
python -m glm53_setup lpa-train --captures /out/teacher --output /out/projector --cut 40 --rank 256 --ridge 0.001
```

The sampling sizes and rank/ridge values are pilot parameters, not established optima. The sampler pins the LLM-jp corpus revision, retains source metadata, limits compressed bytes read and partitions normalized document hashes into train/validation/test. The code sample uses the dataset's per-repository license metadata to retain MIT/Apache/BSD/ISC entries. It does not apply the repository's Apache license to all corpus data. Preserve the emitted attribution and subset licenses; see the [LLM-jp corpus README](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3).

## Teacher collection and experimental control

The reference-image server needs `--worker-extension-cls glm53_setup.runtime.lpa.LPAWorkerExtension` and `VLLM_SERVER_DEV_MODE=1` for the private `/collective_rpc` endpoint. It accepts the named `lpa_configure` and `lpa_report` worker methods. No request may enter between configuration and the controlled request. Use the server's tokenizer and identical template settings to determine the actual prompt length.

`lpa_configure` takes `mode`, `cut`, `prompt_length`, `tail`, and, for prediction, `predictor_path`. Modes are `off`, `capture`, `oracle`, `identity`, and `predict`. The response distinguishes requested mode from effective mode. `capture` records teacher attention inputs; `oracle` reuses that exact prompt's capture to test state injection. `identity` is a diagnostic baseline, not a trained approximation.

`lpa_report(output=...)` writes `rank-N/layer-L.pt` tensors under a fresh directory. The trainer expects a completed collection `result.json` containing `cut` and `cases`, with each case carrying `id`, `split`, and `prompt_tokens`; tensors live at `<id>/rank-0/layer-L.pt`. Keep complete-document splits even when selecting interior code windows. The same wider capture can train a later cut. Fit on train only, select with validation, and reserve test for final evaluation.

`profile=true` records per-layer and attention/MLP CUDA event spans. Use a single output token when isolating prefill, and keep instrumentation off for final wall-time comparisons. Communication inside those modules is included in their spans, not measured as an independent term.

Query omission is request-scoped, disabled by default, and requires the fixed reference backend. A layout mismatch or an unexpected backend call fails explicitly. Cache creation remains upstream of the read-only query calculation. The oracle fixture verifies both normal restoration and the actual number of skipped queries.

## Evidence and limits

The initial four-layer oracle experiments covered pool, convolution, chunk and cache-block boundaries. Full-model oracle trials also covered inputs through 8,192 tokens. Token/text agreement and numerical drift were recorded separately. The one-token fixture control was unstable even without approximation and is not a qualified case.

A pilot projector fitted from LLM-jp Japanese/English Wikipedia and permissively labeled C++ samples supports the experimental cut-32 profile: 13 later layers, a 512-token normal tail, and reference MLA query omission. A cut-24 candidate returned an incorrect long-context lookup value and was rejected.

| Pre-tokenized input | Ordinary path, mean of A/A-restored medians | Approximate path median | Latency reduction |
|---:|---:|---:|---:|
| 2,048 tokens | 6.174 s | 5.118 s | 17.1% |
| 8,192 tokens | 24.763 s | 19.409 s | 21.6% |

These are one-output-token requests, five measurements per condition with warmup excluded, on the serial TP=2 reference profile. Inputs came from validation documents. Configuration/tokenization were outside the timing window. This measures prefill/first-token response latency, not faster decode or a universal halving of latency.

All six active long-context/long-code functional cases and the long-context tool round-trip passed. Eight untouched documents yielded all eight lookup values correctly; one response added an explanation, so initial strict-format success was 7/8. Both paths passed four repeats of that affected case. Streaming, recovery after closing an active stream, and a cold restart followed by a long-context query passed. The baseline also exhibited format variability; the small sample is not a statistical non-inferiority guarantee.

Raw data, projector files and site-specific controllers remain private under `records/`. The ordinary deployment/harness qualification gate is unchanged.
