# MTP speculative decoding — experimental

[日本語](speculative-decoding.ja.md) · [Baseline benchmarks](benchmarks.md)

The first TP=2 baseline used no speculation. The speculative profiles use the MTP tensors already in the pinned NVIDIA checkpoint. No external draft model, EXL3 conversion or DFlash2 weights are needed; this introduces no additional model license. Existing [artifact licenses](licensing.md) still apply.

Both k=1 and k=3 now have limited experimental evidence. Use [the k=3 comparison](#measured-k3-comparison) for the current candidate choice; k=1 remains the comparison baseline.

## Why a flag alone is insufficient

The fixed checkpoint contains 889 MTP tensors at layer 45: 888 BF16 and one F32, totaling 13.844 GiB. Standard MTP configuration inherits `modelopt_fp4`, while the supplied quantization exclusions do not cover the MTP layer. Applying that global NVFP4 interpretation to these floating-point tensors is incorrect.

The tested candidate adds `*.layers.45.*` to the modern and legacy quantization exclusion metadata **in a separate view**. Original weights and snapshot metadata remain unchanged. Layers 0–44 retain their original quantization. The draft uses Triton unquantized MoE; the target continues using Marlin W4A16. A draft-only metadata view is insufficient for this pinned loader, which also consumes the target's `VllmConfig`.

## Prepare a view on each Linux host

First complete official checksum verification of the original snapshot. From this checkout, create a fresh view under the same Hugging Face cache root so relative links survive a common container mount:

```sh
HF_ROOT="$HOME/.cache/huggingface"
REVISION=$(python -c 'from glm53_setup.config import REVISION; print(REVISION)')
python tools/prepare_mtp_view.py \
  --snapshot "$HF_ROOT/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/$REVISION" \
  --output "$HF_ROOT/local-views/glm53-mtp-compatible/$REVISION"
```

The tool checks the declared MTP tensor headers and refuses quantized MTP data rather than silently treating it as BF16. Header inspection is not a substitute for the preceding full checksum. It creates links and modified metadata, not a second copy of the tensor data. Keep the emitted provenance/hash report privately and compare the view configuration on both nodes. Use a new destination; existing views are preserved.

Serve the **view path**, with the existing tested TP=2 reference image/settings, and add:

```sh
--speculative-config "$(cat examples/speculative.mtp1.json)"
```

The JSON is the [k=1 candidate](../examples/speculative.mtp1.json). For the tested k=3 profile, use [speculative.mtp3.json](../examples/speculative.mtp3.json) instead; it changes only `num_speculative_tokens` to 3. Both select `mtp` and a separate `triton` MoE backend for the draft. Keep the target's Marlin configuration, eager mode, one active sequence and the benchmark's context/KV settings, with prefix caching disabled as in that baseline (the distributed template enables APC; see [server configuration](server-configuration.md#distributed-defaults)). Mount the cache root read-only, including both original snapshot and view; mounting the view alone breaks its links.

This is an experimental recipe, not an acceptance for routine use. To return to the baseline, use the original snapshot and omit the speculative configuration. Preserve the view and evidence.

## Acceptance and comparison

- Confirm the target remains NVFP4/Marlin while the MTP modules load as unquantized floating-point layers; never fill nonexistent quantization scales with dummy values.
- Record actual memory and maintain host headroom. Tensor payload division alone predicts about 6.92 GiB per TP rank, but replicated parameters, temporary storage and KV require measurement.
- Repeat the same official benchmark workload and independent completed-request/output-count checks used for the baseline. Keep warmup and client concurrency explicit.
- Save raw `/metrics` snapshots before/after each case. Draft acceptance is the accepted-draft-token delta divided by the draft-token delta; conventional mean acceptance length includes the bonus token: `1 + accepted_token_delta / draft_count_delta`. The case window may include warmup and the client's initial probe, so it is distinct from measured-request-only timing. See [vLLM's metric definitions](https://docs.vllm.ai/en/v0.24.0/api/vllm/v1/spec_decode/metrics/).
- Check final answers, tools, SSE, EOS/length termination and state behavior. Keep greedy token/logprob differences as numerical diagnostics instead of demanding identical free-form reasoning text.
- Compare both prefill latency and decode/aggregate throughput. A faster decode path can still lose on long-input, short-output workloads. Increase k only in a separate test after k=1 passes.

## Measured k=1 results

On 2026-09-12 (Asia/Tokyo), the full model completed all five baseline workloads: 21 measured requests and 1,344 output tokens, with no request errors. Each request produced the requested 64 tokens. The image, target arithmetic, network and workload settings match the [MTP-off baseline](benchmarks.md#initial-full-model-results); this is a comparison between separate runs, not a repeated A/B/A study. Client concurrency 2 still queues behind server `max_num_seqs=1`.

| Input tokens | Client concurrency | MTP median TTFT (s) | Decode off → k=1 (token/s) | Aggregate output off → k=1 (token/s) | Draft acceptance |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 0.267 | 14.29 → 24.14 | 13.71 → 22.15 | 92.4% |
| 2,048 | 1 | 6.522 | 14.15 → 22.37 | 6.06 → 6.85 | 91.0% |
| 8,192 | 1 | 26.257 | 13.96 → 20.55 | 2.21 → 2.18 | 70.5% |
| 32 | 2 | 3.116 | 14.27 → 23.54 | 13.71 → 21.38 | 85.7% |
| 2,048 | 2 | 15.820 | 14.22 → 21.49 | 6.08 → 6.80 | 84.0% |

Decode is `1000 / mean_tpot_ms`. Acceptance uses the per-case counter windows described above, including client preparation/warmup where present, not just the timed requests. Accepted/drafted counts were respectively 122/132, 122/134, 105/149, 204/238 and 204/243. Mean acceptance length was 1.70–1.92 tokens per draft step.

The runtime reported 95.17 GiB model memory per rank, approximately 6.97 GiB more than the baseline. Minimum host available memory was 7.90/10.25 GiB; the 6 GiB reserve guard did not trigger, and neither rank was OOM-killed. API readiness took about 14.4 minutes in this run; cold loading remains expensive and is separate from request TTFT.

All 11 basic API checks passed, including final-answer replay, Japanese arithmetic, SSE, automatic tool arguments/return, Messages and token counting. Text responses terminated with `stop`, the tool request with `tool_calls`, and Messages with `end_turn`. Reasoning wording differed on replay and remains diagnostic. These checks establish neither full output-distribution equivalence nor broad application quality.

Short-input k=1 decode improved about 1.69× over MTP off, but 8,192-input/64-output aggregate throughput fell about 1.3% and TTFT rose from 24.387 to 26.257 seconds. Retain the MTP-off baseline for comparison and memory-constrained or prefill-heavy use. The subsequent k=3 comparison below determines the candidate for further evaluation.

Both trial servers were stopped and host memory recovered. Rank 0 exited zero; rank 1 exited 137 after the controller's Docker stop grace period, with `OOMKilled=false`. Distributed graceful shutdown/recovery remains unqualified. This experiment does not promote the routine launcher or produce its qualification receipt.

## Measured k=3 comparison

A separate run on 2026-09-12 (Asia/Tokyo) changed only the speculative-token argument from 1 to 3. The same image, checkpoint view, workload and requested context/KV/chunk settings were retained. The runtime automatically enlarged the aligned attention block from 4,352 to 4,608 tokens; this is a consequence of k=3, not a manually tuned KV budget. API readiness took about 924 seconds, including loading and initialization.

| Input tokens | Client concurrency | k=3 median TTFT (s) | Decode k=1 → k=3 (token/s) | Aggregate output k=1 → k=3 (token/s) |
|---:|---:|---:|---:|---:|
| 32 | 1 | 0.376 | 24.14 → 30.30 | 22.15 → 26.35 |
| 2,048 | 1 | 6.628 | 22.37 → 30.14 | 6.85 → 7.34 |
| 8,192 | 1 | 26.427 | 20.55 → 22.25 | 2.18 → 2.19 |
| 32 | 2 | 2.602 | 23.54 → 25.41 | 21.38 → 22.61 |
| 2,048 | 2 | 15.305 | 21.49 → 27.04 | 6.80 → 7.19 |

Per-position acceptance is accepted tokens at that position divided by all draft steps, not conditional acceptance after the previous position. The same warmup-inclusive metric windows apply.

| Case | Position 1 | Position 2 | Position 3 | Mean acceptance length |
|---|---:|---:|---:|---:|
| short-c1 | 88.2% | 77.6% | 76.3% | 3.42 |
| medium-c1 | 75.3% | 67.1% | 58.8% | 3.01 |
| long-c1 | 65.7% | 45.7% | 36.2% | 2.48 |
| short-c2 | 70.3% | 64.6% | 52.5% | 2.87 |
| medium-c2 | 73.1% | 62.8% | 54.5% | 2.90 |

All 21 measured requests produced 64 tokens without errors, and all 11 basic API checks passed. Reasoning wording again differed on replay; broad output equivalence and actual harness acceptance remain separate. Model memory stayed at 95.17 GiB/rank. Minimum host available memory was 7.33/10.12 GiB, with no reserve-triggered stop or OOM kill. Both servers were stopped; rank 1 again required Docker's stop timeout (exit 137), so recovery qualification remains open.

**Decision:** prefer k=3 for the next experimental text/tool and harness evaluations, while preserving k=1 and MTP-off references. In these samples, k=3 improved short-input decode by 25.5% and medium-input decode by 34.7% over k=1. The long-input aggregate gain was only 0.45%, too small to establish a repeatable benefit, and its TTFT increased slightly. The measurements are small, separate runs, not a statistical optimum search. k=2 and k≥4 were not tested; no claim that k=3 beats k=4 is supported. Two active sequences, graphs, prefix caching, vision, both harnesses and routine deployment remain unqualified.
