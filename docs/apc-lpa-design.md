# APC-first LPA on the uncached suffix: implementation and validation contract

[日本語](apc-lpa-design.ja.md) · [Catalog](optimization-catalog.md)

**Status: CPU contracts, four-layer GPU shared-state isolation, full-model calibration, scoped quality and operational checks, combinations including asynchronous MTP, and the final held-out retrieval evaluation are complete.** Depending on the workload, select the no-MTP prefix-reuse profile. It requires the matching image marker and still rejects APC coexistence through manual RPCs. See the [calibration and combination results](benchmarks.md#apc-first-lpa-crossover-measurement-p22), the [current LPA operating scope](lpa.md#operating-scope) and the [initiative catalog](optimization-catalog.md). Additional evaluation of history edits, branches and retention pressure is carried out separately from this acceptance.

## Goal and first-version policy

Reuse the prefix APC can restore first, and use LPA only when the uncached portion is long. [vLLM's APC description](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) also targets skipping prefill for a common prefix. The first version publishes only ordinarily computed states to the shared cache; approximate states are used only inside their own request. Sharing approximate states, or reusing them through a dedicated namespace, is outside this scope.

| Value | Definition |
|---|---|
| N | The input token count resolved by the runtime |
| H | The prefix length restorable with the required MLA, indexer/pool/tail and KDA states all present. Not the common string length, nor the longest hit of a single cache group |
| T | The tail length computed ordinarily. It is capped at N or below for short inputs |
| R | The uncached length eligible for LPA, `max(0, N - T - H)` |
| B | The crossover decision value determined by measurement. Not treated as a universal constant |

Approximate `[H, N-T)` only when `R > B`. `[0,H)` is not recomputed; the tail `[max(H,N-T),N)` and decode are computed ordinarily. A small R yields APC plus ordinary computation of the remainder. H=0 follows the same rule; no separate routing by total context length is added.

Under MTP the pinned vLLM [cache coordinator](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/v1/core/kv_cache_coordinator.py) excludes the matching final block from the hit, because the speculation is recomputed. KDA checkpoints also follow that resumption boundary. "One ordinarily computed block" therefore does not guarantee "one restorable block", and fixture priming includes an additional ordinary block. Use the value actually returned after this adjustment as H. The block width aligned at startup likewise does not reuse the no-MTP number.

## Shared-cache contract

1. Determine the first approximated position S before GPU computation and shared publication. On the initial ordinary path, S=H when LPA is selected; requests without approximation have no cap.
2. Only complete blocks whose states are entirely ordinarily computed and whose end does not exceed S may be shared. Keep each cache group's compression ratio and block boundaries, and do not round up.
3. Separate using KV/KDA states beyond S to execute the request from publishing them to the shared hash table. In-request allocation, cache updates and decode are retained.
4. The ordinarily computed tail is not shared either, because it depends on the earlier approximation. The same applies to states produced during decode.
5. Chunked continuation, preemption/recomputation, cancellation and teardown must not lose the restriction. Do not silently relax the cap of a request once affected by approximation to a later position.
   The approximation interval selected at the start does not change with ordinary chunk progress. If preemption loses the original shared prefix, reconstruct that region by ordinary computation and keep the original approximation start position. A request that was ordinary stays ordinary.
6. Verify that an LPA request does not destroy the existing KV/KDA checkpoints of the shared prefix. Keep the existing copy-on-write, free and reallocation contracts.

Processing a first-seen long document with LPA does not grow the approximated text into shared APC for the next request. An explicit per-request LPA off exists so that a common system prompt, tool definitions and documents can be primed by ordinary computation. An off request publishes only states that were in fact computed entirely ordinarily.

## Attachment points in the pinned runtime

The target vLLM is the existing lock's `385dce36bcee42309924a5ece951a96db3dce7f2`. No upstream update or fallback to another version is performed; the patched source hashes are verified.

- Take H from the scheduler's cache lookup result. `shared_prefix_boundary` may include a boundary that a lagging cache group cannot restore yet, so it is not a substitute for H.
- `KVCacheManager.allocate_slots()` calls `coordinator.cache_blocks()` before computation. Post-hoc response handling alone cannot suppress it. Apply the shared cap on both that path and explicit `cache_blocks()` calls.
- Pass the request ID, N/H/T/R, the LPA decision and the shared cap from the scheduler to the worker. Do not depend on switching only the worker's global configuration through an external RPC, and detect mismatched requests.
- With asynchronous MTP the CPU-side cursor is optimistic and corrects for the speculative tokens the GPU rejected. Use the GPU's corrected position in decode steps that have generated tokens and where both positions are at or after the end of the prompt. Prefill position agreement, the prohibition on rewinding into the prompt, position-sequence continuity and the shared publication cap are retained. The `speculative_position_corrections` diagnostic retains the number of steps where a correction was observed.
- The LPA hook uses absolute positions and approximates and counts only the uncached positions actually computed. Capture/oracle must not treat a prefix missing because of a cache hit as "collected".
- Prevent a manual RPC during APC-first mode from enabling an approximation the scheduler is not told about. Leave no path by which an ordinary APC control stores approximate state.

The first integration check uses text, eager, TP2, one sequence and local cache. Graphs, PP, an external KV connector, fine-grained Mamba prefix caching and multiple sequences are not added at the same time. Combinations with MTP, fused unpack and asynchronous checks are validated as individual combinations after the base contract passes.

## Implementation and acceptance order

In the four-layer GPU test, with N=18,432 and restored H=8,704, every inspected shared-prefix hash still matched after passing through the approximated suffix. The following ordinary request still had H=8,704, and only after ordinary computation did the shared range grow to 17,408. The 16 output tokens of the control, of the ordinary request following approximation and of the restored control matched, and the synthetic approximation's distribution difference was detected separately. This is a TP1/eager/no-MTP state-isolation result, not acceptance of the full model or of combinations.

Run the shared-cache isolation component test from the following command inside the matching image. `--fixture` takes the four-layer fixture whose tensors have all passed byte verification; `--output` takes a fresh destination. It uses one GPU with 32K context, 1 GiB KV and one sequence, so monitor the container cap and host availability separately.

```sh
python -m glm53_setup apc-lpa-fixture --fixture /fixture --output /out/validation
```

This test builds a fixture-only synthetic projector that differs from ordinary state. It checks the restoration boundary of a real cache lookup, the GPU byte hashes of the shared prefix, the actual number of omitted queries and the output of a following ordinary request. It is not used to evaluate full-model quality or the crossover. Startup warmup is excluded within an explicit range of the pinned source, and real requests require the scheduler-supplied policy. The worker's cache format check uses the `fp8_ds_mla` value normalized at model load.

Measure the crossover from a Linux host on a dedicated TP2 server with LPA/APC enabled and `lpa.break_even_tokens=0`. Do not edit the routine-operation TOML directly; align a measurement TOML across both ranks. The following example corresponds to an actual joint restoration unit of 4,352 tokens. Do not reuse it as is for configurations whose boundary changes, for instance through MTP's speculation recomputation.

```sh
python -m glm53_setup apc-lpa-benchmark \
  --config state/apc-calibration.toml \
  --corpus records/corpus/documents.jsonl --corpus-sha256 '<verified-sha256>' \
  --output records/apc-calibration --cached-prefix-tokens 4352 \
  --eligible-tokens 128 512 1024 2048 4096 8192 --repeats 5
```

Reset the shared cache before each measurement, and rebuild the prefix by ordinary computation when H>0. Check the actual N/H/R and the omitted-query count on both ranks, and keep reset, priming and diagnostic RPCs outside the timing window. Repeat ordinary/LPA/restored controls and exclude the first cycle as warmup. Use only the validation split of the specified corpus as input. Additional measurements at small remainders can specify `--cold-only --eligible-tokens 0 1 4 16 32 64`. The calibration value B=0 is not an operational recommendation.

| Stage | Work and completion criteria |
|---|---|
| Current standalone evaluation | Establish 32K capacity and timing, APC off/on/off speed, actual hits, long-document extraction, revisits, tools and cancellation. Retain the raw results |
| CPU contracts | Check the N/H/T/B boundaries, H=0, near-complete hits, consistency across multiple cache groups, the cap on every publication path, request-ID switching, and cap retention after cancellation/preemption |
| Components and the small-layer fixture | Build a prefix by ordinary computation, restore H, apply LPA to the suffix, then revisit with an ordinary request. Confirm that nothing beyond the approximation is in the shared table, that the original prefix state is unchanged, and the actual omission count |
| Contamination detection test | Use a fixture whose approximate state is deliberately distinguishable from ordinary state, and check that it does not flow into a following LPA-off request. Do not treat oracle values that merely happen to match as a pass |
| Crossover measurement | Vary R stepwise for H=0 and for H>0, and compare APC plus ordinary, APC plus LPA and restored with the auxiliary projector warmup excluded. Select B and record the conditions, the variance and the unmeasured ranges |
| Full model | Check no hit, partial hit and near-complete hit, the pool/block/T/B boundaries, evidence positions in long documents, isolation from other documents, projector settings and LPA off, tools, SSE, cancellation, capacity, and stop/recovery of both ranks |
| Final combination | Validate the combination with the already adopted MTP3, fused unpack and asynchronous checks on the same pinned assets, and record functional acceptance, performance adoption and default settings separately |

Retain, for each request, N, the actual restored H, R, the LPA decision, the first approximated position, the shared cap, the actual omitted-query count and the cache publication range. Do not enable heavy tensor hashing or the profiler during speed measurement. Keep verbatim agreement, task correctness, state diagnostics and the restored control separate for quality, and do not promote an incomplete result to a pass. Record each run's own host reserve, and do not read a fixture's validation reserve as the distributed profile's `resources.reserve_gib`.

Collect the startup options into the existing category TOML files, and document APC-first mode, B and the request option that computes common documents ordinarily. Require the matching image marker, and reject combination with manual RPCs, whose publication cap the scheduler cannot establish. Check invalid request options and injected internal policies in the input processor as well, and turn them into input errors before they enter the scheduler.
