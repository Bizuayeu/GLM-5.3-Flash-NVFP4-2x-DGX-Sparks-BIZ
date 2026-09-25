# APC-first LPA on the uncached suffix: implementation and validation contract

[日本語](apc-lpa-design.ja.md) · [Catalog](optimization-catalog.md)

**Status: complete.** CPU contracts, four-layer GPU shared-state isolation, full-model calibration, scoped quality and operational checks, combinations including asynchronous MTP, and the final held-out retrieval evaluation passed ([evidence](#evidence)). LPA ships off and excludes FA2 prefill (1.6.0). The path requires the matching image marker and rejects APC coexistence through manual RPCs. History edits, branches and retention: [fixtures](component-validation.md#additional-history-fixtures), [full model](benchmarks.md#apc-history-retention-baseline). See the [current LPA operating scope](lpa.md#operating-scope) and the [initiative catalog](optimization-catalog.md).

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

## Evidence

- Shared-state isolation on the four-layer fixture, including MTP, fusion and asynchronous checks: [component validation](component-validation.md#apc-first-lpa-cache-isolation-p22).
- Crossover calibration and the threshold B: [benchmarks](benchmarks.md#apc-first-lpa-crossover-measurement-p22).
- Full model: [the combination with MTP, fusion and asynchronous checks](benchmarks.md#apclpa-with-mtp-fusion-and-asynchronous-checks-p22) and [held-out retrieval](benchmarks.md#repeated-input-tradeoff).

The fixture test runs inside the matching image on one GPU, with a byte-verified four-layer fixture and a fresh output directory. It builds a fixture-only synthetic projector, so it checks isolation, not quality or the crossover.

```sh
python -m glm53_setup apc-lpa-fixture --fixture /fixture --output /out/validation
```

The calibration runs on a dedicated TP2 server with LPA and APC enabled and `lpa.break_even_tokens=0`, from a measurement TOML aligned across both ranks. `--cached-prefix-tokens` is the actual joint restoration unit of that profile (4,352 here); a profile whose boundary differs, for instance under MTP, needs its own value.

```sh
python -m glm53_setup apc-lpa-benchmark \
  --config state/apc-calibration.toml \
  --corpus records/corpus/documents.jsonl --corpus-sha256 '<verified-sha256>' \
  --output records/apc-calibration --cached-prefix-tokens 4352 \
  --eligible-tokens 128 512 1024 2048 4096 8192 --repeats 5
```

`--cold-only --eligible-tokens 0 1 4 16 32 64` adds small remainders. The calibration value B=0 is not an operational recommendation.
