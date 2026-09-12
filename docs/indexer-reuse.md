# Indexer reuse and candidate-restricted rescoring

**Experimental proposal; reuse/reindex is not enabled or GPU-validated.** It complements the [launch/throughput investigation](performance-investigation.md), but savings and quality effects are not assumed additive with LPA.

## Contracts verified in the pinned GLM

The downloaded NVIDIA configuration has `indexer_types` set to `full` for every layer entry. `index_share_for_mtp_iteration=true` applies to draft iterations, not target-layer index sharing. A schema that permits a shared mode is not evidence that this checkpoint or this serving path uses it. Keep target-layer experiments separate from the draft model.

[IndexCache](https://arxiv.org/abs/2603.12201) and its [reference patch](https://github.com/THUDM/IndexCache) motivate cross-layer selection reuse, but their listed GLM-5 architecture is `GlmMoeDsaForCausalLM`, not this hybrid `Glm5Next` implementation. Its published removal/overlap figures are not GLM-5.3-Flash results. [ReTopK](https://arxiv.org/abs/2607.27692) instead reuses historical query-support sets with current-query reranking, recent-window coverage and exact refresh. It retains KV and does not justify sharing GLM's learned KV values. Temporal reuse must also invalidate speculative/rejected-prefix entries and remain isolated by request.

The [kpool indexer](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/layers/sparse_attn_indexer_kpool.py) selects `topk_tokens // index_kpool` pools and expands them into logical token candidates, adding incomplete-tail handling. Thus a 2,048-token budget with kpool=4 is 512 selected pools, not 2,048 pools. Short-context paths may use all causal tokens without normal score/Top-K work.

The indexer also writes compressed keys and seeds/updates tail state. A prefill-only reuse path must keep those writes for subsequent native decode. The [model](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py) passes a shared Top-K buffer to its MLA layers. Snapshot selected rows before another layer overwrites them. Do not transfer raw physical cache addresses across layers, requests or ranks.

## Staged experiment

1. **Timing and capture:** measure the whole indexer as an upper bound, then isolate scoring/selection from projections and mandatory cache writes. Use timing-only runs for the prefill fraction and separate capture runs for overlap. Bound captured query positions/bytes; full hidden-state capture already exceeded the reserve with the full MTP model. Label document kind, input length, query position, rank, layer, coordinate space and native selection size.
2. **Overlap:** compare aligned logical candidates for each proposed layer pair, excluding padding and duplicate IDs. Report Jaccard and target coverage by document kind and 2K/8K input. Flag all-causal selections as trivial overlap and exclude them from claims about sparsity. Broader source Top-2K/4K candidate sets must be collected in a separate diagnostic buffer; the native Top-K output alone cannot reconstruct them.
3. **Reuse:** only after useful cost/overlap is established, replace the target's selection result at a boundary that preserves its own key/tail/cache updates. Apply an explicit request-scoped layer-pair map with restoration on errors. Do not share KV values. Native decode remains the first acceptance scope.
4. **Reindex/hierarchical pool:** score the target only inside the collected source candidate set using a candidate-aware kernel/gather path. Masking a full-context score matrix after computation does not save its scoring cost. Preserve causality, tail candidates, padding, duplicate handling and backend buffer limits. A first-layer fixed pool is a separate quality-risk variant, not automatically equivalent to adjacent-layer reuse.

NoPE and equal latent widths do not establish layer-invariant candidate selection. LPA changes attention inputs and can also change the indexer's inputs, so candidate error and LPA error interact.

Sharing main KV or pooled indexer K, as in an architecture trained for that sharing, is a different intervention from copying candidate IDs. GLM layers have their own learned projections and pool gates; NoPE does not remove causal/tail/within-pool position semantics. Do not treat the difference as only a change in query direction. The suggested Jaccard cutoffs are hypotheses, not established GLM acceptance thresholds. Begin inside the currently tested context envelope; 32K/128K/256K and temporal reuse need separate capacity, state and quality validation.

## Acceptance order

| Stage | Evidence / stop condition |
|---|---|
| Cost | Compare scoring/selection time with total prefill time. If its removable share cannot meet the already-declared practical improvement target, stop before building reuse. Do not count required cache writes as removable work. |
| Overlap | Select layer pairs and candidate budgets on validation data before quality testing; reject poorly covering pairs. Do not invent the cutoff after seeing held-out results. |
| Correctness | capture → off → reuse → restored-off, with token agreement, shared-token logprob differences and KDA/indexer/MLA state checks. A baseline/restoration disagreement invalidates causal attribution until determinism and implementation causes are separated; it is not automatically proof of a code bug. |
| Speed | LPA off/on × reuse off/on, one-output-token prefill, warmups excluded, the existing five-measurement protocol and variability reported. Reject a gain indistinguishable from noise or inferior to LPA alone. |
| Tasks | Re-run long retrieval, code and tool round trips. Preserve the existing eight-task regression comparison without calling previously used cases unseen holdouts. Use sealed, unused test documents for final acceptance, and do not retune on them. |

Training weights remain fixed. Report transport errors, truncation and missing cases separately. The CPU `indexer-overlap` command compares JSON `pairs` containing `source`/`target` rows with `request_id`, `query_position`, `coordinate_space="logical_tokens"` and `indices`; an optional source `candidate_pool` supports expanded-pool recall. It rejects unaligned/non-causal/physical-slot inputs. This analyzer is implemented; GPU capture and runtime reuse are pending.
