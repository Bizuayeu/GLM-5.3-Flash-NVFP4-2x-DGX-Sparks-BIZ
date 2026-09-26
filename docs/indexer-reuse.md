# Indexer reuse and candidate-restricted rescoring

[日本語](indexer-reuse.ja.md)

## Result: stopped at the cost gate (2026-09-21)

**Experimental components; not integrated into model serving.** On the verified four-layer fixture (one indexer layer, eager, chunk 512, one token generated) the indexer's own operations took 0.43 ms of a 501 ms prefill at 2,048 tokens, 6.26 of 1,999 ms at 8,192 and 40.2 of 8,077 ms at 32,768: 0.09%, 0.31% and 0.50%, growing about as n^1.5 while the prefill grows linearly. The full model has eleven indexer layers of the same shape, which puts the whole indexer at about 1.7% of a 32K prefill and about 4% at 200K, and the scoring and selection that reuse could remove at about half of that; the compressed-key writes and tail updates stay. That is below any improvement worth the reuse machinery, so no reuse was built. The design gated reuse on cost, then overlap, correctness, speed and tasks; it stopped at cost. The earlier 2K/8K overlap observations are in [component validation](component-validation.md#indexer-observation).

## What reuse would have had to preserve

- The NVIDIA configuration sets `indexer_types` to `full` for every layer, and `index_share_for_mtp_iteration=true` applies to draft iterations only, not to target layers.
- The [kpool indexer](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/layers/sparse_attn_indexer_kpool.py) selects `topk_tokens // index_kpool` pools and expands them into logical token candidates plus the incomplete tail: a 2,048-token budget at kpool=4 is 512 pools, not 2,048.
- The indexer also writes compressed keys and seeds/updates tail state, which later native decode needs, and the [model](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py) passes one shared Top-K buffer to its MLA layers, so selected rows must be snapshotted before another layer overwrites them. Raw physical cache addresses never cross layers, requests or ranks.
- Copying candidate IDs is not sharing KV: each GLM layer keeps its own learned projections and pool gates. [IndexCache](https://arxiv.org/abs/2603.12201) with its [reference patch](https://github.com/THUDM/IndexCache) (for GLM-5's `GlmMoeDsaForCausalLM`, not this `Glm5Next`) and [ReTopK](https://arxiv.org/abs/2607.27692) motivated the study; their figures are not GLM-5.3-Flash results.

## Retained components

- `runtime/indexer_capture.py`, attached by `runtime/indexer_worker.py` independently of LPA/MTP: scoped CUDA-event timing and logical-candidate capture on bound kpool modules, with byte/event bounds. On the four-layer fixture 2K/8K runs captured layer-3 candidates with equal native/capture/restored output tokens.
- `validation/indexer_candidates.py`: request-local selection snapshots, layer-pair validation, a candidate-only FP32 score reference and expansion of complete pools plus the unfinished tail; ties go to the lower pool ID.
- `validation/indexer_reindex.py`: standalone Triton scoring of supplied pools under the pinned FP8 32-head/128-feature contract, matched against an independent FP64 dense oracle at 1/17/128 candidates (`rtol=2e-5`, `atol=2e-4`).
- `validation/indexer_shared_pool.py`: shared candidate pools scored by the pinned native Tensor Core scorer on each target's own K/scales.

With 512 queries and 1,024 selected pools, the shared-pool path took about 0.129 ms against 0.234 ms for native scoring of 8,192 pools (about 32K tokens at kpool=4), but against 0.068 ms at 2,048 pools (about 8K), a slowdown; `benchmark_reindex` records these and checks the selected scores against native. The first per-query Triton prototype was correct but much slower than native. The CPU `indexer-overlap` command compares aligned `source`/`target` rows (`request_id`, `query_position`, `coordinate_space="logical_tokens"`, `indices`, an optional source `candidate_pool`) and rejects unaligned, non-causal or physical-slot input.

Reproduce the observer check with `python -m glm53_setup.validation.run_indexer_fixture --fixture /verified/four-layer-fixture --output /new/record`, inside the pinned GPU image with current source mounted and LPA, MTP and prefix caching off.
