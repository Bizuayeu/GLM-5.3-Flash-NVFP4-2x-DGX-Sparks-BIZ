# Validation

[日本語](validation.ja.md)

## Evidence, not production qualification

These observations used a GB10 GPU, vLLM source commit `385dce36bcee42309924a5ece951a96db3dce7f2`, and NVIDIA model revision `423acf37583782c51c142d145aef733d72943d93`. Private raw runs are not distributed; this is their reviewed summary.

| Test | Observed result | Limit |
|---|---|---|
| Reference attention with real packed FP8 cache | Candidate widths 63/64/65/2048/2051/2176 checked; padding/empty rows handled; deliberate tail removal detected | Component check, not whole-model correctness |
| Four-layer checkpoint | All 3,591 selected tensors byte-verified; original widths and 288 experts retained | Not a language-quality benchmark |
| Marlin W4A16, one GPU | Load, generation, A→B→A replay, two-request batch token comparison and forced-prefill tests passed | Limited inputs, no production reliability claim |
| Marlin, 8,705-token input | Crossed the measured 8,704-token attention manager block; forced-prefill next token matched; selected logprob difference 0.0031653 | Not every boundary or full context capacity |
| CUTLASS W4A4 | Generation completed; numerical-invariance checks differed | Cause not isolated |
| Batch-invariant mode | Rejected by SM120 sparse MLA; Triton MLA lacks sparse support | Not usable for this pinned stack |

The probability tolerance was a provisional two-BF16-epsilon bound at the reference logprob magnitude. Exact replay, token agreement and tolerance-based comparisons are distinct checks. A truncated model can amplify numerical differences.

The packaged CLI and reorganized Docker build were also checked on GB10: the real-cache component test and the Marlin four-layer test at context 16,384 passed, including the 8,705-token boundary input. Both test containers exited successfully without OOM. This verifies the new package/worker import path, not cross-run bitwise equivalence or full-model TP=2.

Marlin changes the arithmetic: its [linear kernel](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/kernels/linear/nvfp4/marlin.py) is W4A16, and its [MoE selector](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py) selects W4A16 for MARLIN independently of the generic `use_a16` flag. Do not infer precision from that flag alone.

vLLM [does not guarantee default reproducibility](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/usage/reproducibility.md). This does not prove that our adapter is correct either: the W4A4 differences remain observations requiring investigation.

## Reproduce the single-GPU fixture

Use a Linux GB10 host and a verified checkpoint. Run from the checkout root. The example uses the default Hugging Face cache; adjust the host mount if yours differs.

~~~sh
python -m glm53_setup build-reference
mkdir -p state records/fixture-check state/fixture-cache
IMAGE=$(python -c 'import json; print(json.load(open("config/runtime.lock.json"))["reference_candidate"]["tag"])')
HF_CACHE=$HOME/.cache/huggingface
REVISION=$(python -c 'from glm53_setup.config import REVISION; print(REVISION)')
~~~

Create a separate four-layer checkpoint, reading the original cache without modifying it. The fixture contains approximately 7.46 GiB of tensors.

~~~sh
docker run --name glm53-fixture-build --network none --memory 24g --memory-swap 24g -v "$HF_CACHE:/hf:ro" -v "$PWD/state:/data" --entrypoint python3 "$IMAGE" -m glm53_setup fixture-build --source "/hf/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/$REVISION" --output /data/four-layer
~~~

Use fresh output directories and container names for new experiments. Existing fixture output is never overwritten.

~~~sh
docker run --name glm53-fixture-check --gpus all --network none --memory 32g --memory-swap 32g --shm-size 2g -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_HOST_IP=127.0.0.1 -e GLOO_SOCKET_IFNAME=lo -e NVIDIA_TF32_OVERRIDE=0 -v "$PWD/state/four-layer:/fixture:ro" -v "$PWD/records/fixture-check:/out" -v "$PWD/state/fixture-cache:/root/.cache" --entrypoint python3 "$IMAGE" -m glm53_setup fixture-run --fixture /fixture --output /out --backend marlin --context 16384 --chunk 512
python -m glm53_setup fixture-assess records/fixture-check
~~~

The 24/32 GiB budgets are test limits, not full-model requirements. Set an external experiment deadline and stop the specific test container if it is exceeded. Historical tests used 15 minutes. Containers and results are preserved.

Use `--backend auto` or `--chunk 128` in a fresh run for diagnostic comparison. `passed=false` is not a passing numerical result even if all generations completed.

## CPU and component checks

~~~sh
python -m unittest discover -s tests -t . -v
python tools/check_publication.py
~~~

CPU checks cover CLI dispatch without GPU imports, checkout-relative assets, revision/launch guards, fixture selection and result assessment. CPU CI does not run GPU tests or download weights.

The CLI also exposes `inspect-runtime`, `probe-attention` and `test-reference`; use their `--help` inside the reference image. These component checks cannot substitute for TP=2 qualification.

## Remaining qualification

[FreedomBench and political-context evaluation](freedombench.md) is a required business-use evaluation item. One original-English profile has preliminary measurements; the four-profile matrix, Japanese/long-context extensions and human audits remain unqualified. The linked document owns the result and its LPA-bypass limitation.

Official ZCode and Claude Code CLI are separate, required targets in the [harness acceptance matrix](harnesses.md), which owns per-case status. The basic API smoke below feeds the API group there and does not close any client integration case.

The [two-host NCCL diagnostic](nccl-validation.md) has passed the tested collective patterns on the fixed base image. Its scope is transport and synthetic data correctness, separate from the reference attention and full model.

The [independent two-active-sequence evaluation](benchmarks.md#independent-active-batching) has limited task, throughput and 16K×2 capacity evidence. Broader full-model numerical/quality evaluation, sustained mixed load, production recovery, batching combinations, MTP depths other than k=1/k=3 and Graphs remain unvalidated; image input has only the scoped evidence in [image input at 200K](vision.md). Prefix caching has [scoped independent evidence](benchmarks.md#independent-full-model-prefix-caching-p19); APC/LPA and MTP combinations follow the separate [P22 contract](apc-lpa-design.md). Do not turn a fixture, API smoke or collective result into an unrestricted `tp2-kernel-validation` receipt.

## Full-model TP=2 experimental scope

The reference image loaded all 45 language layers on two GB10 hosts with Marlin W4A16, eager execution, one active sequence, context 16,384 and 1 GiB KV per rank. The serial TP=2 four-layer fixture passed all existing state checks. With two active fixture sequences, one greedy path diverged at a near tie; that raw diagnostic remains failed and is separate from task-level acceptance.

The full model passed basic served-ID, English/Japanese final-answer, OpenAI SSE, harmless automatic tool/argument/return, and Anthropic Messages/count_tokens smoke checks. Chat acceptance with low reasoning effort also passed these final-answer/tool criteria. Reasoning text differed on replay; it remains a diagnostic rather than a requirement for identical free-form wording. The unsupported thinking-off request caused parser/content mixing and is not an accepted configuration; see [harness settings](harnesses.md).

**Multibyte output.** ModelOpt NVFP4 checkpoints whose gate and up projections carry different global scales were reported to garble multibyte text ([vLLM #54150](https://github.com/vllm-project/vllm/issues/54150)). The pinned NVIDIA checkpoint is not affected: no full-model log shows the `w1_weight_scale_2 must match` warning, and in the four-layer fixture layer 3's gate and up scales match for all 288 experts. `server mojibake` on rank 0 monitors this anyway. It asks for Japanese and Korean answers of at least 400 characters, three times each at temperature 0, and counts U+FFFD, lone surrogates and control characters other than line breaks and tabs in both the answer and the reasoning; a short, off-language, empty, malformed or failed answer is inconclusive, never a pass, and the full answers stay in `records/<stamp>-mojibake-r0/result.json`. On the distributed 1.3.1 profile (2026-09-17) all six answers passed: 852–1,024 characters, 93–95% in the target script, none of those characters, all finished with `stop`. The reasoning field was empty at the profile's low effort, so reasoning text was not exercised.

**Requantization checks on the fixture.** Three commands compare a requantized copy of the four-layer fixture with the original. `quant-error` dequantizes every NVFP4 tensor that replaced a BF16 one and tabulates relative Frobenius error, largest absolute error and the worst output row, and confirms that all other tensors are byte-identical. `agreement-fixture` reads a fixture teacher-forced and keeps top-5 rows, full-vocabulary float32 log-probabilities and the sparse-MLA candidate sets of layer 3 at 93 query positions of an 8,192-token prompt; `agreement-compare` turns two such records into full-vocabulary KL, argmax agreement and candidate-set Jaccard. The fixture is not a language model (teacher-forced top-1 is 1–3%), so only movement against the original is read, and it has to be read against the original's own restart-to-restart difference: on the reference image two starts of the unmodified fixture differed in every one of the 93 candidate sets (mean Jaccard 0.986) while repeats inside one process were bit-identical. `agreement-fixture` also reads the eight-layer fixture, where the candidate sets of the two MLA layers are reported apart; there the unmodified fixture took a different state on each of three starts, and repeats inside one process were no longer bit-identical either (argmax agreement 0.99–1.00, full-vocabulary KL up to 0.03 on the 8,192-token prompt), with one GPU, no MTP and no prefix cache. The served full model shows the same behaviour more strongly ([`server agreement`](server-configuration.md#commands)), so it grows with depth and is not a product of two-host serving, MTP or prefix caching. `python -m glm53_setup.validation.run_repeat_trace` names the source on the fixture: the first module whose output differs between two passes is always the routed experts of some MoE layer, with everything before it, the router included, bit-identical. `moe_align_block_size` hands the Marlin MoE kernel the tokens of each expert in a different order on every call even for identical input, and the kernel's result depends on a row's position: 1 to 37 of about a million output elements move by 1e-4 to 1.5e-2, and later routers amplify that. Zeroing the kernel's scratch buffers changed nothing; fixing the order inside each expert before the kernel (`--canonical-align`) made every pass bit-identical, on all five texts. A pass with the order fixed equals an earlier plain pass bit for bit on four of the five texts and differs at the usual noise level on the fifth, so the fix changes no result beyond the noise it removes. On the fixture, with five MoE layers, the fix left prefill of 2,048 tokens unchanged (1.231 and 1.233 s plain, 1.234 and 1.235 s fixed) and slowed 128 decoded tokens by about 1% (3.090 and 3.098 s plain, 3.122 and 3.132 s fixed), about 0.05 ms per MoE layer and step. The pinned vLLM's align kernel assigns slots with an atomic add from many CUDA threads, so the order follows thread scheduling; with more than 64 experts its deterministic small-batch path is never taken. Upstream tracks this as vLLM issue #52525 with unmerged fixes (#52532, #48032). From this version the reference image installs that fix behind `runtime.canonical_moe_order` ([server configuration](server-configuration.md#distributed-defaults)), on by default. On the reference pair (TP=2, MTP k=3, image `e7a2a606…`, one restart per arm, everything else equal) the switch made `server agreement` repeat bit for bit — argmax agreement 1.0 and zero log-probability movement on all four texts, and identical NLL to four decimals between two separate runs — where the same image with the switch off agreed on 0.926–0.977 of positions. Decode was not slower: 30.81/31.27/31.06 tok/s against 31.02/25.80/31.17 with the switch off, prefill 568.9 against 572.1 tok/s, and the mean MTP acceptance length rose from 3.09 to 3.35 (3.21 to 4.00 during the 199,652-token passphrase request, answered correctly in 358.2 s), which is where the sort's cost went.

**Indexer top-k ties.** A second source of forks survived the expert order fix. With MTP depth 4 on the reference pair (requantized attention projections, FA2 prefill), nine repeats of one prose request split: log-probabilities moved from position 298 by up to 0.178 and the tokens parted at position 320, while the count and code requests, and all three at depth 3, repeated bit for bit. A trace of module fingerprints inside the serving workers named the same first differing call every time, the output projection of the second MLA layer in a five-row verification step, in 3 of 16 repeats. Its cause is upstream of it: the kpool indexer selects 512 pools per query row, and the pinned vLLM's `persistent_topk` (decode) and `top_k_per_row_prefill` return a different set of pools for the same input when pools tie across the 512th rank. On one GB10 the decode kernel gave three sets in 1,200 identical calls and the prefill kernel four in 18,000 rows; on the four-layer MTP fixture 17 of 48 identical requests differed first at the same indexer call, 0 of 36 with ties settled by the lower pool index. The same setting gave 0 of 12 in one launch and 12 of 24 in another, so not seeing the fork is no evidence of its absence, and the speculation depth only decides which positions a completion passes through. `runtime.stable_indexer_topk` ([server configuration](server-configuration.md#distributed-defaults)) installs that tie rule. On the reference pair (image `1b7dc6fa…`, depth 4) prose, count and code then repeated nine of nine with zero log-probability movement; decode was 24.68 tok/s on prose and 45.99 on count against 24.28 and 45.13 before, prefill 1,248.1 against 1,255.5 tok/s on 38,962 tokens, and the 199,652-token passphrase request was answered correctly in 169.1 s against 164.3 s, with at least 10.09 GiB available on the head throughout. Repeats across launches were not measured.

[Official vLLM synthetic benchmarks](benchmarks.md) completed all planned measured requests. Test containers were stopped afterward. These results do not establish full application quality or certify production deployment.

The [MTP k=1 candidate](speculative-decoding.md) subsequently passed its small loading fixture, all five matched full-model benchmark cases and the same 11 basic API checks. A separate metadata view preserves the original checkpoint while excluding its BF16 MTP layer from global NVFP4. The guide records draft acceptance, additional memory, workload-dependent gains and the unresolved distributed shutdown limitation. Actual ZCode and Claude Code acceptance remains separate.

The same full-model benchmark and API cases also passed with k=3. Its [measured comparison](speculative-decoding.md#measured-k3-comparison) records per-position acceptance and effective cache alignment; it is preferred for further experimental evaluation but does not establish an optimal speculative depth or close the remaining gates.
