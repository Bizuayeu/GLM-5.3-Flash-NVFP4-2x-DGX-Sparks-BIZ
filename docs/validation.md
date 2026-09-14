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

Official ZCode and Claude Code CLI are separate, required targets in the [harness acceptance matrix](harnesses.md). Their end-to-end cases are NOT RUN; the basic API smoke below does not close the complete matrix or client integration cases.

The [two-host NCCL diagnostic](nccl-validation.md) has passed the tested collective patterns on the fixed base image. Its scope is transport and synthetic data correctness, separate from the reference attention and full model.

The [independent two-active-sequence evaluation](benchmarks.md#independent-active-batching) has limited task, throughput and 16K×2 capacity evidence. Broader full-model numerical/quality evaluation, sustained mixed load, production recovery, batching combinations, MTP depths other than k=1/k=3, graphs and vision remain unvalidated. Prefix caching has [scoped independent evidence](benchmarks.md#independent-full-model-prefix-caching-p19); APC/LPA and MTP combinations follow the separate [P22 contract](apc-lpa-design.md). Do not turn a fixture, API smoke or collective result into an unrestricted `tp2-kernel-validation` receipt.

## Full-model TP=2 experimental scope

The reference image loaded all 45 language layers on two GB10 hosts with Marlin W4A16, eager execution, one active sequence, context 16,384 and 1 GiB KV per rank. The serial TP=2 four-layer fixture passed all existing state checks. With two active fixture sequences, one greedy path diverged at a near tie; that raw diagnostic remains failed and is separate from task-level acceptance.

The full model passed basic served-ID, English/Japanese final-answer, OpenAI SSE, harmless automatic tool/argument/return, and Anthropic Messages/count_tokens smoke checks. Chat acceptance with low reasoning effort also passed these final-answer/tool criteria. Reasoning text differed on replay; it remains a diagnostic rather than a requirement for identical free-form wording. The unsupported thinking-off request caused parser/content mixing and is not an accepted configuration; see [harness settings](harnesses.md).

[Official vLLM synthetic benchmarks](benchmarks.md) completed all planned measured requests. Test containers were stopped afterward. These results do not unlock the current routine launcher, establish full application quality or certify production deployment.

The [MTP k=1 candidate](speculative-decoding.md) subsequently passed its small loading fixture, all five matched full-model benchmark cases and the same 11 basic API checks. A separate metadata view preserves the original checkpoint while excluding its BF16 MTP layer from global NVFP4. The guide records draft acceptance, additional memory, workload-dependent gains and the unresolved distributed shutdown limitation. Actual ZCode and Claude Code acceptance remains separate.

The same full-model benchmark and API cases also passed with k=3. Its [measured comparison](speculative-decoding.md#measured-k3-comparison) records per-position acceptance and effective cache alignment; it is preferred for further experimental evaluation but does not establish an optimal speculative depth or close the remaining gates.
