# GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ

**Short name: NVFP4 BIZ** (cite as "NVFP4 BIZ 1.10.2"). It names this serving stack, which serves NVIDIA's pinned checkpoint as distributed. The published option's weights are **NVFP4 BIZ AXL** (AXL: the attention projections and `lm_head` in W4A16; on Hugging Face as [Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16), whose repository name describes the contents). The repository names stay as they are.

**BIZ** is the maintainer's mark (Bizuayeu) and states the intent: a business-use setup with commercially usable licensing, pinned assets, recorded checks and reversible operation. It is not a product tier, a support commitment, a warranty or a certification.

[日本語](README.ja.md) · [Setup runbook](SETUP.md) · [Operations](docs/operations.md) · [Validation](docs/validation.md) · [Architecture](docs/architecture.md) · [Document map](docs/README.md)

## Summary

- **What it is.** A community setup and validation toolkit that serves NVIDIA's GLM-5.3-Flash NVFP4 checkpoint on **two DGX Spark or compatible GB10 systems**, partitioned TP=2 over a QSFP/RoCE link, through a pinned vLLM built into a reference image. Published measurements come from two MSI EdgeXpert (MS-C931) systems. The focus is commercially usable licensing, pinned artifacts, observable checks and reversible operation.
- **Status.** The serial serving profile is **accepted for routine use since 2026-09-22, for one active sequence**. [SETUP step 6](SETUP.md#6-qualify-the-full-model) records what that acceptance rests on; harness acceptance is recorded per case in [harnesses](docs/harnesses.md#acceptance-matrix-and-status). Other hardware, more than one active sequence and video input are outside the accepted scope.
- **Two served profiles.** The **distributed defaults** serve the pinned weights exactly as NVIDIA distributes them. The **published option (NVFP4 BIZ AXL)** repacks the attention projections and `lm_head` to W4A16 for faster decode at a measured quality cost, and is an operator opt-in. [What has been verified](#what-has-been-verified) compares them and lists the status of every scope.
- **Precision.** Serving runs Marlin W4A16 on GB10. NVIDIA's model card evaluated its checkpoint under a different recipe on different hardware, so its accuracy table does not describe this stack; [validation](docs/validation.md#evidence-not-production-qualification) says which numbers do.
- **Licensing.** Apache-2.0 code; MIT weights that the operator downloads, not bundled; each artifact keeps its own terms ([licensing at a glance](#licensing-at-a-glance)).
- **Not validated.** Concurrent serving, video input, full application quality, production reliability and maximum performance ([status by scope](#status-by-scope)).

## What you deploy and supported hardware

The stack is **Z.ai's original model → NVIDIA's distributed NVFP4 checkpoint → this repository's GB10 runtime adaptation and validation tools**.

| Item | Deployment information |
|---|---|
| Original model | [Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| Checkpoint and download source | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4); [runtime.lock.json](config/runtime.lock.json) owns the fixed revision |
| This distribution's role | Acquire and verify weights, adapt the runtime for GB10, launch and evaluate performance/quality. The distributed defaults serve NVIDIA's base checkpoint as it is, without project-specific requantization or fine-tuning. Serving a requantized copy is an [optional setting](docs/server-configuration.md#distributed-defaults) that an operator enables; nothing requantized is bundled in this repository |
| Hardware | Two Linux ARM64 systems, each with GB10, 128 GB-class unified memory and NVIDIA GPU-enabled Docker. TP=2 partitions the model over a QSFP/RoCE connection |
| Tested scope | Published measurements are from MSI EdgeXpert. Other DGX Spark-compatible systems require driver/GPU/memory/fabric checks in the [setup runbook](SETUP.md#1-collect-inputs-and-inspect-both-hosts); a product name alone does not qualify them. Windows supports management/CPU checks; inference runs on the Linux hosts |
| Storage | Weights live in each Linux host's Hugging Face cache. Reserve approximately 205 GB of disk per host plus images and working space. Each host stores the complete checkpoint even with TP=2; partitioning happens at load time. [Paths and verification](docs/operations.md#artifact-storage-and-paths) |
| Server configuration | [One server TOML](docs/server-configuration.md) groups context, cache, MTP, LPA, generation and per-node settings for the launcher and client. The distributed template enables the serial optimized profile on the pinned weights; [server defaults and required assets](docs/server-configuration.md#distributed-defaults) |

The source checkout contains code, pinned references and build instructions. The base checkpoint and built Docker images are acquired/built separately. MTP uses checkpoint-provided tensors through a separate metadata view; the trained LPA auxiliary projector is available as a separate [GitHub Release asset](docs/lpa.md#download-the-trained-projector). See [artifact roles, package contents and storage](docs/operations.md#artifact-storage-and-paths).

NVFP4 names the downloaded weight format. The tested reference profile executes with Marlin **W4A16**, which differs from NVIDIA's W4A4 recipe; the accuracy table on NVIDIA's model card was measured under that recipe, on other hardware and another engine path, and is not a quality claim for this serving. See [precision and validation scope](docs/validation.md#evidence-not-production-qualification) for what the card's figures describe and which numbers describe this stack.

[LPA (late-prefill approximation)](docs/lpa.md) ships disabled in the distributed server template and is a batch opt-in, because an approximated request publishes nothing to the shared prefix cache. Teacher replay, corpus sampling and projector fitting tools are included for that path; its quality/speed acceptance is separate from the verified scope below.

### Licensing at a glance

Each artifact keeps its own terms; obligations and the rationale are in the [licensing guide](docs/licensing.md), provenance in the [third-party notices](THIRD_PARTY_NOTICES.md).

| Artifact | License | Where it comes from |
|---|---|---|
| Original setup code and documents | **Apache-2.0** | This repository |
| GLM-5.3-Flash NVFP4 weights | **MIT** (stated in the pinned NVIDIA model card; upstream Z.ai model is MIT) | Downloaded by the operator; not bundled |
| Attention and `lm_head` W4A16 repack (the published option) | **MIT**, with NVIDIA's model card beside it | Optional [Hugging Face weights](docs/licensing.md#weight-notices); outside Git |
| LPA cut32 auxiliary projector | **Apache-2.0**; training-data notices retained separately | Optional [Release asset](docs/lpa.md#download-the-trained-projector); outside Git |
| Built container image | Per bundled component (CUDA, Torch, NCCL and others); not treated as one blanket license | Built by the operator from the pinned official base image |
| ZCode / Claude Code harnesses | Each product's own terms | Installed separately; nothing is relicensed here |

Distributing this repository as source, pinned references and build steps requires Apache-2.0 compliance plus retention of the copyright and license notices of the adapted third-party code (MIT and Apache). Redistributing weights or built images adds those artifacts' conditions. The setup does not require EXL3/TR3 weights, DFlash2 weights, or Mia's current AGPL distribution. See [commercial use, modification and redistribution](docs/licensing.md) for permissions and obligations by artifact.

## Prerequisites

- Python 3.11+ for checkout-local tools. CPU checks run on Windows and Linux.
- Linux ARM64, NVIDIA GPU-enabled Docker and a GB10 GPU for GPU validation.
- Two suitable systems and a verified QSFP/RoCE path for the planned TP=2 configuration.
- Host kernel: the measurements used `6.17.0-1032-nvidia`. Current DGX OS updates install `7.0.0-1019-nvidia`, whose defaults can break two-host RoCE; keep the previous kernel or boot with `kho=off`. See [host kernel and multi-node RoCE](docs/operations.md#host-kernel-and-multi-node-roce).
- Storage on each deployment node for approximately 205 GB of model files, plus images, caches and optional fixtures. A full checkpoint does not fit one 128 GB node.

## Start from a checkout

Run these commands from this repository's root on the target Linux host:

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/huggingface.lock.txt
python -m glm53_setup --help
python -m glm53_setup --version
~~~

This repository is a checkout-based operator toolkit, not a published PyPI package. The ordered deployment gates, from host inspection to acceptance, are in the [setup runbook](SETUP.md).

### Prepare assets

~~~sh
python -m glm53_setup download --background
python -m glm53_setup verify-download --hf .venv/bin/hf --output records/checksum --wait
python -m glm53_setup prepare-image --background
python -m glm53_setup build-reference
~~~

Downloads reuse the Hugging Face cache. A process lock prevents overlapping downloads; status is written atomically. A paused download causes the verification wait to exit instead of resuming it. These commands explicitly start work: do not run another download while transferring the same cache.

The fixed model revision, base-image digest and local reference tag are in [config/runtime.lock.json](config/runtime.lock.json). Building the reference image does not start inference or qualify TP=2. [Canonical sparse candidate ordering](docs/candidate-order.md) and the other runtime patches live in the image: updating source requires a rebuild, and a runtime rebuilt at another site still needs qualification.

### Validate before serving

Follow the [single-GPU fixture procedure](docs/validation.md#reproduce-the-single-gpu-fixture). Its results distinguish completed execution, repeatability, and numerical differences.

The TP=2 reference profile is **measured and accepted for routine use (2026-09-22)**. `server preflight` checks assets, fabric, image identity, exclusive use of the GPU and memory on each host before a start; it does not certify quality or availability. See [operations](docs/operations.md#full-model-launch-checks) for what the launch checks cover and [the setup runbook](SETUP.md#6-qualify-the-full-model) for where each acceptance item's evidence is recorded.

## What has been verified

**Distributed defaults select the serial optimized profile with image input at 256K (262,144 tokens), KV 3 GiB per rank, reserve 3 GiB and no lifetime deadline; video input is rejected.** The checks behind these defaults are [image input](docs/vision.md) and [measurements on 1.6.0](docs/benchmarks.md#measurements-on-160); [256K capacity checks](docs/benchmarks.md#real-input-checks-at-256k) cover the text-only alternative, and [release candidate measurements](docs/benchmarks.md#release-candidate-measurements) keep the earlier speed/tool-eval results, including the unmet Safety Gate.

### Headline measurements (1.9.0)

Two GB10 systems, TP=2, one active sequence, FA2 prefill, one token order inside each expert, indexer top-k ties settled, MTP k=3. Two profiles are compared: the **distributed defaults** (the pinned NVIDIA weights, exactly what the template serves) and the **published option** (the attention projections and `lm_head` repacked to W4A16 NVFP4, served through `runtime.derived_checkpoint`; weights at [Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16)). Since 1.8.0 the option declares that checkpoint's KDA input projection split into `q_proj` / `k_proj` / `v_proj` and a merged `b` / `f_a` / `g_a` tail, the same weight bytes read a different way (profile tag `attn-lmhead-w4a16-splitkda`). Medians of three or nine runs; ranges, conditions and earlier versions are in [measurements on 1.6.0](docs/benchmarks.md#measurements-on-160), [1.7.0](docs/benchmarks.md#measurements-on-170), [1.7.1](docs/benchmarks.md#measurements-on-171), [1.8.0](docs/benchmarks.md#measurements-on-180) and [1.9.0](docs/benchmarks.md#measurements-on-190), which own these numbers. The 1.9.0 night (the same serving profile with `runtime.prefix_page_dedup`, six launches) decoded inside these ranges and kept older histories cached under re-sends; the numbers below are the 1.8.0 night's. The prefill, 199,652-token, 261,461-token, short-prompt, 2,048-token decode and memory rows come from the same-night three-arm run of 2026-09-22 on the 1.7.0 runtime; the 255,950-token, capacity and NLL rows are the earlier series. Task types are always listed counting / prose / code.

| Measure | Distributed defaults (NVFP4 BIZ on the pinned weights) | Published option (NVFP4 BIZ AXL) |
|---|---|---|
| Identical requests at temperature 0 | same completion nine times of nine, zero log-probability movement | same; on the 1.9.0 night five launches of six repeated one another's completions and one did not (row below) |
| Decode after a 2,048-token prompt: counting / prose / code | 32.01 / 20.67 / 26.68 tok/s | **45.44 / 30.01 / 37.17 tok/s** |
| Prefill, 38,962-token prompt | 1,232.8 tok/s (1,277.0 on the 1.7.1 night) | **1,294.8 tok/s** |
| Decode, 512 tokens after a fixed short prompt | 26.87–27.31 tok/s | **41.8 tok/s** |
| 199,652-token input, one passphrase at the midpoint | 173.5 and 173.6 s, correct | **163.7 and 163.7 s, correct** |
| 255,950-token input, one passphrase at the midpoint | 217.3 s, correct | 220.9 s, correct |
| 261,461-token three-position reference, explicit prompt | 235.9 s, correct 3 of 3 | **227.2 s, correct 3 of 3** |
| Maximum capacity, 262,080 input + 64 output tokens | 240.3 s, finite logprobs | 245.7 s, finite logprobs |
| Teacher-forced NLL: Japanese / English / code / mathematics | 1.5963 / 2.0241 / 0.9479 / 0.5931 | 1.6645 / 2.0024 / 1.0031 / 0.6279 |
| Lowest available memory on the head during the same-night bench | 5.53 GiB | 10.50 GiB |

| | Distributed defaults | Published option |
|---|---|---|
| **Pros** | Lossless with respect to the pinned NVIDIA weights. Ships in the template with no extra download | Decode step 12–13 ms shorter on every input (78 ms mean over ten inputs at depth 3). Prefill 2 to 5% faster than the defaults on the same night, the split projection having removed the earlier penalty. Identical requests still repeat bit for bit, and the head keeps about 4 GiB more available at 256K |
| **Cons** | The slower decode of the two on every task type | Not lossless: NLL 4 to 6% higher on three of four texts. Outside the template: a second checkpoint to acquire and place. Above about 250K tokens it needs the slot-mapping guard that images built from 1.7.0 carry |
| **Choose it for** | Code and tool use, and any workload that must match the pinned weights | Japanese prose and other generation-heavy serial work where the NLL cost is acceptable |

How fast MTP decodes depends on how predictable the text is: the same profile gives 45 tok/s on counting and 30 on prose. [The serving profile](docs/benchmarks.md#the-reference-pairs-serving-profile-attention-and-lm_head-repacked-depth-3), [depth three for both checkpoints](docs/speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21) and [profiles by workload](docs/optimization-overview.md#profiles-by-workload) hold the numbers and the choice.

### Status by scope

Each row states the status and the document that owns the evidence; the narrative lives there. One active sequence unless stated otherwise.

| Group | Scope | Status |
|---|---|---|
| Tooling | Pinned checkpoint download and official checksum verification; official ARM64 image preparation and reference-image build | Implemented |
| Fixture | Candidate-preserving NoPE reference attention | GPU-tested |
| Fixture | Four-layer, single-GB10 fixture with Marlin W4A16 | Generation and state comparisons passed, including an 8,705-token input; [validation](docs/validation.md) |
| Fixture | Default CUTLASS W4A4 fixture | Generation completed; tested numerical-invariance criteria not met |
| Fixture | Batch-invariant mode with the pinned SM120 sparse MLA backend | Unsupported |
| Full model | Two-host NCCL collectives on the pinned base | Tested patterns passed over RoCE; [conditions and limits](docs/nccl-validation.md) |
| Full model | 45-layer TP=2 reference profile | Loaded; basic API text/tools checked; [benchmarks](docs/benchmarks.md) |
| Full model | Identical requests at temperature 0 | Repeat bit for bit after two fixes, the token order inside each expert and ties in the indexer's top-k. On the four-layer fixture launches fell into two numerical states, and on the pair one launch in six computed in another state with the kernel tables unchanged ([measurements on 1.9.0](docs/benchmarks.md#measurements-on-190)), so a new launch is checked, not assumed; [how it was found](docs/validation.md#full-model-tp2-experimental-scope) |
| Full model | Image input (vision) at 200K and 256K | One synthetic image answered correctly at both lengths, text/tool regressions passed, video rejected; large images and direct attachment in harness user interfaces not checked; [measurements and limits](docs/vision.md) |
| Full model | Long Japanese and Korean output | Six answers of 852–1,024 characters without broken characters; reasoning text not exercised; [check and limits](docs/validation.md#full-model-tp2-experimental-scope) |
| Concurrency | More than one active sequence | **Not supported** in the accepted profile (`max_num_seqs = 1`; requests queue). Concurrent serving needs KV per sequence, which means more ranks: TP=4 recommended, TP=3 not recommended; neither measured here. [Concurrency scope](docs/validation.md#concurrency-scope) |
| Harness | ZCode / Claude Code integration | Basic API group passed; the common group H-01–H-11 ran once on the npm ZCode CLI (5 PASS, 6 PARTIAL). The official ZCode Desktop is **BLOCKED** (its bundled CLI cannot start interactively, [feedback #270](https://github.com/zai-org/feedback/issues/270)) and Claude Code was **skipped by decision** (its configuration conflicts with an Anthropic subscription setup on the same machine); the accepted harness route is the npm ZCode CLI; [acceptance matrix](docs/harnesses.md#acceptance-matrix-and-status) |
| In the template | FA2 prefill (`runtime.fa2_attention`) | Adopted: prefill 2.2 times 1.5.0, decode on the reference path, excludes LPA; [measurements](docs/benchmarks.md#measurements-on-160) |
| In the template | MTP k=3 with the BF16 draft | Depths 1 to 5 measured on ten inputs on both checkpoints; k=3 kept for both; [speculative decoding](docs/speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21) |
| In the template | Prefix caching (APC) | Accepted for the measured serial long-prefix reuse workload (experimental); [measurements](docs/benchmarks.md#independent-full-model-prefix-caching-p19) |
| In the template | Checkpoint retention | Adopted for the exact-primed mid-edit workload after history and A/B/A tests at the native interval 4,352; the block-independent `dense` setting is equivalent in the measured layout and the final combined integration is qualified separately; [contracts](docs/launch-safety.md) |
| In the template | Fused unpack, async index checks | Independently measured and enabled; [overview](docs/optimization-overview.md) |
| Optional, off | Requantized attention projections and `lm_head` (`runtime.derived_checkpoint`, P23) | The published option above; adding the shared experts was measured and not adopted; [serving profile](docs/benchmarks.md#the-reference-pairs-serving-profile) / [catalog](docs/optimization-catalog.md) |
| Optional, off | APC-first LPA (P22) | Calibrated, combined with MTP/fusion/async checks and checked on held-out documents; a batch opt-in; [contract](docs/apc-lpa-design.md) |
| Optional, off | Two-active-sequence batching | Accepted within scope; [overview](docs/optimization-overview.md) |
| Measured, not adopted | Expert Parallel, PP2, decode CUDA Graphs, a draft depth from acceptance history, a confidence gate on the draft, two draft-side settings | Each measured on the full model with its number in the owner document; [overview](docs/optimization-overview.md), [speculative decoding](docs/speculative-decoding.md#beyond-a-fixed-depth-2026-09-21) |
| Measured, not adopted | Cross-layer indexer reuse (CSA2, P16) | Stopped at its cost gate: the indexer is under 1% of prefill on the fixture and about 4% projected at 200K; [design and result](docs/indexer-reuse.md) |
| Not validated | Video input, full application quality, production reliability, maximum performance | **Not validated** |

The fixture keeps the original widths, experts and selected tensor bytes, but is a truncated model. It is not a language-quality benchmark. Marlin W4A16 is a different arithmetic profile from NVIDIA's W4A4 recipe. See [the evidence and limits](docs/validation.md).

## Business-use objectives (BIZ)

This project makes **`nvidia/GLM-5.3-Flash-NVFP4` on two DGX Spark-class systems easier to evaluate, adapt and operate for business use**. Its engineering work covers three connected concerns:

- **License and provenance selection:** prefer commercially usable MIT/Apache components, pin their origin and preserve notices. The [licensing guide](docs/licensing.md) distinguishes the terms for code, weights, containers and harnesses.
- **Evidence about political bias and source fidelity:** use [FreedomBench and business-context extensions](docs/freedombench.md) to examine political-topic answers, refusals and unsupported claims inserted into supplied material. Report the tested scope and failures; a benchmark score is not proof of universal ideological neutrality. Closed for the serving profile on 2026-09-22 (the pinned English suite 60 of 60 on the first attempt with no refusal, the long-prefix pilot 6 of 6); the Japanese translation of the suite, opposed framings and evidence placement were not run.
- **Measured performance tuning:** investigate MTP, LPA, prefix caching, CUDA fusion, batching and parallel execution while checking task quality, memory and recovery. The [optimization overview](docs/optimization-overview.md) shows where each measure acts and which profile fits which workload; the [performance and quality catalog](docs/optimization-catalog.md) records candidates, evidence and deferred work as a comparison baseline for future GLM versions.

For decode acceleration, we selected **the checkpoint's standard MTP with three speculative tokens (k=3)**, without adding an external draft model, and one depth for both checkpoints; the reasoning is in [depth three for both checkpoints](docs/speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21).

Business-use readiness is an acceptance outcome, not implied by the BIZ suffix. Completed measurements and remaining gates are identified above and in the linked validation documents.

## Related research outside this repository

**Euryale** is a separate, unpublished research project that proposes several draft tokens from a frozen model's intermediate representations with a light auxiliary proposer, taking GLM-5.3-Flash on two GB10 hosts as its first target. It is not part of this distribution and, beyond the [canonical candidate ordering](docs/candidate-order.md) that came out of it, changes nothing in this repository's checkpoint, runtime or defaults. Its speed, quality and memory advantage over the checkpoint's standard MTP is unproven: a four-layer fixture has been checked at effective draft widths 5–12, while full-model teacher capture, proposer training and same-condition comparison have not started. Euryale would replace MTP k=3 as the default speculation path only after a same-condition comparison passes its quality, performance, memory and recovery gates, judged with the [catalog's separation](docs/optimization-catalog.md#functional-acceptance-and-defaults) of functional acceptance, performance adoption, defaults and combined-mode acceptance; until then MTP k=3 remains the measured candidate.

### Other GLM-5.3-Flash recipes for DGX Spark systems

Several public recipes serve the same model on the same class of hardware with different engines, quantization and trade-offs. They are worth comparing before choosing one. This table owns their links, their licenses as read on 2026-09-18 (0xSero on 2026-09-20) and what this repository took from each; other documents cite them by name and pull request only. Code that was adapted carries its notice in [third-party notices](THIRD_PARTY_NOTICES.md).

| Recipe | License | What this repository took from it |
|---|---|---|
| [amasu/glm53-flash-cluster](https://github.com/amasu/glm53-flash-cluster), preserving the kingjones30 recipe | Apache-2.0 / MIT | **Code adapted:** the NoPE zero-padding patch structure and recipe |
| [tenhkspark/glm53-flash-nvfp4-2node](https://github.com/tenhkspark/glm53-flash-nvfp4-2node) and the [Wabi checkpoint](https://huggingface.co/tenhkspark/GLM-5.3-Flash-NVFP4-Wabi) | Apache-2.0 (code), MIT (weights) | No code, and none of its weights. Its W4A16 NVFP4 requantization of the BF16 attention projections was evaluated as P23 and grew into the published option above (attention and `lm_head`); the measurements are in the [optimization catalog](docs/optimization-catalog.md) |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | AGPL-3.0 | No code. Mechanisms and measurements: warmup ladder, stall detection, KV capacity readout, the NCCL channel setting, launch-safety requirements, field runbooks |
| [sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark) | MIT | No code. The SM90 attention path and the foreign-container launch guard as reference points |
| [drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated) | Apache-2.0 | No code. Its zero-RoPE shim and reduced `index_topk` as a comparison for the attention probes |
| [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) | none | No code. Measurements and field reports: GB10 memory behaviour, power loss during checksums, mean acceptance length, concurrency results. Its 2026-09-20 note on quantizing the attention and MLP projections reached the same tensor set as P23 independently, at TP=4 and with quality unmeasured ([catalog](docs/optimization-catalog.md)) |
| [0xSero/GLM-5.3-Flash-EXL3-1x-DGX-Spark](https://github.com/0xSero/GLM-5.3-Flash-EXL3-1x-DGX-Spark) and the [EXL3 Spark mosaic](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark) | MIT (repository code), MIT (model card for the separate weights) | No code or weights adopted. Reference for the mosaic quality panel, cold/warm measurements and verification that an overlay was actually loaded. The single-Spark mcg MTP recipe and the mul1 mosaic use different artifacts and runtimes; their speed, quality and MTP results must not be combined |

## Local data and contribution

`state/`, `records/`, credentials, site-specific configuration and weights are excluded from Git and the Docker build context. Publish reviewed summaries, not raw local logs.

[Contributing](CONTRIBUTING.md) describes CPU checks and the publication audit. [CHANGELOG.md](CHANGELOG.md) tracks changes; [LICENSE](LICENSE) and [NOTICE](NOTICE) define project licensing and attribution.
