# GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ

**Short name: NVFP4 BIZ** (cite as "NVFP4 BIZ 1.20.2"). It names this serving stack, which serves NVIDIA's pinned checkpoint as distributed. The published option's weights are **NVFP4 BIZ AXL** (AXL: the attention projections and `lm_head` in W4A16; on Hugging Face as [Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16), whose repository name describes the contents). The repository names stay as they are.

**BIZ** is the maintainer's mark (Bizuayeu) and states the intent: a business-use setup with commercially usable licensing, pinned assets, recorded checks and reversible operation. What it does not mean is in the [disclaimer](#disclaimer).

[日本語](README.ja.md) · [Setup runbook](SETUP.md) · [Operations](docs/operations.md) · [Validation](docs/validation.md) · [Architecture](docs/architecture.md) · [Document map](docs/README.md)

## Summary

- **What it is.** A community setup and validation toolkit that serves NVIDIA's GLM-5.3-Flash NVFP4 checkpoint on **two DGX Spark or compatible GB10 systems**, partitioned TP=2 over a QSFP/RoCE link, through a pinned vLLM built into a reference image. Published measurements come from two MSI EdgeXpert (MS-C931) systems. The focus is commercially usable licensing, pinned artifacts, observable checks and reversible operation.
- **Status.** **Accepted for routine use:** both profiles for one active sequence since 2026-09-22, and the published option's two-sequence profile for two sequences of up to about 200K tokens each since 2026-09-23. [SETUP step 6](SETUP.md#6-qualify-the-full-model) records what each acceptance rests on; harness acceptance is recorded per case in [harnesses](docs/harnesses.md#acceptance-matrix-and-status). Other hardware, more sequences than those and video input are outside the accepted scope ([status by scope](#status-by-scope)).
- **Two served profiles.** The **distributed defaults** serve the pinned weights exactly as NVIDIA distributes them. The **published option (NVFP4 BIZ AXL)** repacks the attention projections and `lm_head` to W4A16 for faster decode at a measured quality cost, and is an operator opt-in. [What has been verified](#what-has-been-verified) compares them and lists the status of every scope.
- **Precision.** Serving runs Marlin W4A16 on GB10. NVIDIA's model card evaluated its checkpoint under a different recipe on different hardware, so its accuracy table does not describe this stack; [validation](docs/validation.md#evidence-not-production-qualification) says which numbers do.
- **Licensing.** Apache-2.0 code; MIT weights that the operator downloads, not bundled; each artifact keeps its own terms ([licensing at a glance](#licensing-at-a-glance)).
- **Not validated.** Concurrent serving beyond the published option's two sequences, video input, full application quality, production reliability and maximum performance ([status by scope](#status-by-scope)).

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

Routine-use acceptance is a record, not a command: [SETUP step 6](SETUP.md#6-qualify-the-full-model) states its scope and where each item's evidence is. `server preflight` checks assets, fabric, image identity, exclusive use of the GPU and memory on each host before a start, and certifies neither quality nor availability ([launch checks](docs/operations.md#full-model-launch-checks)).

## What has been verified

**Distributed defaults select the serial optimized profile with image input at 256K (262,144 tokens), KV 3 GiB per rank, reserve 3 GiB and no lifetime deadline; video input is rejected.** The checks behind these defaults are [image input](docs/vision.md) and [measurements on 1.6.0](docs/benchmarks.md#measurements-on-160); [256K capacity checks](docs/benchmarks.md#real-input-checks-at-256k) cover the text-only alternative.

### Headline measurements (1.19.0)

Two GB10 systems, TP=2, FA2 prefill, one token order inside each expert, indexer top-k ties settled, MTP k=3. Two profiles: the **distributed defaults** (the pinned NVIDIA weights, exactly what the template serves) and the **published option** (the attention projections and `lm_head` repacked to W4A16 NVFP4, served through `runtime.derived_checkpoint` with the KDA input projection declared split; weights at [Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16)). The option's column is the profile the reference pair serves, the [two-sequence AXL profile](examples/server.axl.example.toml). **Both columns were measured on 1.19.0 on 2026-09-28, in one window**: the pair switched from the served option to the defaults, then to the option at one sequence (for repeatability), then back to the served option, with the same drivers and both ranks of both profiles on performance cores ([measurements on 1.19.0](docs/benchmarks.md#both-profiles-in-one-window-with-a-gpu-clock-cap-2026-09-28)). **All three nodes ran with the GPU clock capped at 2,200 MHz**, and each stage waited for temperatures to come down first. GB10 machines can power off under sustained load; the cap cost about 2% of prefill and 1–5% on long inputs ([GPU clock cap](docs/operations.md#gpu-clock-cap)). Medians of three or nine runs; ranges, conditions and every earlier version are in [benchmarks](docs/benchmarks.md). Task types are always listed counting / prose / code.

| Category | Measure | Distributed defaults (NVFP4 BIZ on the pinned weights) | Published option (NVFP4 BIZ AXL, the served two-sequence profile) |
|---|---|---|---|
| Prefill | Prefill, 38,962-token prompt | 1,233.4 tok/s | **1,259.1 tok/s** |
| Decode | Decode after a 2,048-token prompt: counting / prose / code | 32.59 / 21.12 / 28.21 tok/s | **46.73 / 30.06 / 39.48 tok/s** (32.87 / 22.43 for counting and prose while the other runs) |
| Decode | Decode, 512 tokens after a fixed short prompt | 27.18 tok/s | **42.63 tok/s** |
| Decode | sparkDash DecodeBench, 128 tokens: structured / prose / code / json | 33.24 / 25.45 / 27.53 / 26.32 tok/s | **48.04 / 31.28 / 37.34 / 34.72 tok/s** |
| Startup | Weight loading, rank 0 (`Loading weights took`, main model) | 122.6 s (`Model loading took` 265.6 s; 787–811 s before the clone) | **120.7 s** (532.0 s on 1.18.0) |
| Long input | ~200K-token input, one passphrase at the midpoint | 178.9 and 176.6 s, correct (199,652 tokens) | **170.1 s, correct** (199,649 tokens); two such requests together: 336.0 s, both correct, no preemption |
| Long input | 255,950-token input, one passphrase at the midpoint | 229.5 s, correct | **220.9 s, correct** |
| Long input | 261,573-token three-position reference, fenced prompt (the canonical form from 1.13.0) | 239.8 s, correct 3 of 3 | **235.0 s, correct 3 of 3** |
| Long input | Maximum capacity, 262,080 input + 64 output tokens | 253.0 and 252.7 s, finite logprobs | **244.8 and 244.7 s, finite logprobs** |
| Quality | Teacher-forced NLL: Japanese / English / code / mathematics | 1.5963 / 2.0241 / 0.9479 / 0.5931 | 1.6270 / 1.9946 / 0.9601 / 0.6275 (1.6645 / 2.0024 / 1.0031 / 0.6279 on the one-sequence profile, 2026-09-21) |
| Quality | tool-eval-bench, 69 standard scenarios | 91/100, three failures, Safety Gate not passed | 88/100, the same three failures, Safety Gate not passed |
| Repeatability | Identical requests at temperature 0 (`max_num_seqs = 1`) | Same completion nine times of nine (the decode check, three task types three times each); two sent together queue and each repeats its lone completion (18 of 18) | The same when served with `max_num_seqs = 1` (two sent together: 18 of 18); the decode check's completions were the same on all eight 1.19.0 launches. Not claimed for the served two-sequence profile while two requests are in flight |
| Memory | Lowest available memory on the head during the bench | 6.2 GiB (3 GiB of KV) | 7.56 GiB, 7.34 GiB during two 200K requests (6 GiB of KV) |

| | Distributed defaults | Published option |
|---|---|---|
| **Pros** | Lossless with respect to the pinned NVIDIA weights. Ships in the template with no extra download | Decode step 12–13 ms shorter on every input (78 ms mean over ten inputs at depth 3, measured on 1.7.0 on 2026-09-21). Prefill 2% faster than the defaults in the same window, the split projection having removed the earlier penalty. Identical requests still repeat bit for bit when served with `max_num_seqs = 1`, and the head keeps about 4 GiB more available at 256K |
| **Cons** | The slower decode of the two on every task type | Not lossless: NLL 1 to 6% higher on three of four texts. Outside the template: a second checkpoint to acquire and place. Above about 250K tokens it needs the slot-mapping guard that images built from 1.7.0 carry |
| **Choose it for** | Code and tool use, and any workload that must match the pinned weights | Japanese prose and other generation-heavy serial work where the NLL cost is acceptable |

How fast MTP decodes depends on how predictable the text is: in the table above, either profile decodes counting about 1.5 times as fast as prose, because more of each draft is accepted. [Depth three for both checkpoints](docs/speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21) has the acceptance lengths, and [profiles by workload](docs/optimization-overview.md#profiles-by-workload) the choice. Every decode figure has both ranks' workers on performance cores; with either rank on efficiency cores decode falls to about a third, which [`nodes[].cpuset_cpus`](docs/server-configuration.md#optional-cpu-placement) prevents ([measurements on 1.15.0](docs/benchmarks.md#measurements-on-1150)).

### Status by scope

Each row states the status and the document that owns the evidence; the narrative lives there. One active sequence unless stated otherwise.

| Group | Scope | Status |
|---|---|---|
| Tooling | Pinned checkpoint download and official checksum verification; official ARM64 image preparation and reference-image build | Implemented |
| Fixture | Candidate-preserving NoPE reference attention | GPU-tested |
| Fixture | Four-layer, single-GB10 fixture with Marlin W4A16 | Generation and state comparisons passed, including an 8,705-token input; [validation](docs/validation.md) |
| Fixture | Batch-invariant mode with the pinned SM120 sparse MLA backend | Unsupported |
| Full model | Two-host NCCL collectives on the pinned base | Tested patterns passed over RoCE; [conditions and limits](docs/nccl-validation.md) |
| Full model | 45-layer TP=2 reference profile | Loaded; basic API text/tools checked; [benchmarks](docs/benchmarks.md) |
| Full model | Identical requests at temperature 0 | With `max_num_seqs = 1`, repeat bit for bit within a launch and, from 1.12.0, across launches, through the [repeatability switches](docs/server-configuration.md#repeatability-switches) that both templates turn on; every launch after a switch is still checked (weight digest, decode check, kernel hashes). Not claimed while more than one sequence is in flight ([concurrency scope](docs/validation.md#concurrency-scope)); [how each cause was found](docs/validation.md#repeatability) |
| Full model | Image input (vision) at 256K | One synthetic image answered correctly, text/tool regressions passed, video rejected; on 1.19.0 both profiles passed the seven regression checks and answered images of up to 7,776 tokens and up to eight images in order; direct attachment in harness user interfaces not checked; [measurements and limits](docs/vision.md) |
| Full model | Long Japanese and Korean output | Six answers of 852–1,024 characters without broken characters; reasoning text not exercised; [check and limits](docs/validation.md#full-model-tp2-experimental-scope) |
| Concurrency | More than one active sequence | **Not supported** in the distributed defaults (`max_num_seqs = 1`; requests queue). The published option's example serves two sequences from 6 GiB per rank and is **accepted for routine use since 2026-09-23** for two at up to about 200K tokens each. Repetition is not claimed in this scope (a request sharing steps with another can get a different completion); serve `max_num_seqs = 1` when completions must repeat. More sequences: TP=4 recommended, TP=3 not; neither measured here. [Concurrency scope](docs/validation.md#concurrency-scope) |
| Harness | ZCode / Claude Code integration | Basic API group passed; the shared group H-01–H-11 passed on the npm ZCode CLI 3.14.1 at 262,144 tokens (2026-09-28), the accepted route. The official ZCode Desktop is **BLOCKED** and Claude Code was **skipped by decision**; reasons and per-case status in the [acceptance matrix](docs/harnesses.md#acceptance-matrix-and-status) |
| In the template | FA2 prefill (`runtime.fa2_attention`) | Adopted: prefill 2.2 times 1.5.0, one sequence's decode on the reference path, excludes LPA; [measurements](docs/benchmarks.md#measurements-on-160) |
| In the template | MTP k=3 with the BF16 draft | Depths 1 to 5 measured on ten inputs with the requantized checkpoint, 1, 3 and 4 with the pinned one; k=3 kept for both; [speculative decoding](docs/speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21) |
| In the template | Prefix caching (APC) | Accepted for the measured serial long-prefix reuse workload (experimental); [measurements](docs/benchmarks.md#independent-full-model-prefix-caching-p19) |
| In the template | Checkpoint retention | Adopted for the exact-primed mid-edit workload after history and A/B/A tests at the native interval 4,352; the block-independent `dense` setting is equivalent in the measured layout and the final combined integration is qualified separately; [contracts](docs/launch-safety.md) |
| In the template | Fused unpack, async index checks | Independently measured and enabled; [overview](docs/optimization-overview.md) |
| Optional, off | Requantized attention projections and `lm_head` (`runtime.derived_checkpoint`, P23) | The published option above; adding the shared experts was measured and not adopted; [measurements](docs/benchmarks.md#the-reference-pairs-serving-profile-attention-and-lm_head-repacked-depth-3) / [catalog](docs/optimization-catalog.md) |
| Optional, off | APC-first LPA (P22) | Calibrated, combined with MTP/fusion/async checks and checked on held-out documents; a batch opt-in; [contract](docs/apc-lpa-design.md) |
| Measured, not adopted | Expert Parallel, PP2, decode CUDA Graphs, a draft depth from acceptance history, a confidence gate on the draft, two draft-side settings | Each measured on the full model with its number in the owner document; [overview](docs/optimization-overview.md), [speculative decoding](docs/speculative-decoding.md#beyond-a-fixed-depth-2026-09-21) |
| Measured, not adopted | Cross-layer indexer reuse (CSA2, P16) | Stopped at its cost gate: the indexer is under 1% of prefill on the fixture and about 4% projected at 200K; [design and result](docs/indexer-reuse.md) |
| Not validated | Video input, full application quality, production reliability, maximum performance | **Not validated** |

The fixture keeps the original widths, experts and selected tensor bytes, but is a truncated model. It is not a language-quality benchmark. Marlin W4A16 is a different arithmetic profile from NVIDIA's W4A4 recipe. See [the evidence and limits](docs/validation.md).

## Business-use objectives (BIZ)

This project makes **`nvidia/GLM-5.3-Flash-NVFP4` on two DGX Spark-class systems easier to evaluate, adapt and operate for business use**. Its engineering work covers three connected concerns:

- **License and provenance selection:** prefer commercially usable MIT/Apache components, pin their origin and preserve notices. The [licensing guide](docs/licensing.md) distinguishes the terms for code, weights, containers and harnesses.
- **Evidence about political bias and source fidelity:** use [FreedomBench and business-context extensions](docs/freedombench.md) to examine political-topic answers, refusals and unsupported claims inserted into supplied material. Report the tested scope and failures; a benchmark score is not proof of universal ideological neutrality. Closed on 1.19.0 for both profiles on 2026-09-28 (the pinned English suite 60 of 60 on the first attempt with no refusal on each, the long-prefix pilot 6 of 6); the Japanese translation of the suite, opposed framings and evidence placement were not run.
- **Measured performance tuning:** investigate MTP, LPA, prefix caching, CUDA fusion, batching and parallel execution while checking task quality, memory and recovery. The [optimization overview](docs/optimization-overview.md) shows where each measure acts and which profile fits which workload; the [performance and quality catalog](docs/optimization-catalog.md) records candidates, evidence and deferred work as a comparison baseline for future GLM versions.

For decode acceleration, we selected **the checkpoint's standard MTP with three speculative tokens (k=3)**, without adding an external draft model, and one depth for both checkpoints; the reasoning is in [depth three for both checkpoints](docs/speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21).

Completed measurements and remaining gates are identified above and in the linked validation documents; what BIZ does not imply is in the [disclaimer](#disclaimer).

## Related research outside this repository

**Euryale** is a separate, unpublished research project on proposing several draft tokens from a frozen model's intermediate representations, taking GLM-5.3-Flash on two GB10 hosts as its first target. It is not part of this distribution and, beyond the [canonical candidate ordering](docs/candidate-order.md) that came out of it, changes nothing in this repository's checkpoint, runtime or defaults. Its first design, a light auxiliary proposer, was trained in three variants on teacher data captured from the full model and compared with the checkpoint's standard MTP under the same conditions: every candidate was slower than MTP k=3 in every measured condition, so it was not adopted. The project has moved to a DFlash-style draft with its own layers that read the target's features as keys and values, so far trained only as a pilot from stored features. A Euryale draft would replace MTP k=3 as the default speculation path only after a same-condition comparison passes its quality, performance, memory and recovery gates, judged with the [catalog's separation](docs/optimization-catalog.md#functional-acceptance-and-defaults) of functional acceptance, performance adoption, defaults and combined-mode acceptance; until then MTP k=3 remains the measured candidate.

### Other GLM-5.3-Flash recipes for DGX Spark systems

Several public recipes serve the same model on the same class of hardware with different engines, quantization and trade-offs. They are worth comparing before choosing one. This table owns their links, their licenses as read on 2026-09-18 (0xSero on 2026-09-20) and what this repository took from each; other documents cite them by name and pull request only. Code that was adapted carries its notice in [third-party notices](THIRD_PARTY_NOTICES.md).

| Recipe | License | What this repository took from it |
|---|---|---|
| [amasu/glm53-flash-cluster](https://github.com/amasu/glm53-flash-cluster), preserving the kingjones30 recipe | Apache-2.0 / MIT | **Code adapted:** the NoPE zero-padding patch structure and recipe |
| [tenhkspark/glm53-flash-nvfp4-2node](https://github.com/tenhkspark/glm53-flash-nvfp4-2node) and the [Wabi checkpoint](https://huggingface.co/tenhkspark/GLM-5.3-Flash-NVFP4-Wabi) | Apache-2.0 (code), MIT (weights) | No code, and none of its weights. Its W4A16 NVFP4 requantization of the BF16 attention projections was evaluated as P23 and grew into the published option above (attention and `lm_head`); the measurements are in the [optimization catalog](docs/optimization-catalog.md) |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | AGPL-3.0 | No code. Mechanisms and measurements: warmup ladder and its correctness canary (#268), stall detection, KV capacity readout, the NCCL channel setting, launch-safety requirements, the RoCE GID shift (#277), field runbooks |
| [sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark) | MIT | No code. The SM90 attention path and the foreign-container launch guard as reference points |
| [drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated) | Apache-2.0 | No code. Its zero-RoPE shim and reduced `index_topk` as a comparison for the attention probes |
| [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) | none | No code. Measurements and field reports: GB10 memory behaviour, power loss during checksums, mean acceptance length, concurrency results. Its 2026-09-20 note on quantizing the attention and MLP projections reached the same tensor set as P23 independently, at TP=4 and with quality unmeasured ([catalog](docs/optimization-catalog.md)) |
| [0xSero/GLM-5.3-Flash-EXL3-1x-DGX-Spark](https://github.com/0xSero/GLM-5.3-Flash-EXL3-1x-DGX-Spark) and the [EXL3 Spark mosaic](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark) | MIT (repository code), MIT (model card for the separate weights) | No code or weights adopted. Reference for the mosaic quality panel, cold/warm measurements and verification that an overlay was actually loaded. The single-Spark mcg MTP recipe and the mul1 mosaic use different artifacts and runtimes; their speed, quality and MTP results must not be combined |

## Disclaimer

- **BIZ is an intent, not a promise.** It is not a product tier, a support commitment, a warranty or a certification. Business-use readiness is an acceptance outcome for the declared scope ([status by scope](#status-by-scope)), not implied by the suffix.
- **The kpool tail ring fix is partial.** The port of [vLLM #58454](https://github.com/vllm-project/vllm/pull/58454) (`patch_kpool_ring`) is a partial fix by upstream's own account, and follow-up changes are expected ([operations](docs/operations.md#full-model-launch-checks)).
- **The tail ring depends on the MTP depth.** Its block is 4 slots without MTP, 8 for depths 1 to 4 and 16 for depth 5, so the KV-capacity breakdown and the figures recorded from a boot change with the depth ([KV capacity](docs/server-configuration.md#kv-capacity-and-ram-requirements)).
- **Repeatability references for decodes whose context passes 2,048 tokens were re-baselined on 1.19.0.** Past the indexer's `index_topk` (2,048), the ring fix can change the compressed keys of pools built during decode with MTP, so reference hashes recorded on earlier images are not a baseline for them. For a prompt past that length, the decode check's hashes on both profiles are the reference ([measurements on 1.19.0](docs/benchmarks.md#measurements-on-1190)); no reference was taken for an output that passes it.
- **`runtime.stable_indexer_topk = false` exposes the defect that [vLLM #58785](https://github.com/vllm-project/vllm/pull/58785) fixes**, a pull request still open upstream: the persistent top-k can lose candidates on overflow. Keep the key on (both templates set it).

## Next Action

Each item is a trigger and what this repository then does.

- [vLLM #56868](https://github.com/vllm-project/vllm/issues/56868) / [#56605](https://github.com/vllm-project/vllm/issues/56605) follow-ups to #58454 merge → port them as source-pinned patches.
- [vLLM #57161](https://github.com/vllm-project/vllm/pull/57161) (kpool compress rework) merges → re-read `patch_kpool_seed` and `patch_kpool_ring` against it.
- A fix equivalent to #58785 reaches the pinned vLLM → only then allow `runtime.stable_indexer_topk = false`; when [vLLM #55122](https://github.com/vllm-project/vllm/pull/55122) merges, consider replacing the local stable sort with its kernel.
- [vLLM #58979](https://github.com/vllm-project/vllm/pull/58979) (the indexer key normalised without a rank-local compiled kernel, for [#58636](https://github.com/vllm-project/vllm/issues/58636)) or an equivalent fix reaches the pinned vLLM → check the launch states on both profiles with `runtime.inductor_deterministic` off, and let the templates drop the key only if the ranks agree and the completions repeat. The draft passed that check on 2026-09-28 ([repeatability](docs/validation.md#repeatability)).
- A vLLM release contains #58454 and its successors → move the pin to it as a minor release of its own and drop the kpool patches; the move also brings [#55736](https://github.com/vllm-project/vllm/pull/55736) (decode improvements), [#55353](https://github.com/vllm-project/vllm/pull/55353) (retention as a CLI flag) and [#53007](https://github.com/vllm-project/vllm/pull/53007) (KV LCM change), each to be re-measured. It also changes the KDA prefill kernel: newer vLLM chooses FlashKDA on SM 9.x, 10.x and 12.x, GB10 included, and this checkpoint's KDA layers (head dimension 128, a bounded gate) meet its conditions, where the pinned vLLM always runs the Triton kernel. Check that the release carries [#58846](https://github.com/vllm-project/vllm/pull/58846) (FlashKDA keeps the recurrent state in fp32; v0.30.0 does not) or keep the Triton kernel with `additional_config.kda_prefill_backend = "triton"`, and take the decode hashes again either way.
- [vLLM #57128](https://github.com/vllm-project/vllm/pull/57128) (Mamba prefix-cache hits ignoring the speculative margin, [#53912](https://github.com/vllm-project/vllm/issues/53912)) merges → port it as a source-pinned patch. The serving profiles match the reported conditions (prefix caching, MTP k=3, Mamba cache mode `align`); a field report on another stack describes content from one request appearing in another, which has not been seen here.
- [vLLM #54296](https://github.com/vllm-project/vllm/pull/54296) merges and reaches the pin → drop the slot-mapping guard (`patch_slot_mapping`).
- A user request still compiles a sampling kernel after the warmup ladder → add a rung with that request's sampling settings. The ladder's temperature-0 rungs never reached the three `_topp_sb_*` kernels that 1.19.0 compiled on a client's first request; a request without a temperature gets the checkpoint's 1.0 and top_p 0.95 and compiles them, so a rung with those settings now runs. `_gumbel_sample_kernel`, seen on 1.18.0, has not been traced to its settings.
- The next minor → evaluate TP=3 on three GB10 hosts, aiming at 1,048,576 tokens per request with six such requests held at once; the head widths that do not divide by three, the KV estimate and what decides it are in [catalog P28](docs/optimization-catalog.md#performance-initiatives).

## Local data and contribution

`state/`, `records/`, credentials, site-specific configuration and weights are excluded from Git and the Docker build context. Publish reviewed summaries, not raw local logs.

[Contributing](CONTRIBUTING.md) describes CPU checks and the publication audit. [CHANGELOG.md](CHANGELOG.md) tracks changes; [LICENSE](LICENSE) and [NOTICE](NOTICE) define project licensing and attribution.
