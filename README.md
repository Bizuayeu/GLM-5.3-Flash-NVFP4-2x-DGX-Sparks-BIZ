# GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ

**BIZ** is the maintainer's mark (Bizuayeu) and states the intent: a business-use setup with commercially usable licensing, pinned assets, recorded checks and reversible operation. It is not a product tier, a support commitment, a warranty or a certification.

**BETA — a limited full-model TP=2 reference profile has been tested. Production and harness qualification are not complete.**

[日本語](README.ja.md) · [Setup runbook](SETUP.md) · [Operations](docs/operations.md) · [Validation](docs/validation.md) · [Architecture](docs/architecture.md) · [Document map](docs/README.md)

A community setup and validation toolkit for NVIDIA's GLM-5.3-Flash NVFP4 checkpoint on **two DGX Spark or compatible GB10 systems**. Measurements use **two MSI EdgeXpert (MS-C931) systems**. The focus is commercially usable licensing, pinned artifacts, observable checks, and reversible operations.

## What you deploy and supported hardware

The stack is **Z.ai's original model → NVIDIA's distributed NVFP4 checkpoint → this repository's GB10 runtime adaptation and validation tools**.

| Item | Deployment information |
|---|---|
| Original model | [Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| Checkpoint and download source | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4); [runtime.lock.json](config/runtime.lock.json) owns the fixed revision |
| This distribution's role | Acquire and verify weights, adapt the runtime for GB10, launch and evaluate performance/quality. Preserve NVIDIA's base checkpoint without project-specific requantization or fine-tuning |
| Hardware | Two Linux ARM64 systems, each with GB10, 128 GB-class unified memory and NVIDIA GPU-enabled Docker. TP=2 partitions the model over a QSFP/RoCE connection |
| Tested scope | Published measurements are from MSI EdgeXpert. Other DGX Spark-compatible systems require driver/GPU/memory/fabric checks in the [setup runbook](SETUP.md#1-collect-inputs-and-inspect-both-hosts); a product name alone does not qualify them. Windows supports management/CPU checks; inference runs on the Linux hosts |
| Storage | Weights live in each Linux host's Hugging Face cache. Reserve approximately 205 GB of disk per host plus images and working space. Each host stores the complete checkpoint even with TP=2; partitioning happens at load time. [Paths and verification](docs/operations.md#artifact-storage-and-paths) |

The distribution contains source, pinned references and build instructions. Weights and built Docker images are acquired/built separately. MTP uses checkpoint-provided tensors through a separate metadata view; LPA uses a separately trained auxiliary projector. See [artifact roles and storage](docs/operations.md#artifact-storage-and-paths).

NVFP4 names the downloaded weight format. The tested reference profile executes with Marlin **W4A16**, which differs from NVIDIA's W4A4 recipe. See [precision and validation scope](docs/validation.md).

Licensing at a glance. Each artifact keeps its own terms; obligations and the rationale are in the [licensing guide](docs/licensing.md), provenance in the [third-party notices](THIRD_PARTY_NOTICES.md).

| Artifact | License | Where it comes from |
|---|---|---|
| Original setup code and documents | **Apache-2.0** | This repository |
| GLM-5.3-Flash NVFP4 weights | **MIT** (stated in the pinned NVIDIA model card; upstream Z.ai model is MIT) | Downloaded by the operator; not bundled |
| Built container image | Per bundled component (CUDA, Torch, NCCL and others); not treated as one blanket license | Built by the operator from the pinned official base image |
| ZCode / Claude Code harnesses | Each product's own terms | Installed separately; nothing is relicensed here |

Distributing this repository as source, pinned references and build steps requires only Apache-2.0 compliance. Redistributing weights or built images adds those artifacts' conditions. Adapted MIT and Apache notices are retained. The setup does not require EXL3/TR3 weights, DFlash2 weights, or Mia's current AGPL distribution.

See [commercial use, modification and redistribution](docs/licensing.md) for permissions and obligations by artifact. [Harness integration](docs/harnesses.md) covers official ZCode and experimental Claude Code connectivity; both are required acceptance targets, and that document is the single source for their run status.

## Business-use objectives (BIZ)

This project makes **`nvidia/GLM-5.3-Flash-NVFP4` on two DGX Spark-class systems easier to evaluate, adapt and operate for business use**. Its engineering work covers three connected concerns:

- **License and provenance selection:** prefer commercially usable MIT/Apache components, pin their origin and preserve notices. The [licensing guide](docs/licensing.md) distinguishes the terms for code, weights, containers and harnesses.
- **Evidence about political bias and source fidelity:** use [FreedomBench and business-context extensions](docs/freedombench.md) to examine political-topic answers, refusals and unsupported claims inserted into supplied material. Report the tested scope and failures; a benchmark score is not proof of universal ideological neutrality. Full evaluation is still pending.
- **Measured performance tuning:** investigate MTP, LPA, prefix caching, CUDA fusion, batching and parallel execution while checking task quality, memory and recovery. The [optimization overview](docs/optimization-overview.md) shows where each measure acts and which profile fits which workload; the [performance and quality catalog](docs/optimization-catalog.md) records candidates, evidence and deferred work as a comparison baseline for future GLM versions.

For decode acceleration, we selected **the checkpoint's standard MTP with three speculative tokens (k=3)**, without adding an external draft model. The example uses three tokens when MTP is enabled, based on the [comparison against k=1](docs/speculative-decoding.md#measured-k3-comparison). The distributed startup template enables this serial optimized profile; see [startup defaults and required assets](docs/startup-configuration.md#distributed-defaults).

Business-use readiness is an acceptance outcome, not implied by the BIZ suffix. Completed measurements and remaining gates are identified below and in the linked validation documents.

## What works in this beta

**Distributed defaults select the serial optimized profile at 256K, KV 3 GiB per rank, reserve 3 GiB and no lifetime deadline.** See [release candidate measurements](docs/benchmarks.md#release-candidate-measurements) for the earlier speed/tool-eval results, including the unmet Safety Gate; [256K capacity checks](docs/benchmarks.md#real-input-checks-at-256k) validate the new context/KV defaults.

[One startup TOML](docs/startup-configuration.md) groups context, cache, MTP, LPA, generation and per-node settings for the experimental reference launcher and client.

[LPA (late-prefill approximation)](docs/lpa.md) ships disabled in the distributed startup template and is a batch opt-in, because an approximated request publishes nothing to the shared prefix cache. Teacher replay, corpus sampling and projector fitting tools are included for that path. Its quality/speed acceptance is separate from the baseline below.

| Scope | Status |
|---|---|
| Pinned checkpoint download and official checksum verification | Implemented |
| Official ARM64 image preparation and reference-image build | Implemented |
| Candidate-preserving NoPE reference attention | GPU-tested |
| Four-layer, single-GB10 fixture with Marlin W4A16 | Tested generation and state comparisons passed, including an 8,705-token input |
| Default CUTLASS W4A4 fixture | Generation completed; tested numerical-invariance criteria were not met |
| Batch-invariant mode with the pinned SM120 sparse MLA backend | Unsupported |
| Two-host NCCL collectives on the pinned base | Tested patterns passed over RoCE; [measured conditions and limits](docs/nccl-validation.md) |
| Full 45-layer TP=2 reference profile, one active sequence | Loaded; basic API text/tools checked; [initial benchmarks](docs/benchmarks.md) measured |
| ZCode / Claude Code harness integration | Basic API group passed; official ZCode Desktop and Claude Code client cases **not closed** — status per case in [harnesses](docs/harnesses.md#acceptance-matrix-and-status) |
| MTP k=1 / k=3 with BF16 draft, one active sequence | Basic API and matched benchmark cases passed; k=3 preferred for further experiments; [setup, gains and costs](docs/speculative-decoding.md) |
| Prefix caching (APC), APC-first LPA and checkpoint retention, one active sequence | APC accepted for the measured serial long-prefix reuse workload (experimental), enabled in the startup template; APC-first LPA calibrated, combined with MTP/fusion/async checks and checked on held-out documents; checkpoint retention adopted for the exact-primed mid-edit workload after history and A/B/A tests at the native interval 4,352, with the block-independent `dense` setting equivalent in the measured layout and its final combined integration qualified separately; [measurements](docs/benchmarks.md#independent-full-model-prefix-caching-p19) and [contracts](docs/launch-safety.md) |
| Two-active-sequence batching, Expert Parallel, PP2, fused unpack, async index checks | Independently measured; two sequences and async checks accepted within scope, EP and PP2 not adopted, fused unpack enabled in the startup template; [overview](docs/optimization-overview.md) |
| Other MTP depths, vision, full application quality, production reliability and maximum performance | **Not validated** |

The fixture keeps the original widths, experts and selected tensor bytes, but is a truncated model. It is not a language-quality benchmark. Marlin W4A16 is a different arithmetic profile from NVIDIA's W4A4 recipe. See [the evidence and limits](docs/validation.md).

[Canonical sparse candidate ordering](docs/candidate-order.md) is a shared GLM runtime change, enabled by default in newly built reference images. Updating source requires a rebuild; existing images and containers do not acquire the patch automatically. The [setup runbook](SETUP.md#4-prepare-images-and-test-the-reference-implementation) makes this an explicit installation step. The initial optimization comparisons used pre-change images. The candidate-order guide above records the subsequent full-model combined regression; [release measurements](docs/benchmarks.md#release-candidate-measurements) cover the earlier 200K combination; [256K checks](docs/benchmarks.md#real-input-checks-at-256k) cover the current context/KV defaults. A runtime rebuilt at another site still needs qualification.

## Related research outside this repository

**Euryale** is a separate, unpublished research project that proposes several draft tokens from a frozen model's intermediate representations with a light auxiliary proposer, taking GLM-5.3-Flash on two GB10 hosts as its first target. It is not part of this distribution and, beyond the canonical candidate ordering described above, changes nothing in this repository's checkpoint, runtime or defaults. Its speed, quality and memory advantage over the checkpoint's standard MTP is unproven: a four-layer fixture has been checked at effective draft widths 5–12, while full-model teacher capture, proposer training and same-condition comparison have not started. The canonical candidate ordering above is a shared runtime change that came out of that work. Euryale would replace MTP k=3 as the default speculation path only after a same-condition comparison passes its quality, performance, memory and recovery gates, judged with the [catalog's separation](docs/optimization-catalog.md#functional-acceptance-and-defaults) of functional acceptance, performance adoption, defaults and combined-mode acceptance; until then MTP k=3 remains the measured candidate.

## Prerequisites

- Python 3.11+ for checkout-local tools. CPU checks run on Windows and Linux.
- Linux ARM64, NVIDIA GPU-enabled Docker and a GB10 GPU for GPU validation.
- Two suitable systems and a verified QSFP/RoCE path for the planned TP=2 configuration.
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

This repository is a checkout-based operator toolkit, not a published PyPI package.

### Prepare assets

~~~sh
python -m glm53_setup download --background
python -m glm53_setup verify-download --hf .venv/bin/hf --output records/checksum --wait
python -m glm53_setup prepare-image --background
python -m glm53_setup build-reference
~~~

Downloads reuse the Hugging Face cache. A process lock prevents overlapping downloads; status is written atomically. A paused download causes the verification wait to exit instead of resuming it. These commands explicitly start work: do not run another download while transferring the same cache.

The fixed model revision, base-image digest and local reference tag are in [config/runtime.lock.json](config/runtime.lock.json). Building the reference image does not start inference or qualify TP=2.

### Validate before serving

Follow the [single-GPU fixture procedure](docs/validation.md#reproduce-the-single-gpu-fixture). Its results distinguish completed execution, repeatability, and numerical differences.

The TP=2 launcher is **not yet qualified**. It retains a validation gate and requires measured per-node network settings. This beta does not provide a completed TP=2 qualification workflow; do not fabricate its validation receipt. Use `service plan` and `service preflight` for inspection, and see [operations](docs/operations.md) for the remaining qualification steps.

## Local data and contribution

`state/`, `records/`, credentials, site-specific configuration and weights are excluded from Git and the Docker build context. Publish reviewed summaries, not raw local logs.

[Contributing](CONTRIBUTING.md) describes CPU checks and the publication audit. [CHANGELOG.md](CHANGELOG.md) tracks changes; [LICENSE](LICENSE) and [NOTICE](NOTICE) define project licensing and attribution.
