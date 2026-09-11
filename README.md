# GLM-5.3-Flash on 2× DGX Spark Enterprise Setup

**BETA — a limited full-model TP=2 reference profile has been tested. Production and harness qualification are not complete.**

[日本語](README.ja.md) · [Setup runbook](SETUP.md) · [Operations](docs/operations.md) · [Validation](docs/validation.md) · [Architecture](docs/architecture.md)

A community setup and validation toolkit for NVIDIA's GLM-5.3-Flash NVFP4 checkpoint on two DGX Spark-class GB10 systems. The focus is commercially usable licensing, pinned artifacts, observable checks, and reversible operations.

Original project code is **Apache-2.0**. Adapted MIT and Apache notices are retained. Model weights and container dependencies keep their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). The setup does not require EXL3/TR3 weights, DFlash2 weights, or Mia's current AGPL distribution.

See [commercial use, modification and redistribution](docs/licensing.md) for permissions and obligations by artifact. [Harness integration](docs/harnesses.md) covers official ZCode and experimental Claude Code connectivity; both are required acceptance targets and remain untested.

## What works in this beta

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
| ZCode / Claude Code harness integration | Required acceptance tests defined; **not run** |
| MTP k=1 with BF16 draft, one active sequence | Basic API and matched benchmark cases passed; [setup, gains and costs](docs/speculative-decoding.md) |
| MTP k≥2, vision, full application quality, production reliability and maximum performance | **Not validated** |

The fixture keeps the original widths, experts and selected tensor bytes, but is a truncated model. It is not a language-quality benchmark. Marlin W4A16 is a different arithmetic profile from NVIDIA's W4A4 recipe. See [the evidence and limits](docs/validation.md).

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
