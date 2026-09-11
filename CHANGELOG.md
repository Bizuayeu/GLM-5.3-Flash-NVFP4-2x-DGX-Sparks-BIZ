# Changelog

## Unreleased — beta

### Added

- A reproducible two-rank PyTorch/NCCL diagnostic with transport interpretation and bilingual instructions.
- A bilingual QSFP/NetworkManager hands-on guide covering Windows SSH access, persistent profiles, routing checks and recovery.
- Named beta distribution: **GLM-5.3-Flash on 2× DGX Spark Enterprise Setup**.
- Apache-2.0 license, preserved third-party notices, and English/Japanese entry documentation.
- A unified `python -m glm53_setup` CLI for preparation, launch inspection and validation.
- A digest-pinned reference-image build command and portable checkout paths.
- CPU CI and checks for clean-checkout CLI operation and publication boundaries.
- English/Japanese setup runbooks with prerequisites, ordered gates, acceptance checklists and AI handoff instructions.

### Changed

- Grouped operator commands, validation tools, runtime adaptations, configuration and Docker assets by responsibility.
- Centralized the model ID and revision in `config/runtime.lock.json`.
- Serialized downloads with a process lock and atomic status updates; paused downloads terminate verification waits.
- Removed local experiment links and host-specific image identities from public entry points.

### Validation status

- Two-host RoCE collective patterns passed; the final 256 MiB AllReduce measurement was 1.18–1.21 GB/s at MTU 1500, with a separate disk-checksum job active. This does not qualify full-model TP=2.
- A single-GB10, four-layer fixture passed the tested generation/state-consistency cases using Marlin W4A16 and the candidate-preserving NoPE reference path.
- Rebuilt the reorganized Docker image and repeated the GPU component and long-input fixture checks through the packaged CLI successfully.
- CUTLASS W4A4 completed generation but differed across tested execution geometries.
- Batch invariance is unavailable for the pinned SM120 sparse-MLA backend.
- **Full-model TP=2, production reliability, model quality and performance remain unvalidated.**
