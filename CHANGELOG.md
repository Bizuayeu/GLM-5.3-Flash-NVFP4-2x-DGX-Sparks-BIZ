# Changelog

## Unreleased — beta

### Added

- Initial full-model TP=2 serial-reference benchmark results and an independent checker rejecting incomplete runs even when the benchmark CLI exits zero.
- Bilingual guides for commercial use, modification and redistribution, including upstream model MIT notices.
- Official ZCode and experimental Claude Code connection plans with separate required end-to-end acceptance cases (not yet run).
- A reproducible two-rank PyTorch/NCCL diagnostic with transport interpretation and bilingual instructions.
- A bilingual QSFP/NetworkManager hands-on guide covering Windows SSH access, persistent profiles, routing checks and recovery.
- Named beta distribution: **GLM-5.3-Flash on 2× DGX Spark Enterprise Setup**.
- Apache-2.0 license, preserved third-party notices, and English/Japanese entry documentation.
- A unified `python -m glm53_setup` CLI for preparation, launch inspection and validation.
- A digest-pinned reference-image build command and portable checkout paths.
- CPU CI and checks for clean-checkout CLI operation and publication boundaries.
- English/Japanese setup runbooks with prerequisites, ordered gates, acceptance checklists and AI handoff instructions.

### Changed

- Add prominent modification notices to generated vLLM patch outputs without changing their executable syntax tree.
- Grouped operator commands, validation tools, runtime adaptations, configuration and Docker assets by responsibility.
- Centralized the model ID and revision in `config/runtime.lock.json`.
- Serialized downloads with a process lock and atomic status updates; paused downloads terminate verification waits.
- Removed local experiment links and host-specific image identities from public entry points.

### Validation status

- Full 45-layer TP=2 loading, final-answer/tool API smoke and 21 measured benchmark requests completed under the documented serial reference profile. Harnesses and routine deployment remain unqualified.
- Thinking-off is not used for the fixed GLM template. Final-answer/tool acceptance is distinct from exact reasoning-text and cross-batch numerical diagnostics; raw mismatches are preserved.
- Two-host RoCE collective patterns passed; the final 256 MiB AllReduce measurement was 1.18–1.21 GB/s at MTU 1500, with a separate disk-checksum job active. This does not qualify full-model TP=2.
- A single-GB10, four-layer fixture passed the tested generation/state-consistency cases using Marlin W4A16 and the candidate-preserving NoPE reference path.
- Rebuilt the reorganized Docker image and repeated the GPU component and long-input fixture checks through the packaged CLI successfully.
- CUTLASS W4A4 completed generation but differed across tested execution geometries.
- Batch invariance is unavailable for the pinned SM120 sparse-MLA backend.
- **Full-model TP=2, production reliability, model quality and performance remain unvalidated.**
