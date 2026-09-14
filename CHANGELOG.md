# Changelog

## Unreleased — beta

### Changed

- Rewrite the harness document as the single source for connection design, the acceptance matrix and per-case status: add a status column (API-01–03 PASS, API-04 PARTIAL, official ZCode Desktop NOT RUN / bundled CLI BLOCKED on the missing TUI package, npm distribution smoke as supporting evidence, Claude Code and shared H cases NOT RUN), separate the ZCode distributions, replace the hedged telemetry text with a per-distribution table from a static read of the bundles (npm CLI sends traces only with an OTLP endpoint set; Desktop hardcodes trace and event endpoints and strips inherited switches), align the example port with the startup template, and add a rule to deny harness writes to its own config directory. README and validation now point to the matrix instead of restating status. Record a dated TCP sample for the npm distribution (only the loopback tunnel observed during one headless prompt) as supporting evidence under H-09.
- Rename the distribution from “GLM-5.3-Flash on 2× DGX Spark Enterprise Setup” to **GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ**. BIZ marks the maintainer and a business-use intent; the word “enterprise” was read as implying support or certification, so prose now says “business use”. Stable identifiers are unchanged: the reference image tag `glm53-enterprise:reference` (baked into existing records) and catalog IDs E01–E03. The Python distribution name becomes `glm53-flash-biz-setup`.
- Align distributed startup defaults with the serial optimized profile: 256K context, 3 GiB KV per rank, MTP3, LPA cut32/tail512/B128, APC with dense checkpoint retention, fused unpack and asynchronous index checks. Disable the lifetime deadline by default (`run_seconds=0`) while retaining memory supervision with a host memory reserve (lowered to 3 GiB by a later entry below). Preserve omitted/numeric retention settings even when the template explicitly selects dense retention. Image IDs, projector/hash and node details remain installation-specific placeholders; existing operator TOMLs require explicit migration and running supervisors require restart.

### Added

- Optional `runtime.vision` startup key (false when absent, explicit in the template). `true` drops `--language-model-only` from both ranks so the vision tower loads, and adds `--limit-mm-per-prompt '{"video": 0}'`: **video input stays disabled**, because startup profiling would otherwise encode a 30,000-token video. Image input remains unqualified and the shipped profile stays text/tools only.

- Real-input 256K capacity and three-position retrieval checks on the existing combined image, with 3 GiB KV per rank and the existing 4 GiB memory reserve. Preserve the earlier 200K speed, tool-eval and FreedomBench measurements under their original profile.

- ZCode permission-mode facts from a static read of the Desktop-bundled runtime (mode enum and normalizers, unimplemented `auto`, per-tool risk descriptors, decision order, PreToolUse hook merge) and an existing-file guard hook (`examples/zcode-hooks/`) that runs `yolo` but asks before changing existing files, the client configuration directory or running destructive shell commands; recorded a headless verification of the Write and Bash cases and noted that the bundled CLI reaches the local model in headless mode. Document how `limit.context`/`limit.output` feed the context window, `max_tokens` and the auto-compaction budget (effective window minus min(output, 21000), threshold minus 13000), the resulting ceiling on `limit.output` under vLLM, and the stream idle timeout that aborts long prefills. Add the operator note that `/effort max` thinks without a practical bound and `high` or lower is the everyday setting. Add the optional `api.prompt_tokens_details` startup key (`--enable-prompt-tokens-details`) and document why a harness cache display reads zero under the LPA-first profile, with the per-request `glm53_lpa_mode = "off"` opt-out as configured in ZCode. Record the measured block-aligned reuse (4,608-token blocks; 9,216 cached of a repeated 16,859-token prompt). Lower the distributed `resources.reserve_gib` from 4 to 3 after two supervised memory stops at the 256K profile. Ship `lpa.enabled = false` in the template and document LPA as a batch opt-in that forfeits shared prefix reuse, keeping 256K as the distributed default. Add `tools/check_prefix_cache.py`, which sends one long prompt twice and separates "the server never reports cached tokens" from "the request was approximated", and a setup guide for the ZCode guard hook.

- Bilingual performance/capacity Q&A covering 256K KV fit, prospective 1M memory budgets, fixed MTP weights, long prefill waits and concurrent-serving/EP/PP tradeoffs. Link measured evidence and distinguish the illustrative 40-minute 1M extrapolation from actual measurements.

- Release-candidate measurements on the canonical-order image: twelve valid sparkDash streams, all 69 tool-eval scenarios (90/100, with TC-43 failing its Safety Gate), all 60 original-English FreedomBench questions correct, and completed 200K capacity/three-position retrieval checks. Preserve invalid transport attempts separately and retain business/harness qualification limits.

- A bilingual document map (`docs/README.md`) owning document roles, language availability and the single source of truth for each fact, and a bilingual inference optimization overview (`docs/optimization-overview.md`) showing where each measure acts, the serial combination results and workload-specific profiles. The README gains prefix-caching/retention status rows and a description of the external Euryale research project; the catalog document-role table was replaced by a pointer to the map. Japanese counterparts were added for operations, architecture, validation, component validation, performance investigation, indexer reuse and contributing, and an English counterpart for the APC-first LPA design, so every user-facing document is now an English/Japanese pair.

- Canonical logical candidate ordering at the shared GLM sparse-MLA runtime boundary, enabled in newly built reference images. Preserve selection/padding and normalize before physical cache mapping; document automatic source-pinned installation, explicit comparison override and limited GB10 ordinary/speculative regression evidence.

- Model-origin Bearer clients, frozen allocator overrides preserving unset/empty, structured all-rail RoCE checks, and an experimental two-rank pre-stop asset/switch/recovery path. CPU contracts and real-host qualification are recorded separately in the launch-safety guide.

- APC-first, uncached-suffix LPA with scheduler-owned boundaries, exact-only shared cache, explicit exact priming, source-pinned client validation, GPU cache-isolation fixtures and crossover measurement tools.

- Deployment entry points distinguishing the NVIDIA checkpoint, GB10-compatible hardware and MSI EdgeXpert measurements, with canonical cache, MTP-view and LPA-projector storage guidance.

- Independent checked-asynchronous-index results and an explicit auto/sync/async startup selection; checks remain mandatory and combined-mode acceptance stays separate.

- Full-model TP/PP timing and restoration results, preserving faster PP prefill, slower generation, limited memory headroom and an incomplete profiler/capacity result after a profiler restart failure.

- Full-model EP off/on/off results: verified placement and scoped quality/cancellation/capacity, with a decision against EP for the measured throughput workload. Internal-abort telemetry is distinguished from normal completion counters.

- Explicit experimental TP2/PP2 selection and a configurable stage boundary, with hybrid-layout and combination guards.
- An exclusive diagnostic RPC to compare synchronous and asynchronous checked indices independently of Graph capture; no range checks are disabled.

- An eight-layer, byte-verified fixture option for valid hybrid-cache PP stage boundaries, plus attention/KDA-state hashes and actual expert-placement observation.

- Experimental deferred-mHC pipeline buffers and source-pinned common-cache-layout selection, retaining unsuccessful PP startup evidence and pending GPU qualification.
- Matched task-order measurements with a scoped decision against dedicated task grouping for the tested workload.

- Independent prefill chunk sweep with throughput/maximum-ITL tradeoffs and separate 16K x two capacity evidence.
- FP32 NoPE fusion and bounded tile/query-chunk diagnostics, retaining negative speed results despite fewer launches and intermediates.

- An explicit default-off Expert Parallel implementation and independent validation plan, separating EP adoption from qualified two-sequence batching and distribution-only startup acceptance.

- Independent batching A/B/A measurements and a separate 16K × two-request capacity check, with explicit fixed-KV/RAM conditions.
- A padded native-attention component probe retaining failed numerical and unsupported-shape evidence before any serving change.

- Bilingual enterprise objectives and a performance/quality initiative catalog, with stable comparison IDs, evidence links, deferred-candidate criteria and next-GLM evaluation fields. Political-bias evaluation is distinguished from a claim of universal neutrality.

- Independently tested CUDA unpack fusion and CSA2 research components: bounded indexer capture, request-local candidate reuse, candidate selection/tail expansion, and native-backed shared-pool scoring. Component benchmarks distinguish numerical correctness from actual speed gains.
- An explicit component-validation startup profile and full-target CUDA A/B/A/indexer-overlap driver, separate from LPA/MTP integration.

- FreedomBench collection and preliminary original-English/long-context observations, separating political refusals, execution/format errors and source fidelity from the still-unqualified full matrix and human review.

- Categorized TOML configuration shared by an explicit experimental TP=2 launcher and serial chat client, with immutable artifact checks, bounded supervision and MTP-aware LPA opt-in.

- Opt-in late-prefill attention-input approximation, teacher replay checks, bounded LLM-jp corpus sampling and diagonal/low-rank projector fitting, with bilingual experimental-scope guidance.

- MTP k=3 example and matched comparison against k=1, including per-position acceptance and workload-dependent latency/throughput results.
- Opt-in MTP k=1 metadata-view preparation, BF16 draft configuration, CPU rejection checks and bilingual measured comparison/rollback guidance.
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

- Separate per-initiative functional acceptance, performance adoption and defaults; defer final combination qualification until candidates are selected.
- Make experimental non-eager startup explicitly select uncompiled prefill and single-sequence decode Graphs, with image checks and deferred-combination guards. Full-model qualification remains pending.

- Make LPA canonical across modules, worker class, RPCs, mount path, tests and documentation; require the current image contract without legacy interfaces.
- Validate startup profiles at load/command boundaries, removing repeated full-schema validation during serve-argument assembly.

- Allow `resources.run_seconds = 0` to disable the run deadline while retaining low-memory protection; this does not add automatic restart or production availability qualification.

- Add prominent modification notices to generated vLLM patch outputs without changing their executable syntax tree.
- Grouped operator commands, validation tools, runtime adaptations, configuration and Docker assets by responsibility.
- Centralized the model ID and revision in `config/runtime.lock.json`.
- Serialized downloads with a process lock and atomic status updates; paused downloads terminate verification waits.
- Removed local experiment links and host-specific image identities from public entry points.

### Validation status

- MTP k=3 passed all 21 measured requests and 11 basic API checks, improving short/medium-input decode over k=1 with unchanged reported model memory. It is preferred for further experiments; k=2/k≥4 and actual harnesses remain untested.
- Full-model TP=2 MTP k=1 completed the same 21 measured requests and 11 basic API checks. Faster decode costs about 7 GiB per rank and does not improve every workload; actual harnesses and distributed recovery remain unqualified.
- Full 45-layer TP=2 loading, final-answer/tool API smoke and 21 measured benchmark requests completed under the documented serial reference profile. Harnesses and routine deployment remain unqualified.
- Thinking-off is not used for the fixed GLM template. Final-answer/tool acceptance is distinct from exact reasoning-text and cross-batch numerical diagnostics; raw mismatches are preserved.
- Two-host RoCE collective patterns passed; the final 256 MiB AllReduce measurement was 1.18–1.21 GB/s at MTU 1500, with a separate disk-checksum job active. This does not qualify full-model TP=2.
- A single-GB10, four-layer fixture passed the tested generation/state-consistency cases using Marlin W4A16 and the candidate-preserving NoPE reference path.
- Rebuilt the reorganized Docker image and repeated the GPU component and long-input fixture checks through the packaged CLI successfully.
- CUTLASS W4A4 completed generation but differed across tested execution geometries.
- Batch invariance is unavailable for the pinned SM120 sparse-MLA backend.
- **Broader model quality, production reliability and performance beyond the documented experimental profiles remain unvalidated.**
