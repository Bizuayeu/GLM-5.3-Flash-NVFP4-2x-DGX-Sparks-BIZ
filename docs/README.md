# Document map

[日本語](README.ja.md)

Every document has one role; other documents link to it instead of repeating its content. Every user-facing page comes as an English/Japanese pair (`name.md` / `name.ja.md`); only the changelog, the agent instructions and the license/notice texts are English-only by design. Update both when user-visible instructions change ([Contributing](../CONTRIBUTING.md)).

## Entry points

| Document | Role | EN | JA |
|---|---|---|---|
| README | What is deployed, supported hardware, verified scope, business-use objectives (BIZ) | [EN](../README.md) | [JA](../README.ja.md) |
| Setup runbook | Ordered deployment gates from host inspection to acceptance | [EN](../SETUP.md) | [JA](../SETUP.ja.md) |
| Contributing | CPU checks, publication audit, contribution rules | [EN](../CONTRIBUTING.md) | [JA](../CONTRIBUTING.ja.md) |
| Changelog | Change history and validation status by release | [EN](../CHANGELOG.md) | — |
| Repository instructions | Rules for AI agents and operators editing this checkout | [EN](../AGENTS.md) | — |
| Licensing and notices | Apache-2.0 text, attribution, third-party provenance | [LICENSE](../LICENSE), [NOTICE](../NOTICE), [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md), [LICENSES/](../LICENSES/) | — |

## Deploy and operate

| Document | Role | EN | JA |
|---|---|---|---|
| Operations | Artifact storage paths, acquisition, host preparation, launch checks, recovery | [EN](operations.md) | [JA](operations.ja.md) |
| Server configuration | The categorized server TOML, KV/RAM conditions, image contract, LPA/MTP constraints | [EN](server-configuration.md) | [JA](server-configuration.ja.md) |
| QSFP network | Direct QSFP connection and persistent NetworkManager profiles | [EN](qsfp-network.md) | [JA](qsfp-network.ja.md) |
| NCCL validation | Two-host collective diagnostic and its limits | [EN](nccl-validation.md) | [JA](nccl-validation.ja.md) |
| Launch contracts | API client authentication, allocator propagation, all-rail checks, two-rank switch and recovery, APC history qualification | [EN](launch-safety.md) | [JA](launch-safety.ja.md) |
| Architecture | Package layout and validation boundaries | [EN](architecture.md) | [JA](architecture.ja.md) |

## Validate and accept

| Document | Role | EN | JA |
|---|---|---|---|
| Validation | Evidence versus production qualification, single-GPU fixture, remaining gates | [EN](validation.md) | [JA](validation.ja.md) |
| Component validation | CUDA/indexer components, Graph fixture, PP/EP/APC fixtures, history fixtures | [EN](component-validation.md) | [JA](component-validation.ja.md) |
| Benchmarks | TP=2 benchmark method and the full-model benchmark runs (P08, P11, P13–P15, P17–P19, P21, P22, retention) | [EN](benchmarks.md) | [JA](benchmarks.ja.md) |
| Image input | Vision at 256K: settings, how they were chosen, measurements, limits | [EN](vision.md) | [JA](vision.ja.md) |
| FreedomBench | Political-context evaluation: required matrix and preliminary results | [EN](freedombench.md) | [JA](freedombench.ja.md) |
| Harnesses | ZCode and Claude Code connection plans and the acceptance matrix | [EN](harnesses.md) | [JA](harnesses.ja.md) |
| Licensing guide | Commercial use, modification and redistribution by artifact | [EN](licensing.md) | [JA](licensing.ja.md) |

## Optimize

| Document | Role | EN | JA |
|---|---|---|---|
| Optimization overview | Where each measure acts, adopted stack, profiles by workload | [EN](optimization-overview.md) | [JA](optimization-overview.ja.md) |
| Optimization catalog | Initiative IDs P01–P24 / E01–E03, decisions, reevaluation criteria, comparison record fields | [EN](optimization-catalog.md) | [JA](optimization-catalog.ja.md) |
| Performance investigation | Measurement procedures: launches and synchronization, task grouping, EP, TP versus PP | [EN](performance-investigation.md) | [JA](performance-investigation.ja.md) |
| Speculative decoding | MTP metadata view, k=1 and k=3 measurements | [EN](speculative-decoding.md) | [JA](speculative-decoding.ja.md) |
| LPA | Late-prefill approximation: projector download/model card, enabling it in the server profile, mechanism, scope, reproduction, evidence | [EN](lpa.md) | [JA](lpa.ja.md) |
| APC-first LPA design | Shared-cache contract for combining prefix caching with LPA (P22) | [EN](apc-lpa-design.md) | [JA](apc-lpa-design.ja.md) |
| Candidate order | Canonical sparse-candidate ordering in the reference image | [EN](candidate-order.md) | [JA](candidate-order.ja.md) |
| Indexer reuse | CSA2 candidate reuse and restricted rescoring components | [EN](indexer-reuse.md) | [JA](indexer-reuse.ja.md) |

## Single sources of truth

| Fact | Owner |
|---|---|
| Model ID, pinned revision, base-image digest, local reference tag, pinned vLLM source commit | [config/runtime.lock.json](../config/runtime.lock.json) |
| Distributed LPA projector URL, file hash, format, teacher and training provenance | [config/lpa-projector.lock.json](../config/lpa-projector.lock.json); package layout in [operations](operations.md#artifact-storage-and-paths) |
| Server profile schema and every profile key | [examples/server.example.toml](../examples/server.example.toml), explained in [server configuration](server-configuration.md) |
| MTP speculative configuration examples | [examples/speculative.mtp1.json](../examples/speculative.mtp1.json), [speculative.mtp3.json](../examples/speculative.mtp3.json) |
| FreedomBench item and answer pins | [config/freedombench.lock.json](../config/freedombench.lock.json) |
| Initiative IDs, adoption decisions, reevaluation criteria | [Optimization catalog](optimization-catalog.md) |
| Measured numbers and their conditions | [Benchmarks](benchmarks.md), [image input](vision.md), [speculative decoding](speculative-decoding.md), [LPA](lpa.md), [component validation](component-validation.md), [candidate order](candidate-order.md), [indexer reuse](indexer-reuse.md), [NCCL validation](nccl-validation.md), [FreedomBench](freedombench.md) |
| APC/LPA shared-state contract (N, H, T, R, B) | [APC-first LPA design](apc-lpa-design.md) |
| Client authentication, allocator, rails, switch and recovery contracts | [Launch contracts](launch-safety.md) |
| Storage paths for checkpoint, MTP view, projector, images, state | [Operations](operations.md#artifact-storage-and-paths) |
| What `server preflight` checks before a start, and what it does not certify | [Operations](operations.md#full-model-launch-checks) |
| Host kernel requirement, the `7.0.0-1019-nvidia` RoCE failure and the `kho=off` workaround | [Operations](operations.md#host-kernel-and-multi-node-roce) |
| License permissions and obligations by artifact | [Licensing guide](licensing.md), [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) |
| Harness acceptance cases and their run status | [Harnesses](harnesses.md) |
| ZCode permission-mode facts, model limit and compaction budget rules, and the existing-file guard hook | [Harnesses](harnesses.md#zcode-permission-modes-model-limits-and-the-existing-file-guard), script in [examples/zcode-hooks/](../examples/zcode-hooks/) |
| Validation scope and remaining qualification gates | [Validation](validation.md) |
| Raw responses, traces, all repetitions, failures, host-specific values | Private `records/<run-id>/`, never distributed; public documents carry reviewed summaries |
| Private implementation plans and their stage status | `docs/plans/`, untracked and excluded from publication; `docs/plans/IMPLEMENTATION_PLAN.md` indexes them locally |
| Site configuration and acquisition state | `state/`, untracked |

## Conventions

- Change history lives in the [changelog](../CHANGELOG.md) and Git; documents do not accumulate "what changed" notes.
- A measured number appears once, in its owner document, with image, source and workload conditions. Other pages link to it.
- The release version is owned by `pyproject.toml` and each version is described in the [Changelog](../CHANGELOG.md); `python tools/check_publication.py` requires a plain semantic version there and rejects links to `records/`, plan files or paths outside the repository.
- Pushing a `vX.Y.Z` tag publishes the GitHub Release: `.github/workflows/release.yml` takes that version's section from the Changelog (`python tools/release_notes.py X.Y.Z`) and refuses a tag that disagrees with `pyproject.toml` or has no section.
- Related research outside this repository, such as the Euryale draft-proposer project, is described in the README without links, because it is not part of this distribution; other documents mention it only in passing.
