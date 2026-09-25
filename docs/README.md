# Document map

[日本語](README.ja.md)

Every document has one role; other documents link to it instead of repeating its content. Every user-facing page comes as an English/Japanese pair (`name.md` / `name.ja.md`); only the agent instructions, the license/notice texts and the overlay manifest (a provenance record whose hashes have one owner) are English-only by design. The changelog is a pair from 1.6.0 on: the English file is canonical, the GitHub Release is made from it, and earlier versions exist in English only. Update both when user-visible instructions change ([Contributing](../CONTRIBUTING.md)).

## Entry points

| Document | Role | EN | JA |
|---|---|---|---|
| README | Summary, what is deployed, supported hardware, how to start, the headline comparison of the two served profiles, status by scope, business-use objectives (BIZ) | [EN](../README.md) | [JA](../README.ja.md) |
| Setup runbook | Ordered deployment gates from host inspection to acceptance | [EN](../SETUP.md) | [JA](../SETUP.ja.md) |
| Contributing | CPU checks, publication audit, contribution rules | [EN](../CONTRIBUTING.md) | [JA](../CONTRIBUTING.ja.md) |
| Changelog | Change history and validation status by release | [EN](../CHANGELOG.md) | [JA](../CHANGELOG.ja.md) (from 1.6.0) |
| Repository instructions | Rules for AI agents and operators editing this checkout | [EN](../AGENTS.md) | — |
| Licensing and notices | Apache-2.0 text, attribution, third-party provenance | [LICENSE](../LICENSE), [NOTICE](../NOTICE), [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md), [LICENSES/](../LICENSES/) | — |

## Deploy and operate

| Document | Role | EN | JA |
|---|---|---|---|
| Operations | Artifact storage paths, acquisition, host preparation, launch checks, recovery | [EN](operations.md) | [JA](operations.ja.md) |
| Server configuration | The categorized server TOML, the published option against the defaults, every optional key, KV/RAM conditions, image markers, feature limits | [EN](server-configuration.md) | [JA](server-configuration.ja.md) |
| QSFP network | Direct QSFP connection and persistent NetworkManager profiles | [EN](qsfp-network.md) | [JA](qsfp-network.ja.md) |
| NCCL validation | Two-host collective diagnostic and its limits | [EN](nccl-validation.md) | [JA](nccl-validation.ja.md) |
| Launch contracts | API client authentication, allocator propagation, all-rail checks, two-rank switch and recovery, APC history qualification | [EN](launch-safety.md) | [JA](launch-safety.ja.md) |
| Architecture | Package layout (every module of `glm53_setup/` and `tools/` has a row) and validation boundaries | [EN](architecture.md) | [JA](architecture.ja.md) |
| Overlay manifest | The two vLLM source overlays the published option needs: targets, SHA-256, base hashes, markers, placement | [EN](../overlays/README.md) | — |

## Validate and accept

| Document | Role | EN | JA |
|---|---|---|---|
| Validation | Evidence versus production qualification, single-GPU fixture, remaining gates | [EN](validation.md) | [JA](validation.ja.md) |
| Component validation | CUDA/indexer components, Graph fixture, PP/EP/APC fixtures, history fixtures | [EN](component-validation.md) | [JA](component-validation.ja.md) |
| Benchmarks | TP=2 benchmark method, the independent full-model runs of each initiative, and the measurements of each release | [EN](benchmarks.md) | [JA](benchmarks.ja.md) |
| Image input | Vision at 256K: settings, how they were chosen, measurements, limits | [EN](vision.md) | [JA](vision.ja.md) |
| FreedomBench | Political-context evaluation: the closure on the serving profile, earlier runs and what was not run | [EN](freedombench.md) | [JA](freedombench.ja.md) |
| Harnesses | The accepted harness route (npm ZCode CLI), connection settings and the acceptance matrix | [EN](harnesses.md) | [JA](harnesses.ja.md) |
| ZCode guard hook | Setup of the PreToolUse existing-file guard: why a hook, install, verify, limits | [EN](../examples/zcode-hooks/README.md) | [JA](../examples/zcode-hooks/README.ja.md) |
| Licensing guide | Commercial use, modification and redistribution by artifact | [EN](licensing.md) | [JA](licensing.ja.md) |

## Optimize

| Document | Role | EN | JA |
|---|---|---|---|
| Optimization overview | Where each measure acts, adopted stack, profiles by workload | [EN](optimization-overview.md) | [JA](optimization-overview.ja.md) |
| Optimization catalog | Initiative IDs (the P and E series), decisions, reevaluation criteria, comparison record fields | [EN](optimization-catalog.md) | [JA](optimization-catalog.ja.md) |
| Performance investigation | Measurement procedures: launches and synchronization, task grouping, EP, TP versus PP | [EN](performance-investigation.md) | [JA](performance-investigation.ja.md) |
| Speculative decoding | MTP metadata view, depths one to five, why the template keeps k=3 | [EN](speculative-decoding.md) | [JA](speculative-decoding.ja.md) |
| LPA | Late-prefill approximation: projector download/model card, enabling it in the server profile, mechanism, scope, reproduction, evidence | [EN](lpa.md) | [JA](lpa.ja.md) |
| APC-first LPA design | Shared-cache contract for combining prefix caching with LPA (P22) | [EN](apc-lpa-design.md) | [JA](apc-lpa-design.ja.md) |
| Candidate order | Canonical sparse-candidate ordering in the reference image | [EN](candidate-order.md) | [JA](candidate-order.ja.md) |
| Indexer reuse | CSA2 candidate reuse and restricted rescoring components | [EN](indexer-reuse.md) | [JA](indexer-reuse.ja.md) |

## Single sources of truth

| Fact | Owner |
|---|---|
| Model ID, pinned revision, base-image digest, local reference tag, pinned vLLM source commit | [config/runtime.lock.json](../config/runtime.lock.json) |
| Distributed LPA projector URL, file hash, format, teacher and training provenance | [config/lpa-projector.lock.json](../config/lpa-projector.lock.json); package layout in [operations](operations.md#artifact-storage-and-paths) |
| Server profile schema and every profile key | [examples/server.example.toml](../examples/server.example.toml) (the distributed defaults) and [examples/server.axl.example.toml](../examples/server.axl.example.toml) (the published option: repacked weights, two sequences, 6 GiB KV), explained in [server configuration](server-configuration.md) |
| MTP speculative configuration (the launcher builds it from `mtp.*`; the files show it for a manual reproduction) | [examples/speculative.mtp1.json](../examples/speculative.mtp1.json), [speculative.mtp3.json](../examples/speculative.mtp3.json) |
| FreedomBench item and answer pins | [config/freedombench.lock.json](../config/freedombench.lock.json) |
| Initiative IDs, adoption decisions, reevaluation criteria | [Optimization catalog](optimization-catalog.md) |
| Measured numbers and their conditions | [Benchmarks](benchmarks.md), [image input](vision.md), [speculative decoding](speculative-decoding.md), [LPA](lpa.md), [component validation](component-validation.md), [candidate order](candidate-order.md), [indexer reuse](indexer-reuse.md), [NCCL validation](nccl-validation.md), [FreedomBench](freedombench.md) |
| APC/LPA shared-state contract (N, H, T, R, B) | [APC-first LPA design](apc-lpa-design.md) |
| Client authentication, allocator, rails, switch and recovery contracts | [Launch contracts](launch-safety.md) |
| The decode check after a switch: its routine and the two tools | [Launch contracts](launch-safety.md#after-a-switch-the-decode-check); `tools/decode_check.py`, `tools/decode_divergence.py` |
| Routine-use acceptance of the serving profile: its scope and where each item's evidence is recorded | [Setup runbook step 6](../SETUP.md#6-qualify-the-full-model); the README status table and [validation](validation.md#full-model-tp2-experimental-scope) point there |
| Storage paths for checkpoint, MTP view, projector, images, state | [Operations](operations.md#artifact-storage-and-paths) |
| What `server preflight` checks before a start, and what it does not certify | [Operations](operations.md#full-model-launch-checks) |
| Image capability markers: which setting requires each, and from which version images carry it | [Server configuration](server-configuration.md#current-image-contract) |
| Host kernel requirement, the `7.0.0-1019-nvidia` RoCE failure and the `kho=off` workaround | [Operations](operations.md#host-kernel-and-multi-node-roce) |
| License permissions and obligations by artifact | [Licensing guide](licensing.md), [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) |
| Harness acceptance cases and their run status | [Harnesses](harnesses.md) |
| ZCode permission-mode facts, model limit and compaction budget rules, and the existing-file guard hook | [Harnesses](harnesses.md#zcode-permission-modes-model-limits-and-the-existing-file-guard), script in [examples/zcode-hooks/](../examples/zcode-hooks/) |
| Validation scope and remaining qualification gates | [Validation](validation.md) |
| Raw responses, traces, all repetitions, failures, host-specific values | Private `records/<run-id>/`, never distributed; public documents carry reviewed summaries |
| Other public recipes for this model: links, licenses and what was taken from each | [README](../README.md#other-glm-53-flash-recipes-for-dgx-spark-systems); other documents cite by name and pull request, without links or licenses (`tools/check_publication.py` enforces this) |
| Private implementation plans and their stage status | `docs/plans/`, untracked and excluded from publication; `docs/plans/README.md` indexes them locally and each plan's own leading status line owns its state |
| Site configuration and acquisition state | `state/`, untracked |

## Conventions

- Change history lives in the [changelog](../CHANGELOG.md) and Git; documents do not accumulate "what changed" notes.
- A measured number appears once, in its owner document, with image, source and workload conditions. Other pages link to it. The README's headline section is the one permitted copy: `tools/check_publication.py` ties its heading to the newest measured version.
- `tools/check_publication.py` also keeps this map and the architecture page in step with the tree: the README's short-name citation must carry the version in `pyproject.toml`, every `docs/*.md` must be a link target of this map (Japanese pages of the Japanese map), and every module of `glm53_setup/` and `tools/` must be named in [architecture](architecture.md), either by file name or by a pattern such as `benchmark_*.py`.
- Task types are always listed in the order counting / prose / code, and teacher-forced texts in the order Japanese / English / code / mathematics, in every table and sentence that names more than one.
- The release version is owned by `pyproject.toml` and each version is described in the [Changelog](../CHANGELOG.md); `python tools/check_publication.py` requires a plain semantic version there and rejects links to `records/`, plan files or paths outside the repository.
- Pushing a `vX.Y.Z` tag publishes the GitHub Release: `.github/workflows/release.yml` takes that version's section from the Changelog (`python tools/release_notes.py X.Y.Z`) and refuses a tag that disagrees with `pyproject.toml` or has no section.
- The GitHub repository description and topics restate the README summary with the short name and without any measured number or version, because `tools/check_publication.py` cannot see them; when the summary changes, edit them with `gh repo edit`.
- Related research outside this repository, such as the Euryale draft-proposer project, is described in the README without links, because it is not part of this distribution; other documents mention it only in passing.
