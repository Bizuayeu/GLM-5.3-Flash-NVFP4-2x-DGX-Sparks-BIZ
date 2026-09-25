# FreedomBench and political-context evaluation

[日本語](freedombench.ja.md) · [Validation](validation.md)

## Closure on the serving profile (2026-09-22)

**Closed on 2026-09-22 on the profile the reference pair serves** (as recorded that day: the published option's route l weights with the split KDA projection, one active sequence, 3 GiB of KV per rank, MTP k=3, FA2 prefill, `runtime.prefix_page_dedup`, image `76a1172b…`, fingerprint `945965bf…`). Between 23:58 and 23:59 Asia/Tokyo the repository runner ran all 60 pinned original-English questions: **60 correct of 60 planned, every question on the first attempt, zero upstream `refused`, zero errors**, with the upstream 8,192-token output budget and the pinned classifier (record `records/20260922-freedombench/serving-full`). The long-prefix pilot then prepended the same 6,000-character Japanese validation-text excerpt as on 2026-09-13 to the first six questions (inputs 4,810–4,838 tokens, so the whole prefix sits before the question) and answered **6 of 6** through the ordinary chat endpoint (record `long-pilot-serving`).

What closes with it. LPA is off in both example profiles, so FB-05 (approximation exercised) has nothing to exercise until an LPA profile is served again. The human refusal review has an empty set to review: no answer was refused or unparseable. The source audit is the pinned revision and hashes in `config/freedombench.lock.json`. What does not close: the suite was not translated into Japanese, opposed political framings were not written, and evidence placement beyond the prefix position was not tested.

## Earlier runs

Each earlier run keeps its own image and profile. On the original questions (inputs 158–209 tokens) every input fit inside the 512-token exact LPA tail, so approximation never ran there.

| Date | Run | Result | Image / profile | LPA approximation |
|---|---|---|---|---|
| 2026-09-12 | `freedombench-combined-v12-full` | 60/60 on the first attempt; zero incorrect, `refused` and errors | `32394330…`; TP=2, one sequence, MTP k=3, fused unpack, LPA configured on with tail 512, Graphs off | Not executed |
| 2026-09-13 | `freedombench-integration-v36`, `freedombench-long-pilot-v37` | 60/60 on the first attempt, no transport errors, truncation or unparsed choices; pilot 6/6 in all three arms | [Serial integration profile](benchmarks.md#serial-integration-of-mtp-lpa-fused-unpack-and-async-checks-p18) (MTP3, fused unpack, checked asynchronous index validation) | Pilot only |
| 2026-09-14 | `release-200k-reserve4` | 60/60 on the first attempt; zero refusal labels, incorrect answers and errors; every response `finish_reason=stop`, outputs 6–13 tokens | [Recorded combined profile](benchmarks.md#release-candidate-measurements) | Not executed |

### Integration recheck and long-prefix pilot

The pilot prepended a fixed 6,000-character LLM-jp validation-text excerpt to the first six pinned questions and ran each LPA off/on/restored with MTP3/fusion/async fixed. Inputs were 4,810–4,838 tokens; both ranks reported 4,298–4,326 skipped historical queries at each of layers 35/39/43 in every LPA-on request, and none in either off arm. This is a small, modified-prompt pilot, **not an official full-suite score or completion of FB-05**: it does not cover every topic, Japanese questions, opposed political framing or long-range evidence placement. No projector was trained or selected on these questions.

### Release candidate retest

The retest ran the full original suite on one combined candidate at temperature 0, effort low, `clear_thinking=true` and the upstream 8,192-token output budget.

## Scope and fixed source

Use [FreedomBench](https://github.com/Lore-Hex/FreedomBench/tree/cc037ac7b286ba4f910309162367d856cbd25d58), revision `cc037ac7b286ba4f910309162367d856cbd25d58` (package metadata: `1.0.2`). Its [question set](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/questions.py) contains 60 English multiple-choice items in 12 China-related topics. It tests agreement with the authors' selected answer key on those items. It does not establish general political neutrality, explain a model's training data, or by itself measure contamination of business context.

Review each item's wording, cited source and date before evaluation. Record disputed/ambiguous items separately without silently changing the official question set or answer key. A wrong answer alone does not establish state-aligned framing or a cause of censorship. Published hosted-model scores cannot be assigned to this local NVIDIA checkpoint; model, quantization, template and serving provider can differ.

The [license](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/LICENSE) is Apache-2.0. The local adapter adapts choice extraction and prompt/option construction, with attribution and modifications recorded in NOTICE. Question data is acquired separately and verified against [the benchmark lock](../config/freedombench.lock.json).

## Required cases

| ID | Test | Evidence / completion condition |
|---|---|---|
| FB-01 | Reproducible local route | Fixed benchmark/source hashes, actual model/image/config fingerprint, tokenizer/template and endpoint recorded. Only the selected local model receives evaluation requests; no cloud fallback. |
| FB-02 | Original English MC suite | All 60 unique IDs, unchanged prompts/answer-key mapping and deterministic option shuffle, evaluated on each served profile. Preserve every attempt and profile-specific result. |
| FB-03 | Refusal and scoring audit | Report upstream-compatible scores plus separate transport errors, truncation, empty final content, malformed choices, explicit refusals and wrong choices. Inspect anomalies against original responses. |
| FB-04 | Japanese business use | Human-reviewed Japanese translations preserve meaning, IDs, options and answer mapping. Keep a translation hash and report separately from the original English benchmark. |
| FB-05 | Long business context and source fidelity | A preregistered set covering every topic tests factual extraction/summary from supplied documents with neutral versus politically framed background. Add matched benign control topics. Place decisive evidence early/middle/late; log actual prompt length and active LPA counters. Assess unsupported political insertions, omitted relevant evidence, refusal, attribution and distinction between a source's claim and verified fact. Publish as a separate local extension, not an official FreedomBench score. |

When an LPA profile is served again, rerun FB-02 on it and exercise FB-05 with A/B/A against the matching no-LPA profile: the same compatible immutable image, target weights, prompts, sampling/template settings, context/cache limits and one active sequence, the existing selected projector and boundaries, and the MTP metadata view recorded where needed. Short original questions may fit entirely within the exact LPA tail; report bypassed cases as such, because they cannot prove LPA quality. FB-05 must demonstrably exercise approximation, with document content identical across paired profiles and total input/output within the configured context. Never train or select the projector on these cases, and never mix prefixed or translated runs into the original score.

## Runner and scoring requirements

On the Linux model host, use `python -m glm53_setup freedombench --benchmark-dir <pinned-source-directory> --output records/<new-run> --config state/server.toml`. The local adapter parses literal questions without executing upstream Python, uses the selected local client and holds the single-controller lock. `--limit` produces a labeled pilot, not a full-suite result. Mark each extension or profile without its own evidence NOT RUN.

The pinned [runner](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/run.py) uses TrustedRouter by default, fetches a provider catalog unless models are explicit, and retries responses without an extractable choice up to four additional times. Its defaults include concurrency 8 and an 8,192-token output budget. **Do not run the upstream defaults against this deployment.** Use the local-only adapter around the existing serial client; verify its parser/prompt compatibility offline before GPU execution. A URL override alone is not proof that SDK catalog/failover traffic stays local.

Keep the original system prompt and option renderer for FB-02. Record the necessary GLM template/reasoning settings as a local-run difference. Do not import the convenience client's 512-token output cap without checking reasoning completion. Start from the upstream output budget only when prompt plus output fits the server context; validate timeout against local speed. Persist all attempts, `finish_reason`, usage, final content and separately returned reasoning. Any retry policy/budget change requires a separately labeled condition and must not erase first-attempt failures.

The pinned [scorer](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/classify.py) uses marker/regex/fallback extraction; unparseable text becomes `refused`, while rows with an error are excluded from its percentage denominator. Retain that score for traceability, but also report correct/planned, completed/planned and error counts. Require full unique-ID coverage, detect duplicates, and never describe an incomplete run's percentage as a full-suite pass. Check missing final answers and length termination before attributing refusal to politics. Keep human-reviewed explicit refusals and source/claim errors separate from the upstream label.

## Acceptance and artifacts

Save a manifest, per-attempt requests/responses, raw upstream-compatible summary, topic/language/profile breakdowns, paired differences, parser audits and source-review notes under private `records/<run-id>/`. Record missing/unsupported cases as NOT RUN or BLOCKED. No external judge service is required; any supplementary human review uses a written rubric and preserves disagreement.

Report measurement completion separately from suitability for a particular organization. Set the intended use and acceptance criteria before seeing scores; this specification invents no universal pass percentage. Newly reproducible refusals, factual regressions or unsupported contextual assertions in an optimized profile must be investigated before promoting that profile. A high FreedomBench score does not close the harness, reliability or business-quality gates.
