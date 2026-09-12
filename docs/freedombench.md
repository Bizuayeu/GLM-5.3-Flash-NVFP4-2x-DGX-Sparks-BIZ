# FreedomBench and political-context evaluation

[日本語](freedombench.ja.md) · [Validation](validation.md)

**Required enterprise evaluation item — NOT RUN.** This specification defines the planned evaluation; results are reported only after a complete measured run.

## Scope and fixed source

Use [FreedomBench](https://github.com/Lore-Hex/FreedomBench/tree/cc037ac7b286ba4f910309162367d856cbd25d58), revision `cc037ac7b286ba4f910309162367d856cbd25d58` (package metadata: `1.0.2`). Its [question set](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/questions.py) contains 60 English multiple-choice items in 12 China-related topics. It tests agreement with the authors' selected answer key on those items. It does not establish general political neutrality, explain a model's training data, or by itself measure contamination of business context.

Review each item's wording, cited source and date before evaluation. Record disputed/ambiguous items separately without silently changing the official question set or answer key. A wrong answer alone does not establish state-aligned framing or a cause of censorship. Published hosted-model scores cannot be assigned to this local NVIDIA checkpoint; model, quantization, template and serving provider can differ.

The [license](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/LICENSE) is Apache-2.0. No upstream implementation/data is vendored by this specification. Any later adaptation must retain applicable attribution/license notices, mark modifications and record dependencies.

## Required cases

| ID | Test | Evidence / completion condition |
|---|---|---|
| FB-01 | Reproducible local route | Fixed benchmark/source hashes, actual model/image/config fingerprint, tokenizer/template and endpoint recorded. Only the selected local model receives evaluation requests; no cloud fallback. |
| FB-02 | Original English MC suite | All 60 unique IDs, unchanged prompts/answer-key mapping and deterministic option shuffle, independently evaluated under the four profiles below. Preserve every attempt and profile-specific result. |
| FB-03 | Refusal and scoring audit | Report upstream-compatible scores plus separate transport errors, truncation, empty final content, malformed choices, explicit refusals and wrong choices. Inspect anomalies against original responses. |
| FB-04 | Japanese enterprise use | Human-reviewed Japanese translations preserve meaning, IDs, options and answer mapping. Keep a translation hash and report separately from the original English benchmark. |
| FB-05 | Long business context and source fidelity | A preregistered set covering every topic tests factual extraction/summary from supplied documents with neutral versus politically framed background. Add matched benign control topics. Place decisive evidence early/middle/late; log actual prompt length and active LPA counters. Assess unsupported political insertions, omitted relevant evidence, refusal, attribution and distinction between a source's claim and verified fact. Publish as a separate local extension, not an official FreedomBench score. |

Profile matrix: (A) LPA off / MTP off, (B) LPA off / MTP k=3, (C) LPA on / MTP off, (D) LPA on / MTP k=3. Use the same compatible immutable image, target weights, prompts, sampling/template settings, context/cache limits and one active sequence; record the MTP metadata view where needed. Use the existing selected LPA projector and boundaries. Restore the corresponding no-LPA profile after a candidate run to investigate changes using A/B/A. Do not use these cases to train or select the projector.

Short original questions may fit entirely within the exact LPA tail. Report bypassed cases as such; they cannot prove LPA quality. FB-05 must demonstrably exercise approximation, while keeping document content identical across paired profiles and total input/output within the configured context. Adding prefixes or translations changes the benchmark condition and must never be mixed into the original score.

## Runner and scoring requirements

The pinned [runner](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/run.py) uses TrustedRouter by default, fetches a provider catalog unless models are explicit, and retries responses without an extractable choice up to four additional times. Its defaults include concurrency 8 and an 8,192-token output budget. **Do not run the upstream defaults against this deployment.** Implement a local-only adapter around the existing serial client; verify its parser/prompt compatibility offline before GPU execution. A URL override alone is not proof that SDK catalog/failover traffic stays local.

Keep the original system prompt and option renderer for FB-02. Record the necessary GLM template/reasoning settings as a local-run difference. Do not import the convenience client's 512-token output cap without checking reasoning completion. Start from the upstream output budget only when prompt plus output fits the server context; validate timeout against local speed. Persist all attempts, `finish_reason`, usage, final content and separately returned reasoning. Any retry policy/budget change requires a separately labeled condition and must not erase first-attempt failures.

The pinned [scorer](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/classify.py) uses marker/regex/fallback extraction; unparseable text becomes `refused`, while rows with an error are excluded from its percentage denominator. Retain that score for traceability, but also report correct/planned, completed/planned and error counts. Require full unique-ID coverage, detect duplicates, and never describe an incomplete run's percentage as a full-suite pass. Check missing final answers and length termination before attributing refusal to politics. Keep human-reviewed explicit refusals and source/claim errors separate from the upstream label.

## Acceptance and artifacts

Save a manifest, per-attempt requests/responses, raw upstream-compatible summary, topic/language/profile breakdowns, paired differences, parser audits and source-review notes under private `records/<run-id>/`. Record missing/unsupported cases as NOT RUN or BLOCKED. No external judge service is required; any supplementary human review uses a written rubric and preserves disagreement.

Report measurement completion separately from suitability for a particular organization. Set the intended use and acceptance criteria before seeing scores; this specification invents no universal pass percentage. Newly reproducible refusals, factual regressions or unsupported contextual assertions in an optimized profile must be investigated before promoting that profile. A high FreedomBench score does not close the harness, reliability or enterprise-quality gates.
