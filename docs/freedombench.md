# FreedomBench and political-context evaluation

[日本語](freedombench.ja.md) · [Validation](validation.md)

## The Japanese translation, FB-04 (2026-09-30)

The suite in Japanese is `ja-1`: all 60 questions translated and reviewed by the maintainer on 2026-09-29, and pinned by [the translation lock](../config/freedombench-ja.lock.json) (the translation's hash and the pinned questions' hash). Each item's English text matches the pinned questions byte for byte, and the A–D order and the letter of the correct answer are the same as in English for all 60. The system prompt and the instruction line are in Japanese; the answer line `ANSWER: X` with a half-width letter, the `A) ` options and the upstream extractor are unchanged, with no normalization added. The translation file stays in private records. It is scored apart from the English suite.

| Profile | Result |
|---|---|
| Published option, served at two sequences (image `99e6cf7a…`; run `axl-ja1`, 2026-09-30) | **60 correct of 60 planned**, every question on the first attempt, zero upstream `refused`, zero errors; no question answered differently from the English suite |
| Distribution defaults (image `99e6cf7a…`; run `default-ja1`, 2026-10-01) | **60 correct of 60 planned**, every question on the first attempt, zero upstream `refused`, zero errors; no question answered differently from the English suite |

## Opposed framings and evidence placement, FB-05 (2026-09-29 and 2026-09-30)

A separate local extension in English, **not an official FreedomBench score**. Each case puts a background memo before a pinned question (inputs 4,772–4,901 tokens). The memo is neutral (arm N), states the correct answer (K), or states a wrong answer the maintainer designated before the run: for political questions an option matching the official Chinese government position (S), for six self-written non-political control questions a plausible error (C). The decisive paragraph sits at the front, middle or end of the memo. Case files and predictions were fixed by hash before any result; all runs used the served published option (image `99e6cf7a…`) at temperature 0, one condition per run, with zero errors, refusals and truncations. LPA is off in every template, so these runs test fidelity to supplied material only; **FB-05's LPA part (approximation exercised, A/B/A) remains NOT RUN**.

The pilot (2026-09-29) took the first question of each of the 12 topics plus the six controls, as type M (multiple choice with the upstream prompt and automatic scoring; every arm at every position) and type E (extract, summarize and check the memo, scored by hand against a rubric fixed beforehand; arm N at three positions, the other arms at the middle). Its two missed type M predictions pointed one way: the six political questions never followed the memo, while the controls did 2, 1 and 0 times by position. Type M then ran on all 60 questions (2026-09-30):

| Arm (cases per position) | Front | Middle | End |
|---|---|---|---|
| N, neutral memo (66: the 60 questions and the 6 controls) | 66 correct | 66 correct | 66 correct |
| K, memo states the correct answer (66) | 66 correct | 66 correct | 66 correct |
| S, memo states the designated official-position answer (26 political questions) | 2 wrong | 2 wrong | 1 wrong |
| C, memo states the designated error (6 controls) | 2 wrong | 1 wrong | 0 wrong |

Every wrong answer was the one the memo stated, and the 144 cases whose input was identical to the pilot's chose the same letters. One of this run's six predictions failed: that the political S arm's error rate would not exceed the controls' at any position. At the end it was 1 of 26 against 0 of 6. Only two political questions moved, one on a date detail (at all three positions) and one on what followed a commitment (front and middle); both were answered correctly with no memo and with a neutral one. We read them as the same pattern as the controls (the pilot's photosynthesis gas and bone count): the model adopts a plausible fine detail the memo states with confidence. This is not a skew toward the political position. With six controls a single question moves their rate by 17 points, so the end-position comparison rests on one question against none.

Type E pilot, 84 cases: the extracted catalogue number and count were right in 84 of 84 (mechanical and hand checks agree); no summary inserted a political claim absent from the memo; every S and C case flagged the memo's error (6 of 6 each), and in the K arm one of 18 called a correct statement conflicting before retracting it in the same paragraph; nothing was refused. In type E the controls that type M had followed were flagged, with the correct value given. A first pass scored all 84 cases; the maintainer reviewed 13, agreed on 11 and gave the other two a PARTIAL (0.5): one summary asserted a claim without attributing it to the memo, and the K-arm false alarm above. **The PARTIAL step was added after seeing results** (the fixed rubric was 0 or 1); the first-pass scores are kept beside it, and neither changes the verdict on the predictions.

## Rerun on both 1.19.0 profiles (2026-09-28)

On 1.19.0 (image `99e6cf7a…`) the same runner ran the 60 pinned original-English questions and the long-prefix pilot on both the distribution defaults and the served published option at two sequences, under the 2,200 MHz GPU clock cap ([benchmarks](benchmarks.md#both-profiles-in-one-window-with-a-gpu-clock-cap-2026-09-28)). **Both answered 60 correct of 60 planned, every question on the first attempt, with zero upstream `refused` and zero errors**; the pilot was 6 of 6 (inputs of 4,810 to 4,838 tokens). The output limit was 8,192. The three items the closure below left open were run afterwards: the [Japanese translation](#the-japanese-translation-fb-04-2026-09-30) and [opposed framings and evidence placement](#opposed-framings-and-evidence-placement-fb-05-2026-09-29-and-2026-09-30).

## Closure on the serving profile (2026-09-22)

**Closed on 2026-09-22 on the profile the reference pair serves** (as recorded that day: the published option's route l weights with the split KDA projection, one active sequence, 3 GiB of KV per rank, MTP k=3, FA2 prefill, `runtime.prefix_page_dedup`, image `76a1172b…`, fingerprint `945965bf…`). Between 23:58 and 23:59 Asia/Tokyo the repository runner ran all 60 pinned original-English questions: **60 correct of 60 planned, every question on the first attempt, zero upstream `refused`, zero errors**, with the upstream 8,192-token output budget and the pinned classifier (record `records/20260922-freedombench/serving-full`). The long-prefix pilot then prepended the same 6,000-character Japanese validation-text excerpt as on 2026-09-13 to the first six questions (inputs 4,810–4,838 tokens, so the whole prefix sits before the question) and answered **6 of 6** through the ordinary chat endpoint (record `long-pilot-serving`).

What closes with it. The human refusal review has an empty set to review: no answer was refused or unparseable. The source audit is the pinned revision and hashes in `config/freedombench.lock.json`. What did not close that day, the Japanese translation, opposed political framings and evidence placement beyond the prefix position, ran later (sections above); FB-05's LPA part stays NOT RUN until an LPA profile is served again.

## Earlier runs

Each earlier run keeps its own image and profile. On the original questions (inputs 158–209 tokens) every input fit inside the 512-token exact LPA tail, so approximation never ran there.

| Date | Run | Result | Image / profile | LPA approximation |
|---|---|---|---|---|
| 2026-09-12 | `freedombench-combined-v12-full` | 60/60 on the first attempt; zero incorrect, `refused` and errors | `32394330…`; TP=2, one sequence, MTP k=3, fused unpack, LPA configured on with tail 512, Graphs off | Not executed |
| 2026-09-13 | `freedombench-integration-v36`, `freedombench-long-pilot-v37` | 60/60 on the first attempt, no transport errors, truncation or unparsed choices; pilot 6/6 in all three arms | [Serial integration profile](benchmarks.md#serial-integration-of-mtp-lpa-fused-unpack-and-async-checks-p18) (MTP3, fused unpack, checked asynchronous index validation) | Pilot only |
| 2026-09-14 | `release-200k-reserve4` | 60/60 on the first attempt; zero refusal labels, incorrect answers and errors; every response `finish_reason=stop`, outputs 6–13 tokens; temperature 0, effort low, `clear_thinking=true`, the upstream 8,192-token output budget | [Recorded combined profile](benchmarks.md#release-candidate-measurements) | Not executed |

### Integration recheck and long-prefix pilot

The pilot prepended a fixed 6,000-character LLM-jp validation-text excerpt to the first six pinned questions and ran each LPA off/on/restored with MTP3/fusion/async fixed. Inputs were 4,810–4,838 tokens; both ranks reported 4,298–4,326 skipped historical queries at each of layers 35/39/43 in every LPA-on request, and none in either off arm. This is a small, modified-prompt pilot, **not an official full-suite score or completion of FB-05**: it does not cover every topic, Japanese questions, opposed political framing or long-range evidence placement. No projector was trained or selected on these questions.

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

On the Linux model host, use `python -m glm53_setup freedombench --benchmark-dir <pinned-source-directory> --output records/<new-run> --config state/server.toml`. The local adapter parses literal questions without executing upstream Python, uses the selected local client and holds the single-controller lock. `--limit` produces a labeled pilot, not a full-suite result. A derived case file (the FB-05 framing and evidence-placement cases) runs with `--cases <file> --cases-sha256 <hash fixed before any result> --condition <type-arm-position-language>`, one condition per run: type M keeps the upstream system prompt, answer format and automatic scoring; type E carries its own prompts, is asked once without no-choice retries and is scored by hand. Such a run is recorded as a pilot marked `teacher_excluded`, never as part of the upstream score. The reviewed Japanese translation (FB-04) runs as a full suite with `--benchmark-dir <pinned-source-directory> --translation <translation file>` and no `--limit`: the runner refuses it unless the file matches `config/freedombench-ja.lock.json` and every item's English text matches the pinned questions, then asks with the translation's system prompt and instruction line, the same A–D order and the upstream `ANSWER: X` scoring. The record is `language` `ja-1`, marked `teacher_excluded`, and scored apart from the original English suite. Mark each extension or profile without its own evidence NOT RUN.

The pinned [runner](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/run.py) uses TrustedRouter by default, fetches a provider catalog unless models are explicit, and retries responses without an extractable choice up to four additional times. Its defaults include concurrency 8 and an 8,192-token output budget. **Do not run the upstream defaults against this deployment.** Use the local-only adapter around the existing serial client; verify its parser/prompt compatibility offline before GPU execution. A URL override alone is not proof that SDK catalog/failover traffic stays local.

Keep the original system prompt and option renderer for FB-02. Record the necessary GLM template/reasoning settings as a local-run difference. Do not import the convenience client's 512-token output cap without checking reasoning completion. Start from the upstream output budget only when prompt plus output fits the server context; validate timeout against local speed. Persist all attempts, `finish_reason`, usage, final content and separately returned reasoning. Any retry policy/budget change requires a separately labeled condition and must not erase first-attempt failures.

The pinned [scorer](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/classify.py) uses marker/regex/fallback extraction; unparseable text becomes `refused`, while rows with an error are excluded from its percentage denominator. Retain that score for traceability, but also report correct/planned, completed/planned and error counts. Require full unique-ID coverage, detect duplicates, and never describe an incomplete run's percentage as a full-suite pass. Check missing final answers and length termination before attributing refusal to politics. Keep human-reviewed explicit refusals and source/claim errors separate from the upstream label.

## Acceptance and artifacts

Save a manifest, per-attempt requests/responses, raw upstream-compatible summary, topic/language/profile breakdowns, paired differences, parser audits and source-review notes under private `records/<run-id>/`. Record missing/unsupported cases as NOT RUN or BLOCKED. No external judge service is required; any supplementary human review uses a written rubric and preserves disagreement.

Report measurement completion separately from suitability for a particular organization. Set the intended use and acceptance criteria before seeing scores; this specification invents no universal pass percentage. Newly reproducible refusals, factual regressions or unsupported contextual assertions in an optimized profile must be investigated before promoting that profile. A high FreedomBench score does not close the harness, reliability or business-quality gates.
