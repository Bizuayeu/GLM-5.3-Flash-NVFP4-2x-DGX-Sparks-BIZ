# HLE (Humanity's Last Exam)

[日本語](hle.ja.md) · [Validation](validation.md)

This page records how the served model answered two 100-question subsets of [HLE](https://huggingface.co/datasets/cais/hle), on both profiles, under a bounded budget on this hardware. **Its numbers are not comparable with published HLE values** and are not to be placed beside them:

- **Budget.** Each answer had at most 16,384 new tokens. The pinned model card reports its benchmarks (GPQA Diamond among them) with up to 327,680. Answers cut off by the budget (`finish_reason` `length`) are counted as missing, not wrong, so a correct rate here covers only the questions the model finished.
- **Sampling.** Temperature 1.0 and top_p 0.95, as the pinned model card and `generation_config.json` give, with reasoning effort low (the profile's default).
- **Judging.** Answers were graded off the model host: an exact match after normalization settles an answer, and every other extracted answer went to two judges, Claude Opus 5.5 (effort low) and Claude Sonnet 5, through the Claude Code CLI. This is not HLE's official judge; both judges' counts are reported.
- **Questions.** Not the full set. The text set is 100 exact-answer questions without images, sampled in proportion to subject for a separate evaluation; the image set is 100 exact-answer questions with images, drawn the same way. Both are pinned by hash.

Questions, answers and grades are private records marked `teacher_excluded`. The dataset asks that its data never appear in training corpora, so no public document quotes a question, an answer or a question ID.

## Results

| Set | Profile | Answered (stopped with an extractable answer) | Correct among the stopped answers: Opus 5.5 low / Sonnet 5 |
|---|---|---|---|
| Text | Distribution defaults (2026-09-28 to 09-30) | **61 of 100** | **9 / 10 of 62** (4 by exact match; the judges disagree on 1) |
| Text | Published option | <!-- PENDING: HLE text × AXL --> | |
| Image | Published option | <!-- PENDING: HLE image × AXL --> | |
| Image | Distribution defaults | <!-- PENDING: HLE image × default --> | |

Text on the distribution defaults (1.19.0 image `99e6cf7a…`, run `c-default-text`): 62 answers stopped and 38 reached the budget; one stopped answer had no extractable answer line. No question reached the 1,200 s client timeout; the longest took 921 s.

## How it was run

On the Linux model host, with the served profile running:

~~~sh
python -m glm53_setup hle --questions <pinned question file> --config state/server.toml \
  --output records/<new-run> --label <profile> \
  --max-tokens 16384 --temperature 1.0 --top-p 0.95 --timeout 1200 --max-new 1
~~~

The question file is exported off the host and carries no reference answers. The runner asks one question at a time through the serving API, saves each answer on its own and resumes without resending answered questions; `--max-new 1` pauses after each question so a driver can rest the hosts. `--limit` makes a labeled pilot, never a full-set result.

A question the server refuses with an HTTP client error (4xx other than 401 and 403) is saved with `finish_reason` `rejected` and its HTTP status, and the run goes on to the next question; like an answer cut off by the budget, it counts as missing, not wrong. Authentication and server errors, timeouts and lost connections still stop the run, which then resumes from the next unanswered question. The refusal seen on the distribution's profiles is the encoder-cache gap: an image of 7,922 to 8,000 tokens is refused with HTTP 400, because the profile's encoder cache holds 7,921 tokens while the model's processor allows 8,000 ([Vision](vision.md#limits-and-open-items)).

Why this budget: a five-question pilot on the published option at temperature 0 and 8,192 tokens reached the limit on every question without a final answer, and one question given 32,768 spent them all on reasoning. The run therefore took the checkpoint's sampling and 16,384 tokens, which fits the 1,200 s timeout at the slowest decode measured.

Operational notes. The text set on the defaults ran as two overnight batches: switch to the defaults, check decode, answer questions, switch back to the published option and check decode again. Before each question the driver waited until both hosts had cooled (below 60 °C, at most 600 s, from the seventh question on; a stricter band before), and both hosts ran under the [GPU clock cap](operations.md#gpu-clock-cap) of 2,200 MHz.
