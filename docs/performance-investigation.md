# Performance investigation

These are measurement hypotheses, not diagnosed bottlenecks or promised speedups. Keep prefill latency, per-request decode latency and aggregate throughput separate. Nominal LPDDR bandwidth alone does not establish the cost of quantized MoE routing, sparse attention, scheduling or communication.

## Kernel launches and synchronization

Set `profiling.enabled=true` in a dedicated startup profile. The launcher enables the pinned vLLM Torch profiler and mounts a fresh per-container output directory under `records/profiles/`. Invoke the server's on-demand profiler around one warmed, exclusive request and collect both ranks' traces. Record the exact prompt and number of completed output tokens; keep cold JIT, prefill and speculative decode separate.

`python -m glm53_setup profile-assess <trace.json.gz>` counts GPU kernel events, CUDA launch API events and NCCL kernel events separately. Duration sums may overlap and are not end-to-end latency. Compare a one-output-token prefill control against the same prompt with longer fixed output to estimate additional launches per generated token. With MTP, also record acceptance and engine step counts; draft tokens are not accepted output tokens. Measure final latency with tracing disabled.

Use traces to prioritize kernel work. CUDA events in LPA's layer profiler do not count launches. The reference attention uses Python-driven query chunks, and LPA requires eager mode; CUDA graphs or a fused NoPE kernel are separate correctness projects requiring full-candidate, cache/state and quality checks.

## Throughput versus determinism

Use a separate no-LPA profile for `context.max_num_seqs > 1`; the LPA hook requires one contiguous sequence. Start with two and then four active sequences only if resource and task checks pass. A near-tie greedy divergence is a numerical/reproducibility diagnostic, not automatically a task-quality failure. Preserve it while judging content, tool arguments, finish reasons, cross-request isolation, cancellation and resource safety independently. Report aggregate throughput and each request's latency/quality; do not compare aggregate rates to single-request decode.

## TP versus PP

TP=2 versus PP=2 is relevant for two memory-constrained nodes. It is **not currently a supported launch-option toggle** in this pinned GLM. The [fixed model source](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py) states that PP is gated off because `make_empty_intermediate_tensors` is missing. Its partial PP branch also omits deferred mHC `post`/`comb` state. LPA expects all language layers in one local module tree, adding a cross-stage representation issue.

Before adding a PP profile, implement/validate intermediate-state transport on a small fixture with LPA/MTP disabled. Check layer allocation, KDA/MLA state, mHC boundaries, rank memory and failure recovery. Only then compare the full model against TP with identical precision, prompts and memory budgets. MTP/LPA are later PP extensions. Count actual communication events rather than assuming a layers-times-two collective count.

No PP result, graph speedup or fused-kernel qualification is claimed by this plan.
