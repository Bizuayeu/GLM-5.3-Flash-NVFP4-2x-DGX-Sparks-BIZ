# Changelog

[日本語](CHANGELOG.ja.md) (from 1.6.0; this English file is canonical and the GitHub Release is made from it)

## Unreleased

### Changed

- LPA with MTP accepts depth 1, 2 or 3. The LPA and APC/LPA workers and the `--mtp` option of `lpa-fixture` and `apc-lpa-fixture` now allow depth 2, and profile validation refuses LPA with MTP depth 4 or 5 before launch ("LPA with MTP accepts depth 1, 2 or 3"); until now such a profile launched and every native-LPA request then failed. Depth 2 has not been measured with LPA. `glm53_setup/runtime/lpa.py` and `apc_worker.py` are baked into the image, so depth 2 needs an image built from this version; an older image refuses it on every request ([server configuration](docs/server-configuration.md#feature-combinations-and-limits)).
- `server preflight` requires the image marker `GLM53_FA2_ATTENTION_API=1` when `runtime.fa2_attention` is `true` (row `fa2_attention_support`, recovery targets included). Every reference image since 1.6.0 carries it; the FA2 files are still mounted over the image ([image contract](docs/server-configuration.md#current-image-contract)).
- The fixture runners share one gate. `run_graph_fixture`, `run_indexer_fixture`, `lpa-fixture`, `agreement-fixture` and `run_repeat_trace` now require, as `fixture-run` and `apc-lpa-fixture` did, that the fixture's download status is `complete` and `all_tensor_bytes_verified` is `true` itself; none of the five checked the status before, nor required the flag to be `true` rather than truthy. Every runner refuses with one sentence, "Only a complete, byte-verified test fixture with 4 layers is allowed" (4 or 8 layers for `agreement-fixture`, no layer count for `run_repeat_trace`), which replaces the earlier sentences, including "Only a byte-verified four-layer test fixture is allowed" of `fixture-run` and `apc-lpa-fixture`.
- `lpa-fixture`, `run_repeat_trace` and `freedombench` record their failure: when an exception escapes, `result.json` gets `"status": "failed"` and `error` (the exception's repr) instead of staying at `"loading"` or `"running"`. Keys are only added; successful records and exit codes are unchanged.
- The attention component checks read an error that is not a number as above the BF16 bound: the SM90 cases and tail, the query-chunk check and the native revisit timing check let it through when the output itself was finite. It takes a NaN in the reference; successful records are unchanged.
- A prefix-cache reset the server does not acknowledge fails `apc-history` and `apc-lpa-benchmark` with "Prefix cache reset was not acknowledged" (was "Dedicated server cache reset failed" and "Cache reset did not complete; do not compare these conditions"), still before the case is measured.
- `fixture-build` writes `fixture-status.json` atomically (temporary file, fsync, replace), as `quant-error --write-status` already wrote its status, so an interrupted build cannot leave a truncated gate file; both files now end with a newline, and the JSON is otherwise the same.
- `tools/prepare_mtp_view.py` reads the lock through the package and refuses one whose revision or image is not pinned.
- `server agreement` refuses a native-LPA profile ("agreement requires LPA off or APC-first; native LPA rewrites prefill") before it reads rank 0's state, so it gives that reason even when no head is running.
- CI installs numpy from `requirements/dev.lock.txt` and runs the seven CPU tests it always skipped (the NVFP4 dequantisation and the agreement KL).
- Internal, with launch arguments, environment, fingerprints and successful records unchanged: `server_config` takes the vLLM argument template from `host` and the image capability checks and freeze/thaw from `server`, and no longer imports `host`; site validation and the NCCL environment move to `fabric` and `resume` to `switch`; the rank state path, container name, head origin, prefix-cache reset, metrics read, projector path, optional defaults, MTP draft config and LPA mode key each have one owner; the five retired prototypes (`fused_nope`, `fused_nope_dot`, `indexer_candidates`, `indexer_reindex`, `indexer_shared_pool`) move from `runtime/` to `validation/`; `apc-history`'s pass/fail rules, `lpa-fixture`'s case verdict, the graph-launch count and the BF16 parity bound become module functions with tests; `release_notes --match-project` reads the version through the package; and tests tie the reference Dockerfile's markers and patches, the MTP templates, the overlays' hashes and the switch vocabulary to their owners.

### Removed

- `lpa-fixture --graphs`, with `glm53_setup/runtime/patch_graph_prefill.py` (the LPA prefill under decode Graphs). It could not run: profile validation refuses LPA with decode Graphs, no reference image applies the patch, and `--graphs` checked the patch's record, which no image carries, so it failed before loading. `lpa-fixture` now refuses the option; its engine settings are unchanged and its result keeps `decode_graphs`, always `false`.
- The memory probe's `zero_moe_scratch` RPC. Nothing called it, and on a serving worker it replaced the Marlin MoE call for the life of the process; the repeat-trace fixture keeps its own zeroing diagnostic. `validation.memory_probe` mounts the checkout's probe over the image, so from the next launch the RPC is gone on older images too.
- Internal: `glm53_setup/runtime/patch_pipeline_layout.py` with `pipeline_state.common_layout_names` (the reference image never applied it) and `glm53_setup/validation/benchmark_attention_graph.py` (no caller).

### Documentation

- The two-sequence batch-invariance investigation is closed. On 2026-09-26 the serving pair was tried with the Marlin MoE split pinned for decode-sized calls, the shared-expert and router GEMMs computed per sequence, and FA2 off: the switches engaged, but all eight two-sequence completions still differed from their lone ones. Not adopted; repeatability stays with `max_num_seqs = 1` (the distributed default), and the published option keeps two sequences. The optimization catalog (P26) and the benchmarks' reachability section say so.
- Architecture: the memory probe has its own row naming every method, `examples/` names both profiles, the layer table adds `switch.resume`, the capability checks and the base-image verdict and says `server_config` imports no side-effect module, and a new section states where a worker extension belongs (`runtime/` for what a serving launch loads, `validation/` for fixture-only workers, and why `expert_worker` is the exception) and which runners run only as `python -m glm53_setup.validation.<module>`.
- Launcher text: `server --help` describes the serving launcher instead of "a serial TP=2 reference experiment", the supervision banner names the engine stall among the stops, and the usage line of `tools/check_prefix_cache.py` is one command again.
- Example comments, English and Japanese lines in step: `GLM53_MOE_ORDER_API=2` for new launches (1 only to recover a launched pair), `decode_graphs` marked not adopted (P06), the memory probe's readings, the component worker's scope, the Dynamo reset on torch 2.12.1 as well as 2.13, and prefill times pointed at the benchmarks instead of pre-FA2 figures. Both files parse to the same TOML.
- Code comments that 1.6.0 to 1.16.0 had overtaken: the reference attention and its patch describe serving instead of a validation-only fallback, the memory probe's docstring names all its methods (comments only in the mounted files), and `server_config` no longer calls MTP depths 2, 4 and 5 unmeasured nor keeps deferral notes for PP and EP, which were not adopted.

## 1.16.0 — 2026-09-26

### Removed

- `runtime.mla_decode_cpb` is retired. It never ran in serving: the image sets `GLM53_REFERENCE_ATTENTION=1`, and every sparse-MLA forward, the 11 DSA layers and the MTP draft layer alike, returns through the reference NoPE attention before the FlashInfer decode call the key's patch changes. The 1.14.0 kernel evidence came from a harness that set that flag off; in the 2026-09-26 reachability runs a trace on the serving pair counted 434 sparse-MLA calls for a request alone and 448 for a pair, all through the reference attention. The 1.14.0 statements that the key pins the two-sequence decode split and that a request's attention no longer depends on the requests sharing its step are withdrawn; the 1.14.0 measurements (lone completions, decode speed and NLL unchanged, two-sequence completions byte-identical to 1.13.0's) are consistent with no effect. Both examples drop the key. A profile that still carries it launches unchanged, with the same environment, fingerprint and checks (`true` still needs `GLM53_MLA_DECODE_CPB_API=1` and excludes decode Graphs), and `server preflight` records `mla_decode_cpb_retired` under `warnings`. The patch, its helper, the image marker and the key's acceptance go with the next image build ([server configuration](docs/server-configuration.md#repeatability-switches)).

### Documentation

- Attention routing corrected. The documents said decode stays on the reference path; the reference attention sends calls of more than six query rows to FA2, so one sequence's decode steps stay on it, but at MTP depth 3 a partner makes a verification step eight rows and sends it through FA2, whose result for a row moves by one BF16 ulp with the partner's row count and lengths. That is not the main cause of two-sequence differences: with every attention call forced eager on the serving pair in the same runs, 7 of 8 two-sequence completions still differed from their lone ones, and the MoE is the lead suspect. The operating conclusion is unchanged: with `max_num_seqs = 1` completions repeat under any load, with two sequences they do not ([measurements](docs/benchmarks.md#reachability-in-serving-2026-09-26)). Server configuration, validation, the README, the catalog (P26 retired, P05), the optimization overview, operations, architecture and the `fa2_attention` comments of both examples and of `glm53_setup/runtime/fa2_attention.py` (comments only; the file is mounted over the image) say so.
- Benchmarks 1.15.0: CPU placement on the reference pair (2026-09-26). Every published decode figure has both ranks' workers on performance cores; with either rank on efficiency cores decode falls to about a third, with the same completions and acceptance lengths, which `nodes[].cpuset_cpus` prevents. The README headline and server configuration point there.
- Documents brought up to 1.15.0 and shortened, each fact kept in one owner. Statements the later releases had overtaken now read as they are: the two-sequence profile's acceptance (2026-09-23), the repeatability switches of 1.12.0 and 1.14.0 (the latter retired above) instead of "under investigation", MTP k=3 in the template instead of an optional experiment, decode Graphs and Expert Parallel measured and not adopted, FreedomBench closed, the harness route decided, and every "images built from this version" replaced by its version.
- Server configuration is reorganised: the defaults and the published option first, then a key reference grouped as repeatability switches, attention/cache/checkpoint, parallelism, image input, API and diagnostics (the memory probe's methods as a table) and LPA with prefix caching, then the commands. The image contract lists every capability marker, the setting that makes preflight require it and the version from which images carry it; the document map names it as the owner. Two corrections: MTP depths one to five were measured on the requantized checkpoint and 1, 3 and 4 on the pinned one (not all five on both), and LPA excludes `runtime.fa2_attention` and accepts MTP depth 1 or 3 only.
- LPA: the recipe for enabling it now sets `runtime.fa2_attention = false`; without it the launcher refuses the profile, because the template enables FA2.
- Validation: the concurrency scope is a table per profile with its evidence and the repeatability rule; the full-model section gains subsections (loading and API checks, multibyte output, repeatability); the account of the three launch states is one paragraph, with the numbers in benchmarks 1.9.0 and the steps of the search in this changelog (1.10.0 to 1.12.2).
- Benchmarks open with an index of the release sections; each initiative's decision paragraph is one result line pointing at the catalog. The optimization overview, catalog and performance investigation keep their roles but point at the owners of numbers and decisions instead of restating them. Harnesses, FreedomBench, image input, NCCL, licensing, component validation, speculative decoding, APC-first LPA, candidate order and indexer reuse are shortened the same way.
- The setup runbook's evidence row for memory points at the same-night measurements of both profiles on 1.8.0 instead of quoting the published option's figures under the defaults' acceptance.
- The acceptance of 2026-09-22 is stated as it was made: for both profiles at one active sequence, for the public demonstration at 生成AIなんでも展示会#6 on 2026-09-23, and kept for routine use within the same scope (README, setup runbook, validation, operations).

## 1.15.0 — 2026-09-26

### Added

- `nodes[].cpuset_cpus` (optional, per rank; contributed by circlemouth in #2): a Docker CPU list passed to `--cpuset-cpus` when the rank's container is created. On a BIZ 1.11.2 pair, #1 measured short-input decode at 36.13 tokens/s with rank 0 on performance cores and 17.01 on efficiency cores, at the same 2.4 GHz cap and with matching output hashes and MTP acceptance. The launcher rejects malformed, reversed or repeated ranges; `server preflight` checks that the CPUs are in the launcher's affinity mask; after `docker run` a rank is recorded only when `HostConfig.CpusetCpus` names the same CPUs, and the new container is stopped otherwise. Unset, the Docker command is unchanged, and both examples leave the key commented out because core IDs are host-specific ([server configuration](docs/server-configuration.md#optional-cpu-placement)). On the reference pair, rank 0's host has ten 2.8 GHz cores (0–4, 10–14) and ten 3.9 GHz cores (5–9, 15–19), and without a CPU set its worker was on a 3.9 GHz core when checked on 2026-09-26; pinned against unpinned placement has not been measured here.

## 1.14.0 — 2026-09-26

### Added

- `runtime.mla_decode_cpb`: the sparse-MLA decode takes FlashInfer's `chunks_per_block` from the tokens of one sequence (2 for a draft step, 6 for a verification step at depth 3; 32 heads per rank, top-k 2,048) instead of from the whole step. FlashInfer 0.6.18 picks the value from the step's token count, so a second sequence changed how a request's candidates are summed (every attention row of the request, on one GB10); the pinned values are its own at one sequence, so a request alone computes bit for bit as before. The key is on in both examples, needs an image built from 1.14.0 (marker `GLM53_MLA_DECODE_CPB_API=1`, required by preflight; source-pinned patch `glm53_setup/runtime/patch_mla_decode_cpb.py` on the SM120 backend) and is refused together with `runtime.decode_graphs`. On the reference pair, on both profiles, lone completions, decode speed, NLL and long-input answers are unchanged ([measurements](docs/benchmarks.md#measurements-on-1140)). Rebuild the reference image and update `reference_image`, or set the key to `false`.

### Documentation

- Repeatability with more than one sequence, measured on both weights: a request that shares steps with another still changes through the NVFP4 Marlin MoE (its split along K follows the number of expert blocks in the step) and through the prefill-sized kernels of a step shared with a prefill, so completions repeat under any load only with `max_num_seqs = 1`, where requests sent together queue (18 of 18 equal to their lone completions); two sequences give about a quarter more throughput when requests overlap. Validation's concurrency scope, the README row and server configuration say so.
- A prompt longer than one 4,608-token cache block, reused from the prefix cache before and after other long prefills, answers correctly with the same completion each time (the kpool seed fix of 1.13.0).
- Reported upstream: flashinfer-ai/flashinfer#5553 (the decode split follows the call's token count), and the NVFP4 case on vllm-project/vllm#46639 (Marlin MoE batch invariance). The optimization catalog adds P26 with the variants not adopted.

## 1.13.0 — 2026-09-25

### Fixed

- Reference images patch the pinned vLLM's kpool prefill seed kernel to address the indexer's raw-tail blocks by the tail's own strides (marker `GLM53_KPOOL_SEED_STRIDE=1`, `glm53_setup/runtime/patch_kpool_seed.py`; the same change as vLLM pull request #57477, merged upstream on 2026-09-20). The tail shares the indexer cache at the indexer's padded block stride, while the seed kernel of vLLM 385dce36 addressed it densely: every prefill left the request's own tail block unseeded and wrote the last tokens' raw key and gate into a low-numbered indexer block, which can belong to another request. The damage is to sparse top-k selection over long contexts, most of all a long prompt reused from the prefix cache; the MLA KV is untouched. The upstream regression test fails on the 1.12 image and passes on the rebuilt one. On the reference pair no measured output changed: the decode checks, teacher-forced NLL, the two-sequence completions and the defaults' weight digest are bit-identical before and after, and the long-input checks answer correctly on both profiles ([measurements](docs/benchmarks.md#measurements-on-1130)). Rebuild the reference image and update `reference_image`; no check requires the marker, and the patch goes when the pinned vLLM passes the upstream fix.

### Documentation

- Two sequences: a pair's completions repeat whenever the server prefills the two requests in the same order; kernel runs on one GB10 point at the NVFP4 Marlin MoE, whose rows change when another request's decode rows share the call, while cuBLAS BF16 GEMMs keep their bits and Marlin's dense GEMMs change only at prefill-sized row counts. The kpool seed fix is not the cause. Validation's concurrency paragraph says so.
- The three-position reference's canonical form is now a fenced framing that marks the articles as background and asks for the value after `=`; both profiles answered it correctly 3 of 3, and the README row carries it, with the prompt in the benchmarks. The old framing again split the two profiles (AXL copied an article in all three runs, the defaults returned the codes).
- The published option's NLL in the README is now the two-sequence profile's with `runtime.inductor_deterministic` (1.6270 / 1.9946 / 0.9601 / 0.6275, identical on the 1.12 and 1.13 images), lower on all four texts than the one-sequence profile's 1.6645 / 2.0024 / 1.0031 / 0.6279. The Cons row reads 1 to 6% higher than the defaults on three of four texts.
- On this hybrid model the prefix cache hits only in whole 4,608-token blocks; the benchmarks say so where the fix's reach is discussed.

## 1.12.2 — 2026-09-25

### Documentation

- Validation's multibyte paragraph notes that tonyd2wild's checkpoint guard (`abb38bb`) refuses ModelOpt builds that quantize attention as the builds that garble multibyte text, a different reading of vLLM #54150 from the gate/up scale mismatch this repository describes. BIZ AXL quantizes the attention projections (its own W4A16 repack, not ModelOpt) and passed `server mojibake` on 2026-09-21 and again on 2026-09-25 (six of six answers, no replacement, surrogate or control characters); six answers do not decide between the readings.
- The Dynamo reset that turns `TORCHINDUCTOR_DETERMINISTIC` off after the first compiled frame (pytorch/pytorch#198563) also occurs on torch 2.12.1, as reported on a GB10 in vllm-project/vllm#58636 and reproduced with the PyTorch issue's example. Server configuration and validation no longer present it as specific to torch 2.13; the 1.12.0 entry below describes the torch the image carries.

## 1.12.1 — 2026-09-25

### Documentation

- The distributed defaults with `runtime.inductor_deterministic` (the template as shipped) were launched three times: both ranks served the indexer's key norm at one config and received the same key at every call, no Inductor config differed between the ranks, all three launches gave the same completions, and decode matched the defaults' published speeds (32.80–32.84 / 20.76–20.82 / 27.53–27.64 against 32.01 / 20.67 / 26.68 tok/s on counting / prose / code). Validation, benchmarks and the README repeatability rows no longer say the defaults were not launched with the key.

## 1.12.0 — 2026-09-25

### Added

- `runtime.inductor_deterministic` (on in both example profiles; absent or `false` keeps Inductor's timed choice): Inductor picks the configs of its reductions without timing, the same on both ranks, and compiles into a cache of its own (`TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor-deterministic`). The replicated kpool indexer normalises its key through a `torch.compile` leaf with three candidate configs; each rank picked one by timing at every launch, one of them sums a row in a different order, and when the ranks picked from different classes, completions forked where pools tie. With the key, torch 2.13's Dynamo would still turn the mode off after the first compiled frame ([pytorch/pytorch#198563](https://github.com/pytorch/pytorch/issues/198563)), so the launcher also mounts `glm53_setup/runtime/inductor_pin.py` and a one-line `.pth` that keep the setting forced. No image support is needed; the first launch with the key compiles the indexer's leaves anew. Adding the key moves a profile's fingerprint. [Server configuration](docs/server-configuration.md#distributed-defaults) has the details.
- The memory probe reads what each serving worker runs: `autotuners` (each Inductor kernel's file, size hints, the config it serves and whether that came from the cache or a benchmark in this process) and `inductor_state` (Inductor's settings, the `TORCHINDUCTOR_*` environment, the forced value, every write to the deterministic switch with its caller, and one fresh compile). `tools/kernel_hashes.py` records both per rank before and after hashing. The indexer's stage hashes now run the served call on its strided view (`k_norm_served_rows*`) and find the rotary on the MLA wrapper.

### Fixed

- **Launches no longer fall into one of three numerical states.** Until now the pair computed in one of three states across launches (sixteen checked launches of the serving image: eleven, three and two). The cause is the indexer's compiled key norm, timed per rank (see Added). The timing was also redone at every launch: the persistent Inductor cache held graphs copied on 2026-09-15 from a container's `/tmp/torchinductor_root`, which still saved and looked up their choices there. The deterministic key compiles fresh graphs into its own cache. Three launches with the key computed in state 1 with the same completions, decode speed within the spread of the earlier state-1 launches. The same pattern is reported to vLLM as [vllm-project/vllm#58636](https://github.com/vllm-project/vllm/issues/58636).

### Documentation

- Validation (the launch states: the cause, the intervention that reproduced each state, the fix), benchmarks 1.9.0, the README repeatability rows and citation, server configuration and architecture (`inductor_pin`).

## 1.11.3 — 2026-09-23

### Documentation

- README, headline measurements: the table gains a category column and is grouped as prefill, decode, long input, quality, repeatability and memory, each profile's numbers unchanged. The status row on identical requests states what thirteen launches of the serving image showed: bit-for-bit repeats within a launch, three numerical states across launches (nine, three and one; three of the five switches of 2026-09-23 changed state), the checks after every switch, and the one place the states differ.
- Validation and benchmarks 1.9.0: the Probe5 launch (state 1, the 1.11.1 checkout) agreed across the ranks on all sixteen kernel hashes and on the indexer's key at every call; against the state-2 launch rank 0's prefill is identical and rank 1's key differs from the first MLA layer, so a launch's state is whether rank 1's process makes the indexer's key with the same bits as rank 0, and the deciding reading is the compiled norm's hash in a state-2 launch.

## 1.11.2 — 2026-09-23

### Documentation

- **The published option's two-sequence profile is accepted for routine use (2026-09-23)**, for two active sequences at up to about 200K tokens each, on three launches of that profile that day (states 2, 1 and 2), each with the weight digest, the decode check and the traces after its switch, the third with the kernel hashes on both ranks; the task-level checks ran on the first launch and the sparkDash and tool-eval runs on the second, and the three switches completed without recovery. Validation's concurrency scope owns the verdict and its extent; SETUP step 6 gains the evidence table; the README summary and status row, and the operations page, say the same. The changed completion under two sequences stays declared behaviour, not an accepted defect: whether a source-pinned patch can make a completion independent of the requests it shares a step with is under investigation with the launch-state work.

## 1.11.1 — 2026-09-23

### Added

- `kernel_hashes` also hashes the stages between the indexer's projections and its op with one layer's own weights: the compiled layer norm (`_fused_indexer_k_norm`) and its eager fp32 reference, the indexer's rotary embedding on query and key, and the key projection (`layer`, default 19). The 1.11.0 launch (state 2) agreed across the ranks on every fixed-input kernel, while a trace three levels deep with inputs showed the ranks' replicated indexers receiving different keys at prefill in eight MLA layers from identical projection outputs; those stages are where the key is made. A failure in this part names itself in the answer instead of failing the method.

### Documentation

- The setup runbook (its opening status and the step 6 verdict) and the operations page state concurrency the way the README and validation already do: one active sequence is the accepted scope on the distributed defaults; the published option's two-sequence profile is checked at task level on one launch and waits on more launches for routine use, with [concurrency scope](docs/validation.md#concurrency-scope) as the owner.
- Two anchors in the Japanese benchmarks and validation pages named the "evidence, not production qualification" section by another wording and did not resolve; they now match the heading.
- Validation and benchmarks 1.9.0: the Probe4 launch and the deep trace, and the two candidates they leave (the Inductor layer-norm kernel, the key side of the rotary).

## 1.11.0 — 2026-09-23

### Added

- **The memory probe hashes the indexer's kernels inside the serving workers (`kernel_hashes`), and `tools/kernel_hashes.py` compares the ranks and an earlier launch.** On 2026-09-23 one rank's copy of the replicated kpool indexer was the first call to differ between two launches on identical inputs, and thirteen fresh processes on one GB10 computed the indexer's kernels bit-identically, so the difference lives in the serving processes themselves. The method runs the fp32 head gate and bf16 gate score, the fused FWHT quantisation, the pool cache's prefill write and decode tail update, and DeepGEMM's paged MQA logits with the stable top-k, on fixed inputs at the served shapes, and returns a hash of every output; the tool asks both ranks, compares them with each other and with the previous launch's record, and exits 1 on any difference, naming the keys. Taken after every switch beside the weight digest ([launch safety](docs/launch-safety.md#after-a-switch-the-decode-check)), the next launch in another state names the computation on the spot. The probe's other methods are unchanged.

## 1.10.5 — 2026-09-23

### Documentation

- README, headline measurements: the published option's column is the profile the reference pair serves, the two-sequence AXL profile as measured on 2026-09-23 (decode 45.04 / 28.16 / 37.67 tok/s after 2,048 tokens, 165.7 s for a ~200K passphrase and 330.2 s for two together, sparkDash and tool-eval rows, memory at 6 GiB of KV); rows that night did not re-measure keep the last measured value, dated; the paragraph no longer lists every earlier version.
- Validation and benchmarks 1.9.0: the indexer's four computations named on 2026-09-23 (the fp32 head gate, the fused FWHT quantisation, the pool cache's write paths, DeepGEMM's paged MQA logits with the stable top-k) ran in thirteen fresh processes on one GB10 at the served shapes and were bit-identical, so the launch-state difference belongs to the serving process's own state; the next measurement is a probe method that runs them on both ranks of a live launch.

## 1.10.4 — 2026-09-23

### Documentation

- The CUTLASS W4A4 fixture row leaves the README status table and the validation evidence table, and the sentence that called its differences "observations requiring investigation" is gone: that path was tried once on the four-layer fixture in the first week and never served, every profile since runs Marlin W4A16, and a present-tense row read as an open quality defect of the stack. The history stays in the changelog of that week; the precision paragraph and SETUP still say that W4A4 behaviour is not claimed.
- Validation and benchmarks 1.9.0: the difference between two launch states is named. A launch of 2026-09-23 in another state loaded the same weights (digest equal on both ranks) and its request trace named the first differing call, the replicated kpool indexer of layer 19 on rank 1 at a decode verification step in both the prose and the code request, with identical inputs and a different candidate set; in the earlier launch that rank's indexer had disagreed with rank 0's at exactly that call. The difference between the two launches is one rank's copy of the replicated indexer scoring a near tie differently; the third state is traced only on the counting request, and what differs across processes inside the indexer's scoring or the pool cache it wrote is the next measurement.
- Benchmarks 1.10.4: sparkDash DecodeBench and tool-eval-bench repeated unchanged on the profile the reference pair serves. sparkDash decode 48.2 / 31.4 / 41.3 / 34.9 tok/s on its four prompts against 36.2 / 26.7 / 31.7 / 26.3 on 1.5.0, the serving profile's gain since then. tool-eval-bench 88/100 (122/138), 55 pass, 11 partial, 3 fail, the same three failures as 1.0.0, so the Safety Gate is still not passed on TC-43. The README headline points at both.

## 1.10.3 — 2026-09-23

### Added

- A test assembles the docker command and the environment of both examples for each rank with the same image supplied and keeps their difference to the five settings of the published option (the derived checkpoint's mount, its two overlay mounts, the dedup variable, the second sequence, twice the KV) and the two arguments that follow from the first (the model argument, the fingerprint label).

### Documentation

- Server configuration: the published option against the defaults, as a table of the five settings and what each adds to the launch, and the check of 2026-09-23 in which the AXL example, its placeholders replaced by the reference pair's values, passed `server freeze` and `server plan` and every preflight check on both ranks but `startup_memory`, which a serving pair cannot satisfy; the example differs from the served profile only in the probe, the dev routes, the warmup's long rung and the unused LPA placeholders.
- Validation, concurrency scope: the published option's two-sequence profile passed task-level checks on one launch on 2026-09-23 (two ~200K passphrase requests together, two tool calls together, an image with a prose request) without preemption, and its completions under two sequences differ from a request alone because batch-invariant mode is unavailable; routine-use acceptance of that scope waits on more launches. Benchmarks 1.10.2 carry the numbers (330 s for two 200K requests against 166 s alone; decode 32.1 / 21.8 tok/s with two sequences against 45.6 / 28.2 alone; KV 54% at peak). README: the status lines, the concurrency row and the headline paragraph say the same. Launch safety: the decode check after a switch runs with nothing else in flight, for that reason; the KV capacity section no longer calls the 2,112-token evaluation the accepted two-sequence scope.

## 1.10.2 — 2026-09-23

### Added

- **`examples/server.axl.example.toml`, the profile of the published option.** The distributed defaults keep `examples/server.example.toml`; the option's example adds the `runtime.derived_checkpoint` table for the repacked weights (NVFP4 BIZ AXL) with the overlays the repository ships, `runtime.prefix_page_dedup`, two active sequences (`max_num_seqs = 2`) and 6 GiB of KV per rank, which the repacked weights pay for: they load 4.4 GiB less per rank than the pinned ones. First launched on the reference pair on 2026-09-23: 606,881 tokens of KV, 2.32 times a 256K request by the engine's count, warmup passed, two concurrent requests served (counting 32.2 and prose 22.3 tok/s while both ran). A test keeps both examples valid and their shared sections equal.
- **The launcher refuses more than 3 GiB of KV without the derived checkpoint** (`check_kv_budget`). On the reference pair the pinned weights leave the head 5.5 GiB at 3 GiB of KV against a 3 GiB reserve, so twice the KV would stop the container; the repacked weights leave 10.5 GiB. The message says so.

## 1.10.1 — 2026-09-23

### Fixed

- `weight_digest` raised on a 0-dim parameter (a scale): "self.dim() cannot be 0 to view Float as Byte". The fingerprint flattens a tensor before viewing its bytes (`flat_bytes`), which leaves every other tensor's print unchanged. Measured on the reference pair on the first launch with the probe.
- `tools/weight_digest.py` puts the repository root on `sys.path`, so `python3 tools/weight_digest.py` runs from any directory as the documentation shows.

### Documentation

- README opens with a summary: what the stack is, its acceptance status and scope, the two served profiles, the precision it runs at, the licensing shape and what is not validated, each as a pointer to the owner section; it absorbs the two introductory paragraphs it replaced and carries no measured number or version.
- Validation states that NVIDIA's model card does not describe this serving: its BF16-versus-NVFP4 table was measured on GB200 through vLLM and SGLang under the card's W4A4 recipe, while this stack runs Marlin W4A16 on GB10; the numbers that describe this serving are named. The README's precision paragraph points there.
- Document map: the GitHub description and topics restate the README summary without numbers or a version, and are edited with `gh repo edit` when the summary changes.

## 1.10.0 — 2026-09-23

### Added

- **The memory probe fingerprints the loaded weights (`weight_digest`), and `tools/weight_digest.py` records and compares them across launches.** Every parameter and buffer of the served model and the draft, as it sits on each rank after loading, gets the two byte sums the request trace already uses; the record keeps the rows and per-layer digests, and a later launch's record names the tensors that differ. This is the first question a launch in another numerical state ([1.9.0](#190--2026-09-22)) has to answer: same bits computed differently, or different bits loaded. `trace_end` gains `export`, which returns the traced rows so `trace_differences` can compare two launches from records instead of one process. The shared fingerprint and model discovery moved to module functions; the trace behaves as before.

### Documentation

- Launch safety: the weight digest as part of the check after a switch, when the profile carries `validation.memory_probe`. Server configuration and the architecture table name the method and the tool. README: the FreedomBench line says the evaluation is closed for the serving profile instead of pending.

## 1.9.5 — 2026-09-23

### Documentation

- FreedomBench closed for the serving profile: on 2026-09-22 the pinned English suite answered 60 of 60 on the first attempt with no refusal and the long-prefix pilot 6 of 6, both on the profile the reference pair serves; the four-profile matrix and the LPA case retired with the single served profile, and what was not run (Japanese translation, opposed framings, evidence placement) is listed. Validation says the same in one sentence.
- Benchmarks 1.9.0: two checks that narrow the launch-state question — thirteen fresh processes computing the pair's GEMM and KDA kernels bit-identically (pins included), and the four-layer MTP fixture launched TP=2 across the pair twenty times with identical completions — leave the full-size model's load and launch as the remaining candidates.

## 1.9.4 — 2026-09-22

### Changed

- The source-pinned vLLM patches applied at image build share one skeleton, `glm53_setup/runtime/pinned_patch.py` (the hash gate on the pinned file, the `--package`/`--check` command, the record written beside the package); each `patch_*` module states only its target, its pin and its anchors as a pure `patch_text(text)`, so all seven can be tested on an excerpt of the pinned file (three could not be before). The entry points the Dockerfile runs, the flags, the printed hashes and the record files are unchanged, and a contract test reads the Dockerfile to keep them so.

- `apc-lpa-fixture` reads its fixture through the same gate as `fixture-run` (`run_fixture.read_fixture`). A fixture that is not a byte-verified four-layer test fixture is refused with that gate's sentence, "Only a byte-verified four-layer test fixture is allowed", and `all_tensor_bytes_verified` must be `true` rather than merely truthy. Engine settings, the measurement terms and the report are unchanged.

## 1.9.3 — 2026-09-22

### Documentation

- Statements the routine-use acceptance of 2026-09-22 had overtaken: the repository instructions (`AGENTS.md`), the operations page and the validation page no longer say that TP=2 is unqualified or that ZCode Desktop and Claude Code are required targets; they point at SETUP step 6, the record of the acceptance and of where each item's evidence is, and at the harness decision. The setup checklist and its suggested AI task no longer ask for the "receipt workflow" that the removed `service` launcher once gated on; they ask for the step 6 items and their evidence paths. Operations and validation read the same way in Japanese.
- Architecture: the module table names every module of `glm53_setup/`, `glm53_setup/runtime/`, `glm53_setup/validation/` and `tools/` (the source-pinned patches of 1.7.0–1.9.0, the FA2 path, the agreement and requantization checks, the decode tools, `release_notes.py`), and the `overlays/` and `.github/workflows/` directories; the sentence that counted "two vLLM source files" now describes every build-time patch and the launch-time overlay check.
- Document map: the overlay manifest (English-only, as a provenance record) and the ZCode guard hook setup (now an English/Japanese pair) have rows; the catalog row no longer enumerates initiative IDs; the decode check after a switch and the routine-use acceptance have owners in the single-source table.
- `tools/check_publication.py` gains four checks, each with a unit test: the README's short-name citation must carry the version in `pyproject.toml`; each document map must link every `docs/*.md` of its language; every module of `glm53_setup/` and `tools/` must be named in the architecture page, by file name or by a pattern such as `benchmark_*.py`; and `--plans` checks that the relative links of the untracked `docs/plans/` resolve on disk. The audit passes on this tree.

## 1.9.2 — 2026-09-22

### Added

- `tools/decode_check.py` and `tools/decode_divergence.py`: the decode check after a switch (512 greedy tokens after a fixed ~2,048-token prompt, three task types, the completion text and token ids kept through `return_token_ids`) and the comparison of two launches at their first diverging token. The same measurement as the decode rows of the benchmarks since 1.6.0, now in the repository instead of a records directory.

### Documentation

- Launch safety: "After a switch: the decode check", the routine the 1.9.0 launch-state finding calls for (three samples agree within a launch; hashes compared with the previous launch of the profile; token ids and both ranks' container logs kept when they differ).
- Benchmarks 1.9.0: how rare the other states are in the decode records since 2026-09-18 (one profile's completions recur across days; the two states seen once recur nowhere), and that vLLM's batch-invariant mode sets `CUBLAS_WORKSPACE_CONFIG` itself, which this stack leaves unset until a launch in another state points at the BF16 GEMMs.

## 1.9.1 — 2026-09-22

### Documentation

- Short names, so that the stack and the published weights can be cited in one word each: **NVFP4 BIZ** is this serving stack (cite as "NVFP4 BIZ 1.9.1") and **NVFP4 BIZ AXL** is the published option's weights (attention projections and `lm_head` in W4A16). README, the GitHub description and the Hugging Face model card carry them; the repository names, the package name and the body terms (distributed defaults, published option) are unchanged.

## 1.9.0 — 2026-09-22

### Changed

- **A prefix-cache page whose hash is already cached is not registered again (`runtime.prefix_page_dedup`).** The pinned block pool registers every full block under its hash even when another block with the same hash is already in the cache, and under a draft (MTP) the prefix lookup drops the last matching block and recomputes it, so every re-sent history added one block per KV cache group (three here) to the LRU queue under a hash that already had one. On the four-layer fixture with depth 3 and dense retention the copies grew by three per re-send; with the option on they are zero, the cached-token counts and the completions are identical to the pinned pool (27 of 27), and the request times are the same. On the reference pair the copies matter because the KDA state checkpoints leave room for about one long history: a 57K-token history re-sent fifteen times evicted a 28K-token one with the option off, and left it cached with it on. The change is a source-pinned patch (`glm53_setup/runtime/patch_prefix_dedup.py`, applied to `vllm/v1/core/block_pool.py` at image build; `patch_prefix_dedup --check` verifies the pinned file) that asks one question in `_insert_block_hash`: with `GLM53_PREFIX_PAGE_DEDUP=1` a block whose hash already has a cached block keeps no hash and returns to the front of the free queue when its request ends. Block ids and the free-queue order are otherwise unchanged; hits are served by the copy cached first. The profile key (optional; absent = off, as the pinned pool behaves; the template does not set it) writes the environment variable on both ranks and needs an image built from this checkout, marked `GLM53_PREFIX_DEDUP_API=1`; `server preflight` reports `prefix_dedup_support`. The serving profile of the reference pair sets it. The fingerprint of a profile that sets it changes; a profile without it keeps its fingerprint and runs on the earlier images.

### Documentation

- Benchmarks: the re-send probe on the reference pair (57K/28K histories, fifteen re-sends, off and on, same night and image) and the launch-to-launch check behind the adoption: six launches of the new image, one profile relaunched four times, gave the same completions in five launches and a different one in one, independent of the option and of the profile. The image, the launch arguments, the runtime caches and Triton's autotune tables were compared and are not the cause; the remaining candidates leave no trace, so the next such launch is caught with the completion token ids and both ranks' container logs. README: the repeatability row says so, and the launch note in `server_config.py` no longer claims the autotune pin removes every cause.
- Optimization catalog P25 (this change) and server configuration (`runtime.prefix_page_dedup`).

## 1.8.1 — 2026-09-22

### Documentation

- Validation: the full-model scope section opens with the routine-use status of 2026-09-22 and reads its older "does not certify" sentences as history; a concurrency-scope subsection states that more than one active sequence is not supported in the accepted profile and that concurrent serving needs KV per sequence, i.e. more ranks (TP=4 recommended, TP=3 not). README's status table gains the same row.

### Fixed

- **The lock is part of every launch fingerprint, and 1.8.0 had edited it.** `config/runtime.lock.json` is hashed into the fingerprint of every profile, so the 1.8.0 change of its `status` and `full_model_inference_validated` fields changed every profile's fingerprint: a checkout at 1.8.0 computed a different value for the serving profile than the one its pair was launched with, and `cluster switch` refused to address that pair ("Frozen launch manifest no longer matches this checkout/lock"). The lock returns to its 1.7.1 content and fingerprints return to their published values (the serving profile `948613031b31…`); the routine-use acceptance of 1.8.0 stays recorded in the README, SETUP §6 and the harness matrix, which is where it belongs. A unit test now pins the lock's hash so that the next edit is a deliberate one with a CHANGELOG entry.

## 1.8.0 — 2026-09-22

### Changed

- **The published option's KDA input projection is split.** The derived checkpoint (attention projections and `lm_head` in W4A16 NVFP4, `requant_target = "l"`) now declares the KDA input projection of the 34 linear-attention layers as `q_proj`, `k_proj`, `v_proj` and one merged `in_proj_bfg_a` (`b`, `f_a`, `g_a`) instead of the pinned six-in-one `in_proj_qkvbfg_a`; the weights are byte-identical, only `config.json`, `hf_quant_config.json` (`producer.in_proj_layout = "split-qkv-bfg"`) and the two source overlays changed. A kernel measurement placed the option's prefill cost in the width of that one W4A16 Marlin GEMM (every fused width from 12,288 to 12,800 columns costs 1.6–1.8 times a BF16 GEMM on 2,048-row chunks; three 4,096-wide GEMMs plus the 288-wide tail cost 1.14, 1.3 with the concatenation), not in its padding. On the reference pair, same night as the fused layout and the distributed defaults, the split option prefills 2.6% faster than the fused one on 38,962 tokens, 3.2% faster on the 199,652-token request and 3.1% faster on the 261,461-token request, the size of the penalty 1.7.1 measured, and faster than the defaults; decode and memory are unchanged, the 200K and 261K requests are answered correctly, and against the fused layout the four-layer fixture agrees at the level of two launches of one fixture (full-vocabulary KL 1e-4, candidate-set Jaccard 0.986). Completions of the option change with the layout (BF16 rounding order) and repeat bit for bit within it. The reference pair serves the split option since 2026-09-22 (fingerprint `948613031b31…`).
- **The two source overlays ship in `overlays/`** (`kda-quant-split.py` over the pinned `kda.py`, `mla-quant-split.py` over `model.py`, with their SHA-256, base file hashes and markers in `overlays/README.md`). The Hugging Face model card already required them; the repository now provides them. Ruff excludes the directory (vLLM-derived sources).

### Documentation

- Benchmarks and README: measurements on 1.8.0, the three-arm same-night run; the README headline carries the split option. The 1.7.1 kernel figure is annotated: it was taken at a per-rank width of 12,416 columns without the output slice; the real width is 12,576, padded to 12,608, and the slice costs 0.48 ms per layer per chunk.
- Optimization catalog P23 and server configuration: the split layout, the overlay pairing rule (an overlay pair belongs to one checkpoint revision; a mismatch fails at load) and the corrected width.
- Routine-use acceptance closed for the serving profile (2026-09-22): the lock's status and `full_model_inference_validated`, the README status sentences, SETUP §6's evidence table and the harness matrix (the npm ZCode CLI is the accepted route; the official Desktop stays BLOCKED on feedback #270; Claude Code is skipped by decision) say so instead of "still open".
- SETUP: a shortest-path section to a smoke test ahead of the numbered steps.

## 1.7.1 — 2026-09-22

### Documentation

- Optimization catalog and README: tonyd2wild's 2026-09-20 note reached the same attention/MLP projection set as P23 independently (TP=4, quality unmeasured, no code adopted).
- Validation: vLLM pull request #55122 restated itself on 2026-09-21 as a performance change whose determinism is a side benefit, after a census on its own traffic found no boundary ties; this stack reproduced and caught them, so `runtime.stable_indexer_topk` stays.
- Benchmarks: the published option and the distributed defaults measured back to back on the same night and image (1.7.0 runtime): the repack prefills 1.8% slower on 38,962 tokens, 1.2% on the 199,652-token request and 3.4% on the 261,461-token request, against a 0.45% launch-to-launch spread; a kernel measurement places the cost in the fused KDA input projection under W4A16 Marlin at prefill widths. The reading added to the 1.7.0 section on the morning of 2026-09-22, that prefill was unchanged within the spread, compared launches on different nights and is annotated. The defaults on the 1.7.0 runtime match their 1.6.0 numbers.
- README: the headline table carries the same-night pair, and the cons of the option name the prefill cost.

## 1.7.0 — 2026-09-22

### Changed

- With `runtime.canonical_moe_order = true`, a new launch requires the image marker `GLM53_MOE_ORDER_API=2`: `server preflight`, `server start` and the pre-stop checks of `cluster switch` refuse a profile whose image carries only marker 1, which an earlier image with a mis-sized sort buffer shares. Rebuild the reference image from this checkout and update `reference_image`. A pair already serving from a marker-1 image stays restorable: `cluster switch` prepares it, and restarts it after a failed switch, as a recovery target that keeps marker 1 (`server start --recovery`, passed by the coordinator), and records the allowance under `warnings`. Fingerprints are unchanged.
- Launches set `TRITON_CACHE_AUTOTUNING=1`, so the kernel configurations Triton autotunes are kept with the compiled kernels under the persistent `TRITON_CACHE_DIR` instead of being tuned again on every launch. On the four-layer fixture ten launches without it computed in two numerical states, and the state followed the `num_warps` that one KDA kernel (`merge_16x16_to_64x64_inverse_kernel`) happened to pick; with it six launches computed in one state and started about 50 s sooner. The reference pair launched three times with it and repeated the 1.6.0 completions each time; the full model had not shown the split before, so that effect is established on the fixture only. Profile fingerprints do not change.

### Fixed

- Reference images patch the pinned vLLM's slot-mapping kernel to read a block table only inside its row (marker `GLM53_SLOT_MAPPING_GUARD=1`; the same guard as vLLM pull request #54296 for issue #53982, which was open when this was written). Without it the kernel reads past the 32-entry row of the indexer's tail-scratch KV group from position 128 on, one byte further per token, and faults when that reaches unmapped memory: on the reference pair's serving profile every request above about 250K tokens ended in a CUDA illegal memory access on both ranks (248,954 tokens completed, 252,958 faulted, 261,461 faulted four times out of four), while the distributed defaults performed the same read without faulting. With the guard the fixture shows zero invalid reads under compute-sanitizer and log-probabilities byte-identical to an unguarded image, and the serving profile completed 252,914 and 261,461 tokens with correct answers and the 1.6.0 decode completions. Rebuild the reference image and update `reference_image`; no check requires the marker, and the patch goes when the pinned vLLM passes the upstream fix.

### Documentation

- MTP depth: one depth, k=3, for the shipped checkpoint and for the requantized one (user decision, 2026-09-21). Depths 2–5 were measured on the reference pair on ten inputs with tuning and evaluation sets; the depth also changes most completions, so the tables say so. The reference pair, which had served depth 4 with the requantized attention projections since 2026-09-19, serves k=3 from this version ([speculative decoding](docs/speculative-decoding.md#depth-three-for-both-checkpoints-2026-09-21)).
- Requantized checkpoint: the attention projections and `lm_head` repacked to W4A16 NVFP4 (route l) replace the attention-only repack (route g) as the option for Japanese prose. The decode step is 12–13 ms shorter on every input and teacher-forced NLL moves from route g by at most 1.5% (mathematics); the 200K passphrase and the 261,461-token three-position reference are answered correctly. The weights are published on Hugging Face under MIT with NVIDIA's model card beside them ([catalog P23](docs/optimization-catalog.md), [licensing](docs/licensing.md#weight-notices), [server configuration](docs/server-configuration.md)). The template keeps the pinned weights, because the option is not lossless.
- Measured on the full model and not adopted, each with its number in the documents: decode CUDA Graphs (7–9 ms per step slower than eager, `runtime.decode_graphs` stays off), a draft depth chosen from the request's acceptance history, a confidence gate on the draft (its host synchronisation costs about what it saves on two hosts), not sharing the first depth's sparse top-k across draft depths, a rank-local draft argmax, and CSA2 indexer reuse (stopped at its cost gate: the indexer is under 1% of prefill on the fixture and about 4% projected at 200K). None of these settings is in the shipped code.
- Validation: a draft-side change is expected to change completions on this stack (a BF16 tie in the target's logits resolves differently once the drafted candidates change), so such changes are judged by acceptance and NLL, not by equal text.
- Benchmarks (2026-09-22, after the tag): the published option measured on the items the README had listed as not measured on it: prefill of 38,962 tokens, decode after the fixed short prompt, the 255,950-token passphrase, capacity at 262,080 + 64 twice and the three-position reference three times, with the memory minima of both ranks; the README comparison carries the numbers.
- README (2026-09-22, after the tag): reordered for the reader's path (what is deployed, prerequisites, start, what has been verified, objectives, related work). The two headline tables become one comparison of the distributed defaults and the published option, with a pros/cons table and one order for task types; the verified scope becomes a status table grouped by tooling, fixture, full model, harness, template, optional, not adopted and not validated, each row a status and its owner link. Version-to-version asides moved to the benchmarks that own them. The licensing table gains the published repack; the server TOML and LPA paragraphs moved next to what is deployed.
- Document map: task types are listed counting / prose / code and teacher-forced texts Japanese / English / code / mathematics everywhere; the benchmarks role no longer enumerates initiative IDs; the README headline section is named as the one permitted copy of a measured number.
- Benchmarks: the identical-requests sentence lists the prompts in the canonical order.
- README: the headline table carries the measured version in its heading (`tools/check_publication.py` ties it to the newest benchmark section) and points at the benchmark sections that own its numbers; the verified table gains the CSA2 row and describes the published option. The optimization overview's next-candidates list drops the items 1.6.0 and 1.7.0 measured, and its profiles-by-workload table names the code default, the prose option and the batch-prefill option.

## 1.6.2 — 2026-09-20

### Fixed

- Tests: ten host-guard tests added in 1.6.1 renamed the running platform through `os.name`, which pathlib also reads, so they errored on Linux and 1.6.1's CI failed on ubuntu. They now give `server` its own view of the `os` module. The guarded code is unchanged from 1.6.1.

### Documentation

- Operations: distinguish reported GB10 power clamping after an unclean restart from a model slowdown, and record that a fixed KV-byte budget skips memory profiling without validating a larger prefill chunk’s activation peak.
- Validation: require unchanged-arm variation, executed-artifact checks, separate cold/APC measurements, completion hashes and request/token accounting. Reassessed the saved requantization/depth comparisons without changing their measured values or configuration decisions; claims remain scoped to their prompts and available evidence. At the same depth the requantization gain holds per step (+20–22% estimated steps/s on all three prompts).
- Component validation: compared cold-cache FA2 startup on the eight-layer fixture with JIT parallelism unset versus MAX_JOBS=2 / FLASHINFER_NVCC_THREADS=1. Both passed. The run-time JIT build is three translation units, on the fixture and on the reference pair, so the limit can remove one compiler process on the first launch of a new cache; runtime defaults are unchanged.
- README: add the single-Spark 0xSero recipe and separate mosaic checkpoint as comparison sources, with artifact-specific license labels and no code or weights adopted. Keep the headline inference measurements on 1.6.0.
## 1.6.1 — 2026-09-20

### Changed

- The entry layer is restructured without changing behaviour. `server.main`, `cluster.main`, `server_config.validate` and `server_config.serve_args` become sequences of small named steps, the seven fixture runners can state their engine settings without a GPU, and `apc_history` exposes its helpers to tests. Defaults, the generated `serve` arguments, the launch fingerprint and the command line are unchanged; two fixture runners now take the compilation mode as an argument. A comparator found 459 settings, validation and CLI results byte-identical to 1.6.0. On the reference pair, `server plan`, `server preflight` and the `prepare` RPC gave the same results from a 1.6.0 checkout and this one on both ranks, and `apc_history` matched 1.6.0 request for request against a deterministic fake server; the production profile excludes LPA, so that harness was not run against a real server. Tests grow from 325 to 388.

### Documentation

- README: the verified table and its narrative describe 1.6.0, and a new section states what the reference pair's serving profile measures, with its conditions and cost.
- Statements that 1.5.0 and 1.6.0 had overtaken: image-input links no longer say "at 200K", the Japanese optimization overview drops three old default values its English pair never carried, the validation guide says depths 1–5 and decode graphs were measured rather than untested, and the licensing page gains the heading its Japanese pair already had.

## 1.6.0 — 2026-09-20

### Documentation

- Optimization catalog: P23 (requantizing the BF16 attention projections to W4A16 NVFP4) was read twice in this version. The first full-model A/B/A closed it as not adopted: 4.0 GiB less per rank, decode inside the unmodified spread, NLL about 5% higher on the mathematics text. That reading was taken while identical runs still differed by 11%. Measured again on the baseline where they repeat, decode rose 22 to 29% with the acceptance length unchanged and NLL 4 to 6% higher on three of four texts, and the reference pair has served it in trial use since 2026-09-19; the template keeps the pinned checkpoint. Adding the shared experts to the repacked set was measured and not adopted. P05 (FA2 prefill, adopted) and P06 (decode graphs, kept as an option that is off) carry their 1.6.0 outcomes too.
- README: a table of the other public GLM-5.3-Flash recipes for DGX Spark pairs, with their links, licenses and what this repository took from each. That table now owns those facts: the inline citations in the component validation, operations, optimization catalog, server configuration and speculative decoding documents keep the name and pull request and no longer restate a license, and `tools/check_publication.py` rejects a recipe link or a restated license under `docs/`.
- README: a headline table of the measurements on the current defaults (prefill, decode, 200K and 256K requests, the unstable three-position reference and the lowest available memory). `docs/benchmarks.md` still owns the numbers.

### Added

- **`runtime.fa2_attention`** (optional, false when absent). Prefill-sized calls of the candidate-preserving NoPE attention go through FlashInfer's MLA paged wrapper (`fa2`, page size one, candidates as KV pages) over rows unpacked to BF16; decode steps stay on the reference path. On the reference pair prefill of 38,962 tokens rose from 572–577 to 1,242–1,272 tok/s and identical requests still repeat bit for bit. It excludes LPA. The measurements are in `docs/server-configuration.md`.
- **`runtime.stable_indexer_topk`** (optional, template `true`; absent = the image's default). The kpool indexer's top-k kernels in the pinned vLLM (`persistent_topk` for decode, `top_k_per_row_prefill`) return a different set of pools for the same input when pools tie across the 512th rank, and one such tie forks a completion at a fixed place, now and then. The reference image patches both calls to go through `glm53_setup/runtime/stable_topk.py`: decode is a stable sort, prefill keeps the kernel and re-selects only tied rows, so a tie always goes to the lower pool index. `server preflight` asks the image for `GLM53_INDEXER_TOPK_API=1`; `false` is the comparison arm. The measurements are in `docs/server-configuration.md`.
- **`runtime.canonical_moe_order`** (optional, template `true`; absent = the image's default). The reference image now patches the pinned Marlin MoE path to sort the tokens inside each expert by token id before the kernel (`glm53_setup/runtime/moe_token_order.py`, source-pinned `patch_moe_order`), and sets `GLM53_CANONICAL_MOE_ORDER=1` and the marker `GLM53_MOE_ORDER_API` (`2` from this version; preflight accepts `1` too, so a switch can still prepare a running older image as its recovery target). Identical requests then repeat bit for bit on the eight-layer fixture; decode is about 1% slower there and prefill unchanged. `server preflight` refuses `true` on an image without the marker; profiles without the key keep their fingerprint. **Rebuild the reference image to get it.** On the reference pair identical requests then repeat bit for bit, decode is unchanged (31.1 against 31.0 tok/s median) and the MTP acceptance length rises (3.09 to 3.35).
- **`server agreement`.** Teacher-forced agreement with a saved reference run: four self-authored texts through `prompt_logprobs`, the rank and log-probability of each actual next token, and with `--reference` the argmax agreement, top-5 overlap and drift against an earlier record. When the command was written a served full model did not repeat identical requests exactly (argmax agreement about 0.96 on the reference pair), so a run passes when every request was answered in the checked shape and the repeat difference is reported as the yardstick. The two causes were found within this version and are fixed by `canonical_moe_order` and `stable_indexer_topk` above.
- **Requantization checks on the four-layer fixture**: `quant-error` (per-tensor weight-space error of NVFP4 tensors against their BF16 originals, byte identity of everything else), `agreement-fixture` (top-5 rows, full-vocabulary log-probabilities and layer-3 candidate sets) and `agreement-compare` (full-vocabulary KL, argmax agreement, candidate-set Jaccard). [Validation](docs/validation.md) records how far the unmodified fixture moves between two starts, the yardstick for any such comparison.
- **`runtime.derived_checkpoint`** (optional, absent by default, fingerprints of existing profiles unchanged): serve a locally requantized copy of the pinned checkpoint with the source overlays it needs. `server preflight` checks the checkpoint's declared target, that the MTP draft layer stays unquantized, and each overlay's hash, marker and the hash of the image file it replaces. The reference pair used it for the P23 A/B/A and has served it in trial use since 2026-09-19.
- **`validation.run_repeat_trace`** repeats a request inside one process and names the first module whose output differs from the first pass. On the eight-layer fixture that is always the routed experts: the token order inside each expert changes from call to call and the Marlin MoE result depends on it. `--canonical-align` fixes that order as a diagnostic and makes every pass bit-identical; `--zero-moe-buffers` rules out stale scratch memory. `--verify-canonical` shows the fix changes results only at that noise level, and `--timing` measured about 1% slower decode and unchanged prefill on the fixture. Upstream tracks the cause as vLLM issue #52525.
- **`fixture-build --with-mtp`** keeps the BF16 draft layer in the four-layer fixture, renumbered to follow the kept layers and excluded from the global NVFP4 setting in both quantization files, so MTP can be exercised on one GPU.
- **Releases follow tags.** Pushing a `vX.Y.Z` tag runs `.github/workflows/release.yml`, which publishes that version's Changelog section as the GitHub Release through `tools/release_notes.py`. Tags 1.2.0 to 1.5.0 had been pushed without a Release; those were created by hand, and 1.2.1, which had a Changelog section but no tag, was tagged at its release commit.

### Fixed

- **`cluster resume` can confirm a pair that has been serving for a while.** The head's readiness poll looked for the startup line in the last 200 log lines only; the supervisor's `/metrics` reads push it out of that window within minutes, so a resume after a lost observation never saw the head as ready. The poll reads the whole log when the tail lacks the line.
- **The fused FP8 unpack kernel no longer compiles one kernel per input size.** Its element count was a `tl.constexpr`, so every distinct size compiled and kept its own Triton kernel. The reference attention always unpacks the same few sizes, so nothing showed; with `runtime.fa2_attention` the number of distinct cache rows differs on every call, and each serving worker gained about 0.2 MiB of heap per call (about 100 MiB per 100K-token prefill, 0.15 to 0.4 GiB per 256K request, never returned) while `~/.cache/triton` gained one kernel directory per call. The count is a run-time argument now; results are bit-identical.

### Changed

- **The template turns `runtime.fa2_attention` on.** A profile written before the key keeps the reference path and its fingerprint. The FA2 path excludes LPA, so turning `lpa.enabled` on now needs `fa2_attention = false` beside it. The template keeps MTP depth 3: with the pinned checkpoint depth 4 gains 0 to 3% on predictable text and loses 7% on prose, and it pays with `runtime.derived_checkpoint` (`docs/speculative-decoding.md`).
- **`cluster resume` finishes what the interrupted switch did not.** A pair it finds ready now gets the warmup ladder and, with `--config`, the profile text on both ranks, recorded like a switch's and never rolling the pair back; a recovered old pair gets neither. Until now a resumed pair was complete without a ladder and with whatever file the ranks had.
- **`cluster switch` replays `start` and `stop` over a dropped SSH connection** (at most three attempts, like the read-only observations). `stop` was already idempotent; `start` now answers `replayed` for an attempt in flight instead of refusing, and still refuses a finished or cancelled one. A switch has stopped both old ranks by the time it starts the new ones, so a single transport failure there failed the switch and spent a recovery. Reserve, install and warmup are still never replayed.
- **The reference image carries the FA2 path and says so (`GLM53_FA2_ATTENTION_API=1`).** An FA2 profile still bind-mounts the path, its dispatch and the fixed fused unpack from the checkout, redundant on such an image, so that an image built before them stays a valid recovery target of a switch.
- **`cluster switch` writes the profile file on both ranks.** Until now the switch carried the settings in the frozen manifest only and left the `--remote-config` file as it was, so a rank-side command reading that file could see a profile other than the running one. Once the new pair is complete, each rank now writes the `--config` text to that path: it accepts only text that parses to the profile it was started with, keeps a differing old file as `<name>.bak-<UTC time>-<fingerprint prefix>` and replaces the file in one rename. A failed or recovered switch writes nothing; a failed write is recorded under `config` in the journal and never rolls the pair back. `--no-send-config` keeps the earlier behaviour. Both checkouts need this version.
- **The supervisor's `resources.jsonl` records the container's cgroup memory and its processes' RSS** beside `MemAvailable` every 2 seconds, so a `memory-reserve` stop can be read afterwards: flat cgroup and RSS with falling `MemAvailable` is device-side growth on GB10's shared memory, rising RSS is process growth.
- **`validation.memory_probe` traces a request inside the serving worker.** `trace_begin`/`trace_end` fingerprint what the traced modules, the draft model and the sparse NoPE attention take and return, keep one request as the reference and name the first call in execution order that differs in a later identical request; `host_census` counts live Python objects; `fa2_stage` runs part of the FA2 path and serves the rest from the reference path. The trace gathers touched cache rows for decode-sized calls only (for a prefill chunk that gather is gigabytes per MLA layer on memory the GPU shares with the host). [Server configuration](docs/server-configuration.md) describes the methods.
- **`validation.memory_probe` reads the host side too.** `host_stats` reports a worker's resident anonymous memory, glibc's `mallinfo2` split (bytes in use, free bytes it keeps, mapped blocks) and torch's pinned host cache; with `trim` it calls `malloc_trim(0)` between two readings, so retained free memory, live objects and memory outside malloc can be told apart when a worker's own memory grows.
- **`validation.memory_probe`** (optional, default `false`): a worker extension whose `allocator_stats` method, reached through the dev `/collective_rpc` route, reports the torch caching allocator per rank (reserved, allocated, segments, retries, `mem_get_info`), for telling allocator growth from other host-memory growth on GB10's shared memory.
- **`runtime.derived_checkpoint.enabled`** (optional, default `true`): `false` keeps the table in the profile and serves the pinned snapshot, so the requantized checkpoint is a one-key switch like the other measures.
- **`runtime.decode_graphs`** (optional, template `false`) is the positive one-stop switch for decode Graphs; `runtime.enforce_eager` is still read as the earlier spelling and existing profiles keep their fingerprint.
- **Decode Graphs may be combined with MTP and prefix caching for one sequence.** `runtime.enforce_eager = false` had been refused whenever MTP or prefix caching was on, and its capture size was fixed at `[1]`, which the pinned runtime rejects outright with MTP (decode sizes are rounded up to a multiple of `num_speculative_tokens + 1`). The launcher now passes `[num_speculative_tokens + 1]` with MTP on and `[1]` otherwise, and still refuses `max_num_seqs > 1`, LPA and synchronous index checks with Graphs. Ground: on the four-layer MTP fixture with the expert token order fixed, eager and Graph runs are identical in tokens and logprobs at every length, with and without prefix caching (component validation). Full-model speed and acceptance are not claimed.
- `validation.run_graph_fixture` takes `--apc` (prefix caching on), `--lengths` and `--seqs` (a batch of distinct prompts), and records each sample's cached tokens.
- `mtp.num_speculative_tokens` accepts 1 to 5 instead of only the measured 1 and 3, so a depth sweep can be launched from the profile; the template keeps 3.
- `tools/check_publication.py` fails when the version in the README headline table, in either language, is not the newest `Measurements on X.Y.Z` section of the benchmark document.
- Private plans live under `docs/plans/`, which is now the ignored path; the audit had been counting them as public files.

## 1.5.0 — 2026-09-18

### Changed

- **The template serves image input at 256K**: `context.max_model_len = 262144` (was 204800), `cache.kv_cache_memory_bytes = 3221225472` (3 GiB per rank, was 2.5) and `resources.reserve_gib = 3.0` (was 2.5). Eight NCCL channels in 1.3.1 had left the head at 6.40 GiB or more during 200K requests, so 0.5 GiB more KV was expected to leave about 5.9 GiB against the reserve plus the supervisor's 1.6 GiB overshoot. Measured on the reference pair: 5.89 GiB during startup and 5.82 GiB during a 255,950-token request on the head, 8.44 GiB on the peer. The pool holds 301,645 tokens (84 blocks, 73 per full-length request, 1.15×). A 262,080 + 64 request took 488.6 s and the longest 256K request 492.5 s, within the 600 s timeout; prefill ran at 569.8 tok/s, and the image, tool and video checks passed. At 200K on the new profile speed matched 1.4.0 with 0.5–0.7 GiB less free memory on each rank. Profiles without the change keep 200K and their fingerprint. A 300K profile with 4 GiB KV was estimated first and not run: the same arithmetic left 0.3 GiB of margin and a full-length request near 570 s. The text-only alternative stays, now at the same length and KV.

### Documentation

- Benchmarks: measurements on 1.5.0 (startup pool and memory, prefill and decode, 256K passphrase, capacity and three-position requests, and sparkDash and the 200K requests repeated against 1.4.0). The three-position reference answered correctly in 2 of 3 runs at 256K and failed again at 200K after the capacity request, the same misreading as on 1.4.0.
- Server configuration: how the KV pool counted blocks in both measured image-profile configurations (28 blocks per GiB, ceil(L / 4608) + 16 per full-length request), which predicted the 256K pool before the switch; the text-only alternative is described as the default without the vision tower and its 3 GiB reserve as unvalidated.
- Image input: the 256K settings, why they were chosen and the checks on the new profile. README, operations, harnesses, LPA and the optimization catalog and overview follow the 256K defaults.
- Harnesses: the stream idle timeout paragraph gave the tested configuration as both 60000 and 700000 ms; it is 700000.

## 1.4.0 — 2026-09-17

### Added

- **Foreign-GPU launch guard.** `server preflight`, `server start` and `server assets` add `exclusive_gpu`, which fails while a running container without this launcher's label requests a GPU, and list those containers under `foreign_gpu_containers`. On the reference hosts both `--gpus` and CDI requests appear in `HostConfig.DeviceRequests`, and the GPU device nodes never appear under `Devices`. A pair of this launcher carries the label whatever its fingerprint, so a running old pair does not block `cluster switch`; an empty label value does not count, and a container that cannot be inspected while still listed fails the check. There is no override and no new profile key. Informed by sfxnz PR #12 (MIT, no code adopted).
- **Memory readings beside `MemAvailable`.** Each supervisor line records `mem_free_gib` and `free_2mib_gib`, the free memory in buddy blocks of 2 MiB or more. NVRM needs such blocks and does not reclaim the page cache; the head showed 7.1 GiB available with 0.49 GiB in them while serving. Only `MemAvailable` stops a rank, and an unreadable sample is recorded as `memory_sample_error`.
- **`server mojibake`.** Long Japanese and Korean answers at temperature 0, three each, with replacement characters, lone surrogates and control characters counted in the answer and the reasoning. Short, off-language, empty, malformed or failed answers are inconclusive. All six passed on 1.3.1; the reasoning was empty at low effort and was not exercised. The pinned NVIDIA checkpoint is not affected by the gate/up scale mismatch of vLLM #54150.
- **Attention component probes.** `validation.benchmark_sm90_attention` calls FlashInfer's MLA paged wrapper as the pinned SM90 backend does, with `fa2`, on GB10. It passed every numerical check at full candidate width and ran 512 rows in 3.5 ms against 70.6 ms for the reference, with BF16 KV only: FlashInfer 0.6.18 rejects FP8 MLA KV off SM90. `validation.benchmark_native_attention --by-row-kind` revisits the SM120 probe: its 3.03125 came from empty rows, which zeroing fixes, but rows with 17 candidates still exceed the bound. Neither changes serving. Informed by sfxnz, tonyd2wild, drowzeys and Mia (no code adopted).

### Changed

- **The template sets `context.max_num_batched_tokens = 2048`** (was 512). On the 200K image profile with one sequence, a fresh 39K prompt prefilled at 476.9 / 545.2 / 563.0 tok/s at 512 / 1024 / 2048, and the 199,652-token passphrase request took 361.3 s at 2048 against 410.8 s at 512, correct. The peer's lowest free memory fell by up to 0.7 GiB (head 6.71, peer 8.77 GiB during measurement against a 2.5 GiB reserve), loading at 2048 logged 28 NVRM allocation retries on the peer, and decode did not change beyond run-to-run variation. Profiles without the change keep 512 and their fingerprint. With two sequences a longer chunk lengthens the longest pause, as P11 measured at 1024. The 200K three-position reference answered correctly in 1 of 3 runs at 2048 and in 1 of 2 at 512 on the same stack: at both sizes the model sometimes reads each value as the article that follows it, so that check does not separate the two.

### Documentation

- Component validation: the SM120 zero-padding revisit by row kind and the new SM90 FA2 wrapper probe; the catalog's P05 row and the overview follow.
- Operations, launch contracts, server configuration and setup: the foreign-GPU guard and when it runs, what the new memory readings mean, the page-cache effect of checksum runs and large reads on unified memory, `server mojibake`, the two-sequence scope against another recipe's long-request collapse, and vLLM #53366 for a future compiled-Graph plus MTP combination.
- Validation, harnesses and speculative decoding: the multibyte-output check and why the pinned checkpoint is unaffected, the XGrammar and speculative-decoding fixes already in the pinned vLLM, and comparing depths by mean acceptance length.
- Benchmarks: the chunk-budget comparison on the 200K image profile.
- Operations and the catalog's P10: `vm.swappiness` 0 against 60 at chunk 2048. At 60 no engine process had swapped pages; at 0 swap stayed empty, prefill moved within restart variation and the head kept 0.45 GiB less free, so the hosts stay at 60.
- Benchmarks: sparkDash and the 200K checks on 1.4.0 (chunk 2048). Decode medians 36.51 / 25.67 / 29.31 / 26.29 token/s, matching 1.3.1; the passphrase request took 361.4 s and the capacity request 380.0 s, 12% and 19% less than on 1.3.1, with the head never below 6.40 GiB free. The three-position runs are listed with their reasoning lengths and the request that preceded each.
- Operations: how the 2026-09-16 host-memory cause was traced and what held after the fix. Bluetooth and the display manager's greeter were symptoms; counting new process IDs (6,642 against 176 per 10 s), login-message script runs and SSH logs led to the dashboard's per-metric logins. After a power cycle with Bluetooth enabled again and the dashboard watching two hosts, `polkitd`, `bluetoothd` and `wireplumber` stayed the same size over 300 s, and a dashboard restart on 2026-09-17 made no new logins.
- Operations and the template: the timeout note cited 200K at 506 s as the longest measured request; it now cites 380 s at 200K with chunk 2048 and 582 s for the 256K alternative. Server configuration still gave the 4.0–4.2 GiB head headroom measured with 64 NCCL channels as current; it now adds 6.97 GiB with 8 channels and 6.40 GiB at chunk 2048. A link to enabling LPA still pointed at the heading renamed in 1.1.0, and four English/Japanese pairs carried a link or clause on one side only.
- Benchmarks: the release candidate's sparkDash and 200K checks repeated on 1.3.1 (LPA off, 8 NCCL channels, MTU 1500). Decode medians 36.67 / 25.71 / 30.19 / 26.48 token/s for structured / prose / code / json, TTFT shorter for all four. The 199,652-token passphrase request took 410.8 s against 506.1 s on 2026-09-15, capacity 204,736 + 64 took 470.3 s and the three-position reference 435.1 s, all correct, with the head never below 6.97 GiB free. Several changes lie between the runs, so no single one is credited.
- NCCL validation and QSFP network: the MTU 1500 full-model runs had the head's monitoring dashboard running (65 MiB, about 2% CPU) while the MTU 9000 runs did not. 1.3.1 said every run had it stopped. The 2.3–2.6% prefill gain from MTU 9000 is therefore an upper bound. The channel-count comparison at each MTU is unaffected, and so is the memory cost, which the dashboard can only understate.

## 1.3.1 — 2026-09-17

### Changed

- **The template sets `runtime.nccl_channels = 8`.** On the reference pair NCCL alone opens 64 channels, and each engine opens two communicators. At 8 the full model's lowest free memory rose by 2.8 GiB on the head and 3.0 GiB on the peer at MTU 1500, and prefill was 1% faster. A two-rank AllReduce sweep over 64/32/16/8/4 channels at both MTUs put 8 at the memory and 4 MiB (prefill-chunk) optimum. Profiles without the key keep NCCL's choice and their fingerprint. Numbers are in the new [channel-count section](docs/nccl-validation.md#channel-count).

### Documentation

- **MTU:** 9000 bought 2.3–2.6% prefill for 1.1–1.7 GiB less free memory per host (NIC receive buffers; an idle host showed the same 1.4 GiB), so the reference pair stays at 1500 (QSFP network, NCCL validation).
- **Warmup ladder:** its ten kernels are reported on every start, and they are loads, not compiles. The pinned Triton calls its post-compile hook on a kernel's first use in a process even when the binary comes from the on-disk cache, and the jit monitor warns from that hook. Across five starts the runtime cache gained no file. Operations and image input no longer say that the ladder turns empty once the cache is warm.

## 1.3.0 — 2026-09-17

### Added

- **`runtime.nccl_channels`** (optional; absent = NCCL chooses). Sets `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` to the same value on both ranks. NCCL 2.30.7 chooses 64 channels on both ranks of the reference pair, recorded identically on 2026-09-11 and 2026-09-17, so this is not a consistency fix: it lets a measurement test whether fewer channels return memory to a head that runs a few GiB above its reserve. The template sets no value until one is measured, so running pairs keep their profile fingerprint; a value takes effect at the next switch. Informed by Mia PR #200 (the knob only; its TP=2 launcher leaves the value empty as well).

### Documentation

- Operations: an idle half-dead pair is stopped by neither supervisor, a host lockup takes its own supervisor with it, and a lost-peer stop reason remains a design note. From Mia issue #193, which reports the mirror image of the 2026-09-16 half-dead pair.

## 1.2.1 — 2026-09-17

Everything here follows from running 1.2.0 on the reference pair for the first time; `docs/plans/MIA_ADOPTION_2_PLAN.md` Stage 6 owns the run.

- **Long warmup rung converges on its target.** The first live ladder built 82,006 tokens for a 65,536 target: a line costs more once its index grows a digit, so a 256-line sample under-counts. The builder now rescales against the measured count.
- **Host daemon hygiene** in operations, from a half-dead pair: the peer rank stopped at 2.49 GiB against a 2.5 GiB reserve because a monitoring dashboard on the other host opened a new SSH login per metric, 3.6 per second, and every login grew `polkitd` (to 3.40 GiB), made `wireplumber` and `bluetoothd` re-register and grow, and ran every MOTD script. SSH connection reuse on the dashboard's side ended it. How to count logins and compare daemon sizes, why a `MemoryMax` drop-in also needs `Restart=on-failure`, and that a half-dead pair still answers `/health` with 200.
- **Measured, now documented.** During an 82,018-token prefill the two token counters froze for 202.8 s while the four progress signals together never froze for more than 8.3 s, which is the case for keeping KV usage in the set. Prefix caching works in whole blocks: a repeated N-token prompt restores `(floor(N / block) - 1) x block` and nothing below two blocks, measured as 0 / 9,216 / 23,040 tokens at 3,625 / 14,025 / 28,025 on the 4,608-token block.
- **P24 filed**: per-request prefix-cache no-store. Against an 80,024-token conversation warm at 73,728 cached tokens and 14.9 s per turn: two untouched 15,025-token lanes left it intact, four evicted it completely (next turn 182.6 s), and one 42,026-token lane evicted it on its own (184.0 s). The flag would fix the first case and not the second, because it does not reduce what a running lane allocates. Candidate only; it needs a runtime overlay and an image rebuild.

## 1.2.0 — 2026-09-16

### Added

- **Post-readiness warmup ladder** (`server warmup`; `generation.warmup`, `generation.warmup_long_tokens`). Sends the request shapes that were observed compiling kernels while serving (short text, tool call, one image, an optional long prompt sized with the served tokenizer), records per-rung timing and the kernels the jit monitor reported before and during the ladder, and resets the prefix cache afterwards when the dev routes are mounted. `cluster switch` runs it on rank 0 after both ranks are ready and stores the result under `warmup`; a ladder failure never fails or rolls back the switch. Informed by Mia PR #170; rungs come from this kit's own records, not that PR's.
- **Engine stall detection** (`resources.stall_seconds`, template 600, rank 0 only). The supervisor reads `/metrics` in its existing 2 s cycle and stops with `stop-reason: engine-stall` when requests are running but the generation-token count, prompt-token count, KV-cache usage and running count have all been frozen for that long. `/health` stays 200 while the engine is wedged, so it was never a liveness signal. Prefill progress counts as activity; an unreachable `/metrics` never counts. Operations now also carries the swap-cycling note. Informed by Mia PR #70.
- **`server capacity`.** Reads the running head's boot log and `/metrics` and prints the KV pool as it is: the stock `GPU KV cache size` line decomposed into blocks, blocks per maximum-length request and group block widths, stating that the stock figure is `max_concurrency × max_model_len`. A cached-conversation estimate is printed only for a profile that loads the LPA worker extension (whose RPC names the group spec kinds) and is withheld for any group kind it does not model. The vision document's "235,016 tokens of KV capacity" wording was that misreading and is corrected. Informed by Mia PR #94; no runtime patch.
- **`api.dev_endpoints`** (optional, default false). Mounts vLLM's dev routes (`/reset_prefix_cache`, `/collective_rpc`, `/sleep`, …) on the loopback API for a distributed profile, so benchmarks and the warmup cleanup can reset the prefix cache without a restart. The routes are unauthenticated, as launch contracts already state. Informed by Mia PR #37.
- **P23 filed in the catalog**: FP8 weight-only for the projections the NVFP4 checkpoint leaves in BF16 (`self_attn*` 11.29 GiB, `shared_experts*` 1.97 GiB, `lm_head` 1.18 GiB across both ranks, read from the safetensors headers). Candidate only, quality-gated, not started; adaptive verification length stays rejected. The P06 resumption criteria gain the spec-acceptance-pinned-at-1.00 check (vllm#53030).

### Changed

- All new profile keys are optional; existing `state/server.toml` files and running pairs stay valid and recoverable. `supervise` now takes the rank. The SSH transport allows the readiness timeout for the warmup call because the long rung pays a full prefill.

## 1.1.1 — 2026-09-15

- README (both languages): distributing the source also carries the copyright and license notices of the adapted third-party code (MIT and Apache), as the licensing guide and third-party notices already state; the sentence that said "only Apache-2.0" contradicted the sentence after it.
- Include `examples/server.example.toml` in the Docker build context; `.dockerignore` still whitelisted the pre-1.1.0 name, so the reference image would not have built.
- Operations: `server preflight` prints its checks and only `server start` writes them under `records/`; the 1.1.0 text said preflight recorded them. Document the runtime state links for source-archive deployments with `ln -sT` and the `readlink -f` output that distinguishes a correct link from a nested `state/state`.

## 1.1.0 — 2026-09-15

### Changed

- **One launcher.** Remove the `service` command and its `state/site.json` / `state/kernel-validation.json` receipt gate. It was the initial fixed-settings launcher: `service start` still selected the lock's base image, whose native GB10 path is blocked, and required a qualification receipt that no workflow produced, so it never started a full model. Every measurement, the memory supervision and the two-rank switch/recovery live in the other path, which is now the only one. Whether a profile is experimental or accepted for routine use is shown by the recorded status (README status table, harness acceptance matrix), not by a command name. `examples/site.example.json` is removed; the same values live in `[nodes]` of the server TOML. The shared host helpers (site validation, serve arguments, fabric checks, snapshot resolution, subprocess execution) move to `glm53_setup/host.py`.
- **`startup` becomes `server`.** The subcommand is `python -m glm53_setup server {plan,freeze,assets,preflight,start,stop,status,ask}`, the profile is `state/server.toml` (template `examples/server.example.toml`), the modules are `server.py` / `server_config.py`, and the document is `docs/server-configuration.md` (both languages). "startup start" named the profile and the action with the same word. No alias for the old spelling is kept. **Operators copy `state/startup.toml` to `state/server.toml` on each host** (the content is unchanged, so the profile fingerprint does not change) and update both checkouts before the next switch. Keep the old file until the first 1.1.0 pair is confirmed ready: a running pair records the profile path it started from, and switch recovery relaunches the old profile from that path ([launch safety](docs/launch-safety.md#all-rail-checks-and-two-rank-switch)).
- **`--experimental` is gone** from `server start` and `cluster switch`. The flag existed to mark the path that was not the routine `service` launcher; with one launcher it carried no information.
- Stable runtime identifiers are unchanged, as in the BIZ rename: the container label `glm53.experiment.startup`, container names `glm53-startup-r<rank>-<run>`, the per-rank records `state/startup-rank<N>.json` and `state/startup-request.lock`. A pair started by 1.0.x is therefore still recognized by `server status`, `server stop` and the switch.
- Documentation: operations now has one "One launcher" section and a "Full-model launch checks" section that says what `server preflight` covers and what it does not certify; SETUP step 6 lists the open routine-use acceptance items instead of a receipt that could not be produced; the README, document map, architecture table, launch-safety, validation, speculative-decoding, harness and LPA pages drop the two-launcher framing. Nothing became accepted by this change.

## 1.0.1 — 2026-09-15

- Add an explicit "Enable LPA in the startup profile" section to the LPA document (both languages): the TOML keys to set, the image markers preflight requires, the projector checksum check, the text-only condition, the preflight-then-switch order and the per-request opt-out; the document map and operations table point at it. Change the `lpa-train` reproduction example from the cut 40 pilot to `--cut 32`, the setting of the distributed projector.
- Distribute the measured cut32 auxiliary projector (Release `lpa-cut32-v1`) as a separate GitHub Release asset under Apache-2.0, with a pinned manifest, training-data attribution, teacher model card/notices and SHA-256 checksums. Preserve the original tensor bytes and keep weights outside Git.
- Update both languages of the README, operations package layout, architecture, LPA download/model card, licensing, startup prerequisites and document map. Runtime defaults and qualification status are unchanged; LPA remains a text-only batch opt-in.
- Fix the CLI smoke test to compare the version with `pyproject.toml` instead of the retired beta literal, and require a successful exit.

## 1.0.0 — 2026-09-15

### Changed

- Release 1.0.0 and retire the beta label: the version lives in `pyproject.toml`, the publication audit now requires a plain semantic version instead of a beta disclosure, and every README, runbook and document heading that said "beta" now states the same facts without it. The `service` launcher stays gated pending a kernel-validation receipt; nothing about qualification changed.
- Record the first full pass over the shared harness cases H-01–H-11 (`20260915-harness-h-sandbox`): all eleven ran once on the npm `zcode-app-cli` 3.11.2-24 TUI against a fixture repository, 5 PASS and 6 PARTIAL with the untested part named per case; official ZCode Desktop and Claude Code remain NOT RUN. The README status row and the acceptance matrix carry the per-case result.
- Raise the template's `generation.max_tokens` from 512 to 4096. The value only shapes `startup ask` requests, but the fixed GLM template always thinks first and vLLM counts that block against `max_tokens`: at `reasoning_effort=low` a one-line answer spent about 600 tokens, so the 512 default returned `finish_reason: length` with no visible content. Existing `state/startup.toml` files are not changed; editing one changes the profile fingerprint, so a running pair reports `settings_changed` and `startup ask` refuses it until the next switch.

- Documentation refactor for single ownership and freshness, in both languages: recast stale default-value claims in the measurement documents (APC, fused unpack, `index_checks`, the 8 GiB reserve) as dated history that points at the startup configuration; add the `startup`/`service` launcher distinction as one owner section in operations and a document-map row; extend the architecture table to the runtime, cluster, validation, tools and hook modules that exist; split the README prefix-caching status row; fix the stranded `Lifetime` row in the startup defaults table, four Japanese pages whose language switch linked to themselves, Japanese pages that linked English counterparts, and spaces dropped before numbers in benchmark prose; merge the duplicated `Changed` section of this file. Inline the benchmark and NCCL command blocks in the Japanese documents so each language reads on its own. No measured value or run condition changed, and no existing heading was renamed.

- Write every status and result JSON (`prepare-status.json`, `checksum-status.json`, `preflight.json`, `command.json`, fixture and benchmark `result.json`) through the shared atomic writer, so a crash mid-write leaves the previous file instead of a truncated one; each file now ends with a newline. The image-capability preflight table, the host request lock and the rank-0 lookup are single helpers in `startup.py` instead of copies in each validation command; a missing image `Env` now fails the reference-attention check closed rather than raising.

- Switch the distributed startup defaults to image input at 200K: `runtime.vision = true`, `max_model_len` 204,800, KV 2.5 GiB per rank, `mm_processor_cache_gb` 0.1 and `reserve_gib` 2.5, with video still rejected. The 256K text-only profile stays documented as the alternative; a 2.5 GiB reserve is not validated for it. Add `docs/vision.md` (bilingual) with the settings, how they were chosen (quantization exclusion, video profiling, head-only processes, the JIT cache incident, the reserve) and the measurements, and update the README, setup runbook, startup configuration, harness, benchmark, optimization and document-map references. The startup defaults table now also states that LPA ships disabled. Existing `state/startup.toml` files are not changed.
- Rewrite the harness document as the single source for connection design, the acceptance matrix and per-case status: add a status column (API-01–03 PASS, API-04 PARTIAL, official ZCode Desktop NOT RUN / bundled CLI BLOCKED on the missing TUI package, npm distribution smoke as supporting evidence, Claude Code and shared H cases NOT RUN), separate the ZCode distributions, replace the hedged telemetry text with a per-distribution table from a static read of the bundles (npm CLI sends traces only with an OTLP endpoint set; Desktop hardcodes trace and event endpoints and strips inherited switches), align the example port with the startup template, and add a rule to deny harness writes to its own config directory. README and validation now point to the matrix instead of restating status. Record a dated TCP sample for the npm distribution (only the loopback tunnel observed during one headless prompt) as supporting evidence under H-09.
- Rename the distribution from “GLM-5.3-Flash on 2× DGX Spark Enterprise Setup” to **GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ**. BIZ marks the maintainer and a business-use intent; the word “enterprise” was read as implying support or certification, so prose now says “business use”. Stable identifiers are unchanged: the reference image tag `glm53-enterprise:reference` (baked into existing records) and catalog IDs E01–E03. The Python distribution name becomes `glm53-flash-biz-setup`.
- Align distributed startup defaults with the serial optimized profile: 256K context, 3 GiB KV per rank, MTP3, LPA cut32/tail512/B128, APC with dense checkpoint retention, fused unpack and asynchronous index checks. Disable the lifetime deadline by default (`run_seconds=0`) while retaining memory supervision with a host memory reserve (lowered to 3 GiB by a later entry below). Preserve omitted/numeric retention settings even when the template explicitly selects dense retention. Image IDs, projector/hash and node details remain installation-specific placeholders; existing operator TOMLs require explicit migration and running supervisors require restart.
- Point the Triton, TileLang and TorchInductor caches of distributed startup containers into the mounted `state/tp2-runtime-cache` (`TRITON_CACHE_DIR`, `TILELANG_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`). Their defaults (`/root/.triton`, `/root/.tilelang`, `/tmp/torchinductor_root`) lived in the container layer, so every start recompiled about 2,000 entries, some of them while serving; one such compile burst on the head coincided with a memory-reserve stop at the 200K vision profile.
- Accept a fractional `resources.reserve_gib` (the template writes `3.0`; integer values stay valid) and replace the "measure again" note with what the 2-second supervisor can catch: about 9 seconds from breach to container exit, the measured 0.15 GiB/s slide, no recorded OOM kill with 16 GiB of swap, and driver `NV_ERR_NO_MEMORY` messages that do not mark the floor. A later entry sets the template value to 2.5 with the image profile.
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

### Added

- Record one image read inside a ZCode terminal session on the image profile: the model saved a screenshot with a shell command, read it with the file-read tool and described text present only in the image. Direct image attachment in a harness prompt and Claude Code remain unchecked (`docs/vision.md`, README).

- Host kernel guidance for new and updated hosts (`docs/operations.md`, bilingual): current DGX OS updates install `7.0.0-1019-nvidia`, whose default-enabled Kexec HandOver can make two-host RoCE registration fail with `ibv_reg_mr_iova2 ... Cannot allocate memory`. Record NVIDIA's advisory, the open NV-Kernels analysis, the two choices (hold `6.17.0-1032-nvidia`, or boot with `kho=off` and verify it) and a separate firmware-related report with the same error. The setup runbook now records `/proc/cmdline` and makes the kernel choice before the two-host steps; the README prerequisites point there. `7.0.0-1019-nvidia` is not validated by this repository.

- Optional `runtime.vision` startup key (false when absent, explicit in the template). `true` drops `--language-model-only` from both ranks so the vision tower loads, and adds `--limit-mm-per-prompt '{"video": 0}'`: **video input stays disabled**, because startup profiling would otherwise encode a 30,000-token video. It also sets `--mm-processor-cache-gb` from the optional `cache.mm_processor_cache_gb` (0.1 when absent) instead of vLLM's 4 GiB, which is held twice on the head. A later entry makes it the distributed default.

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
