# Changelog

## Unreleased

### Documentation

- Optimization catalog: P23 (requantizing the BF16 attention projections to W4A16 NVFP4) closed as not adopted after one full-model A/B/A: 4.0 GiB less per rank, decode inside the unmodified spread because the MTP acceptance length fell from 2.9 to 2.37, NLL about 5% higher on the mathematics text.
- README: a table of the other public GLM-5.3-Flash recipes for DGX Spark pairs, with their links, licenses and what this repository took from each. That table now owns those facts: the inline citations in the component validation, operations, optimization catalog, server configuration and speculative decoding documents keep the name and pull request and no longer restate a license, and `tools/check_publication.py` rejects a recipe link or a restated license under `docs/`.
- README: a headline table of the measurements on the current defaults (prefill, decode, 200K and 256K requests, the unstable three-position reference and the lowest available memory). `docs/benchmarks.md` still owns the numbers.

### Added

- **`runtime.canonical_moe_order`** (optional, template `true`; absent = the image's default). The reference image now patches the pinned Marlin MoE path to sort the tokens inside each expert by token id before the kernel (`glm53_setup/runtime/moe_token_order.py`, source-pinned `patch_moe_order`), and sets `GLM53_CANONICAL_MOE_ORDER=1` and the marker `GLM53_MOE_ORDER_API=1`. Identical requests then repeat bit for bit on the eight-layer fixture; decode is about 1% slower there and prefill unchanged. `server preflight` refuses `true` on an image without the marker; profiles without the key keep their fingerprint. **Rebuild the reference image to get it.** On the reference pair identical requests then repeat bit for bit, decode is unchanged (31.1 against 31.0 tok/s median) and the MTP acceptance length rises (3.09 to 3.35).
- **`server agreement`.** Teacher-forced agreement with a saved reference run: four self-authored texts through `prompt_logprobs`, the rank and log-probability of each actual next token, and with `--reference` the argmax agreement, top-5 overlap and drift against an earlier record. A served full model does not repeat identical requests exactly (argmax agreement about 0.96 on the reference pair, cause not identified), so a run passes when every request was answered in the checked shape and the repeat difference is reported as the yardstick.
- **Requantization checks on the four-layer fixture**: `quant-error` (per-tensor weight-space error of NVFP4 tensors against their BF16 originals, byte identity of everything else), `agreement-fixture` (top-5 rows, full-vocabulary log-probabilities and layer-3 candidate sets) and `agreement-compare` (full-vocabulary KL, argmax agreement, candidate-set Jaccard). [Validation](docs/validation.md) records how far the unmodified fixture moves between two starts, the yardstick for any such comparison.
- **`runtime.derived_checkpoint`** (optional, absent by default, fingerprints of existing profiles unchanged): serve a locally requantized copy of the pinned checkpoint with the source overlays it needs. `server preflight` checks the checkpoint's declared target, that the MTP draft layer stays unquantized, and each overlay's hash, marker and the hash of the image file it replaces. Used once on the reference pair for the P23 A/B/A.
- **`validation.run_repeat_trace`** repeats a request inside one process and names the first module whose output differs from the first pass. On the eight-layer fixture that is always the routed experts: the token order inside each expert changes from call to call and the Marlin MoE result depends on it. `--canonical-align` fixes that order as a diagnostic and makes every pass bit-identical; `--zero-moe-buffers` rules out stale scratch memory. `--verify-canonical` shows the fix changes results only at that noise level, and `--timing` measured about 1% slower decode and unchanged prefill on the fixture. Upstream tracks the cause as vLLM issue #52525.
- **`fixture-build --with-mtp`** keeps the BF16 draft layer in the four-layer fixture, renumbered to follow the kept layers and excluded from the global NVFP4 setting in both quantization files, so MTP can be exercised on one GPU.
- **Releases follow tags.** Pushing a `vX.Y.Z` tag runs `.github/workflows/release.yml`, which publishes that version's Changelog section as the GitHub Release through `tools/release_notes.py`. Tags 1.2.0 to 1.5.0 had been pushed without a Release; those were created by hand, and 1.2.1, which had a Changelog section but no tag, was tagged at its release commit.

### Changed

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

Everything here follows from running 1.2.0 on the reference pair for the first time; `docs/MIA_ADOPTION_2_PLAN.md` Stage 6 owns the run.

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
