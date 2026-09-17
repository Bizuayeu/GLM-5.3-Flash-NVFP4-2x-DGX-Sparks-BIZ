# Operations

[日本語](operations.ja.md)

**Routine TP=2 deployment is not accepted yet.** A serial full-model reference profile has [experimental evidence](validation.md#full-model-tp2-experimental-scope) and [initial benchmarks](benchmarks.md). Whether a profile is experimental or ready for routine use is shown by its recorded acceptance status (the README status table and the [harness acceptance matrix](harnesses.md#acceptance-matrix-and-status)), not by a command name.

## One launcher

The checkout has one launch path: `python -m glm53_setup server …`, driven by [one server TOML](server-configuration.md) and wrapped by the two-rank [switch and recovery procedure](launch-safety.md#all-rail-checks-and-two-rank-switch) when a pair is already running. Every full-model measurement in this repository ran through it, and the [launch checks](#full-model-launch-checks) below are the checks it performs before a start.

## Artifact storage and paths

This section owns deployment storage paths. The model ID, revision and base-image digest are fixed by [runtime.lock.json](../config/runtime.lock.json). The base checkpoint is acquired upstream; the optional LPA projector is distributed as a separate GitHub Release asset. Keep operator-specific hostnames, absolute home paths and credentials outside the public source.

| Artifact | Default location on each Linux host | Role |
|---|---|---|
| Base checkpoint | `$HOME/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/<revision>/` | Fixed model/config/tokenizer view. Weight files link into the sibling `blobs/` directory, which holds their data |
| MTP metadata view, required by distributed defaults | `$HOME/.cache/huggingface/local-views/glm53-mtp-compatible/<revision>/` | Links existing tensor data and adjusts quantization metadata for the checkpoint's BF16 MTP; [create the view](speculative-decoding.md#prepare-a-view-on-each-linux-host) without editing the original snapshot |
| LPA projector, required when `lpa.enabled = true` (off in the template) | `<checkout>/state/lpa/glm53-lpa-cut32-v1/projector.pt`; selected by `[lpa].projector` relative to the [server TOML](server-configuration.md) or absolute | Separate Release asset, outside the NVIDIA snapshot and source archive. [Download and verify](lpa.md#download-the-trained-projector) on both hosts, or train a matching projector; [enabling](lpa.md#enable-lpa-in-the-server-profile) is a separate profile edit and switch. Plain inference and batching do not require it |
| Docker base/reference images | Docker-managed storage | Pull the fixed base and build the reference image from this source. Source checkout, image and checkpoint are separate artifacts |
| Local configuration and acquisition state | `<checkout>/state/` | The server TOML and `download-status.json`; the latter records the actual acquired `snapshot` path |
| Runtime/JIT cache and evidence | `<checkout>/state/tp2-runtime-cache/`, `<checkout>/records/` | Regenerable runtime data and private execution records; not model weights or distribution inputs. Distributed startup points the Triton, TileLang and TorchInductor caches into the runtime cache so compiled kernels survive restarts; the [warmup ladder](#supervision-stall-detection-and-warmup) records what still compiles |

The LPA asset expands as follows. `manifest.json` is a copy of the [projector lock](../config/lpa-projector.lock.json); source checkout archives do not include this directory.

A source archive also omits `state/` and `records/` by design. A host deployment must create symlinks from the new checkout to its persistent runtime directories before using the server launcher:

```sh
ln -sT /srv/glm53/state /srv/glm53/source/state
ln -sT /srv/glm53/records /srv/glm53/source/records
readlink -f /srv/glm53/source/state /srv/glm53/source/records
```

Use absolute host paths; `-T` makes `ln` fail instead of nesting `state/state` inside an existing directory, and `readlink -f` must print `/srv/glm53/state` and `/srv/glm53/records`, not paths ending in `state/state` or `records/records`. Retain the old checkout for recovery, and never place credentials or raw records in the source archive.

```text
state/lpa/glm53-lpa-cut32-v1/
├── projector.pt
├── manifest.json
├── README.md
├── README.ja.md
├── LICENSE
├── NOTICE
├── TRAINING_DATA.md
├── TEACHER_MODEL_CARD.md
└── LICENSES/
    └── ZAI-GLM-MIT.txt
```

The server launcher reads the default host Hugging Face cache and mounts it read-only at `/hf` in the container. It resolves the selected snapshot or MTP view within that mount. Preserve the entire model cache's `blobs`/`snapshots` relationship; copying a snapshot directory alone is insufficient. Both hosts need the complete checkpoint on disk; TP=2 partitions loaded tensors, not the downloaded files.

The downloader follows Hugging Face cache environment settings, but the current launcher assumes the default cache root. For this release, leave `HF_HOME`/`HF_HUB_CACHE` unset when acquiring these assets and use the documented default. A successful custom-cache download does not establish that the launcher can find or mount it.

Inspect the expected and recorded locations without starting a download, from the checkout on each Linux host:

```sh
python -c 'from pathlib import Path; from glm53_setup.config import MODEL, REVISION; print(Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION)'
python -c 'import json; from glm53_setup.config import STATE; s = json.loads((STATE / "download-status.json").read_text()); print(s.get("status"), s.get("snapshot", "not recorded"))'
```

The second command requires prior acquisition registration in this checkout. Neither printed path nor `status=complete` substitutes for checksum verification. Server settings must reference the image actually built and inspected on both machines.

## Acquire and verify once

Use the pinned revision from `config/runtime.lock.json`. `download` reuses Hugging Face cache files and prevents overlapping downloads in the same checkout. `verify-download` runs the official checksum verifier, which may contact Hugging Face for metadata. Offline inference is distinct from offline checksum verification.

For a second host, transfer the model's complete `blobs` and `snapshots` trees together. A snapshot contains links into `blobs`; copying or mounting only the snapshot can break those links. Preserve existing cache files and avoid delete-sync options.

After the transfer, run `download` once to register and check the fixed snapshot in that checkout. Matching cached files are reused; missing files may be fetched. Then run `verify-download` without `--wait`. Do not claim transfer success solely from file sizes.

Hashing 200 GB fills the page cache, which shares unified memory with the GPU. Another two-Spark recipe reports two power losses in nine checksum passes on idle GPUs (tonyd2wild PR #19, no license, no code adopted); verify before loading a model rather than beside a serving pair. Large reads on a serving host have the same effect on a smaller scale: exporting the 9.7 GiB reference image from the peer rank took its MemFree from 3.0 to 0.87 GiB while MemAvailable stayed above 8.8 GiB. NVIDIA's later checkpoint revision `09b04e5e` (2026-09-11) differs from the pinned revision only in `README.md`.

## Prepare each host

1. Inspect available memory, disk, GPU/driver, active model processes and host state. Stop another model through its own documented procedure before an eventual GLM launch.
2. Run `prepare-image` on each host. It pulls the pinned ARM64 base and records actual package versions, GPU calculation and GLM registration. The base's native NoPE path is not a qualified serving path.
3. Build the reference image once with `build-reference`. Its base digest comes from the lock. To replicate it, use Docker image save/load over the verified local link and compare actual image IDs.
4. Run the [single-GPU validation](validation.md). Keep the image, precision, source hashes and generated records together.

## Host kernel and multi-node RoCE

**Check the kernel before installing updates and before the two-host steps.** The measurements in this repository ran on `6.17.0-1032-nvidia` with driver 580.173.02 and ConnectX-7 firmware 28.45.4028 on MSI EdgeXpert (MS-C931). Kernel `7.0.0-1019-nvidia` is not validated here.

Updates available as of 2026-09-15 move the `linux-nvidia-hwe-24.04` metapackages to `7.0.0-1019-nvidia`, with the 580 open driver modules built for it. An `apt` upgrade or a DGX Dashboard update installs it, so a newly installed system boots it after its first update.

With that kernel's defaults, two-host NCCL over RoCE can fail with `NCCL WARN Call to ibv_reg_mr_iova2 failed with error Cannot allocate memory`. Reports describe the model loading and then failing during vLLM profiling or tensor-parallel communication, while raw RDMA tests such as `ib_write_bw` look healthy. NVIDIA's [update advisory](https://forums.developer.nvidia.com/t/dgx-spark-update-advisory/383254) (2026-09-13) recommends that multi-node/RoCE users hold off on this kernel, including updates through DGX Dashboard, and names no fixed release. Whether single-host workloads are affected is not established.

The analysis in [NV-Kernels PR #590](https://github.com/NVIDIA/NV-Kernels/pull/590) (open; a contributor's analysis, not an NVIDIA statement) traces the failure to Kexec HandOver (KHO). The `7.0.0-1019-nvidia` build sets `CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y` (checked in its package config; `6.17.0-1032-nvidia` does not enable KHO by default). At boot, KHO reserves scratch memory for a later kexec and releases it as CMA pageblocks, about 9.3 GiB (4,761 pageblocks) in that report, without counting them in `CmaTotal`. RDMA memory registration pins pages long-term, and pinned pages must first move out of CMA; under GPU memory pressure that migration fails and registration returns `ENOMEM`.

Configure both hosts the same way:

| Choice | Steps | Notes |
|---|---|---|
| Keep `6.17.0-1032-nvidia` | Before upgrading: `sudo apt-mark hold linux-nvidia-hwe-24.04 linux-image-nvidia-hwe-24.04 linux-headers-nvidia-hwe-24.04 linux-modules-nvidia-580-open-nvidia-hwe-24.04 linux-tools-nvidia-hwe-24.04`. If 7.0 is already installed, the previous kernel stays installed; boot it from the GRUB menu's advanced options (console access required). | The validated state of this repository. Release the holds when a fixed kernel is published. |
| Run `7.0.0-1019-nvidia` with KHO off | Add `kho=off` to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, keeping the existing values. Run `sudo update-grub` and reboot. Confirm `kho=off` in `/proc/cmdline`; `sudo ls /sys/kernel/debug/kho` must fail with "No such file or directory". | Posted in the advisory thread. The PR reports two-host registration, NCCL and TP2 workload tests passing with KHO off. Not yet validated by this repository. KHO serves kexec-based live update, which this deployment does not use. |

After either choice, repeat the [NCCL validation](nccl-validation.md) and a full-model launch before serving.

A separate report with the same error, on MS-C931 systems running an Ubuntu generic 7.0 kernel with driver 595.84, failed with about 118 GiB free before weights loaded and attributes the fix to MSI board firmware updates (embedded controller, SoC firmware, USB-C PD) ([MiaAI-Lab issue #259](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/259)). If the error appears without memory pressure, check the vendor firmware as well.

## Network and site configuration

For physical connection and persistent IPv4 configuration, use the [QSFP hands-on guide](qsfp-network.md).

Record each host's measured values in the `[nodes]` section of the [server TOML](server-configuration.md), which is the same file on both hosts:

- local fabric IPv4 and head fabric IPv4;
- Ethernet interface, RDMA HCA and that interface's RoCEv2 GID index;
- unused API and rendezvous ports in `[api]`.

The HCA and GID number need not match between hosts. Confirm the GID maps to the local IPv4 and net device. Use MTU 9000 only when both endpoints and the whole path support it. Verify the actual NCCL transport and collective correctness before loading the full model; a successful SSH connection is not an RDMA test.

```sh
python -m glm53_setup server plan --rank 0
python -m glm53_setup server preflight --rank 0
```

`plan` prints the container command without starting anything. `preflight` prints its checks as JSON and exits nonzero if any fails; it requires a completed, matching download state. `start` runs the same checks first and, only when they pass, saves them with the settings and the container command under `records/<timestamp>-server-r<N>/`.

## Full-model launch checks

`server preflight --rank N` checks, on each host, the pinned snapshot and MTP view, the fabric settings, the selected image ID and its capability markers, the projector checksum when LPA is enabled, that no other container holds the GPU, and the available memory; `server start` runs the same checks and refuses to start on a failure. The two-rank switch repeats them on both ranks before stopping a running pair and again before starting the new one.

`exclusive_gpu` fails while a running container requests a GPU without this launcher's `glm53.experiment.startup` label, and the result lists those containers under `foreign_gpu_containers`. A container requests a GPU when `HostConfig.DeviceRequests` is non-empty: both `--gpus` and CDI (`--device nvidia.com/gpu=...`) requests appear there, while the GPU device nodes never appear under `Devices`. A pair of this launcher carries the label whatever its fingerprint, so a running old pair does not block the pre-stop checks of `cluster switch`; an empty label value does not count. A container that exits or is removed while the check runs is skipped, and one still listed that cannot be inspected makes the check raise instead of passing. Stop other GPU workloads, such as a component probe or another model, before a start; there is no override. `server assets` runs the same check without the memory reading. Informed by sfxnz PR #12 (MIT, no code adopted).

A passing preflight certifies assets and configuration, not quality or availability: the remaining acceptance items for routine use are listed in [the setup runbook](../SETUP.md#6-qualify-the-full-model), and the current status per scope is in the README status table. Do not relax a failing check, truncate attention candidates or silently substitute precision to get past it.

Rank 1 starts headless first, followed by rank 0 once the worker is waiting for rendezvous. The API binds to the head's loopback address; use an SSH tunnel for a remote client. Internal rendezvous uses the fabric IP. Exposing it as a business service requires a separately reviewed authentication/TLS/access-control layer; this repository does not claim to supply one.

## Supervision, stall detection and warmup

The foreground supervisor on each rank samples `MemAvailable` every 2 seconds and stops its own container below `resources.reserve_gib` (`stop-reason: memory-reserve`). On rank 0 it also reads `/metrics` in the same cycle when `resources.stall_seconds` is positive: if requests are running and none of the generation-token count, prompt-token count, KV-cache usage and running count has moved for that long, it stops with `stop-reason: engine-stall` and records the frozen sample. `/health` keeps answering 200 while the engine is wedged (the V1 health check does not probe the workers), so it is not a liveness signal. KV usage moves during a chunked prefill and the prompt-token count is added at the first output token, so a long prompt is not a stall; the template's 600 s equals `generation.timeout_seconds` and covers the longest measured request, 493 s for a 256K reference request with chunk 2048. An unreachable `/metrics` is no evidence and never counts. Each line of `resources.jsonl` also records `mem_free_gib` and `free_2mib_gib`, the free memory in buddy blocks of 2 MiB or more summed over zones from `/proc/buddyinfo`. NVRM allocates such blocks without reclaiming the page cache, as described in tonyd2wild's GB10 memory notes (no license, no code adopted), which fits `NV_ERR_NO_MEMORY` appearing with more than 4 GiB available. On the reference head while serving, 7.1 GiB available came with 1.1 GiB free and 0.49 GiB in such blocks. Both readings are observations: only `MemAvailable` stops a rank, and an unreadable sample is recorded as `memory_sample_error` instead of stopping one. The same notes report that raising `vm.min_free_kbytes` to 4 GiB lowered vLLM's startup memory check by about 6.2 GiB, so this kit leaves it at the distribution default. The reference hosts measured why KV usage belongs in the set: during an 82,018-token prefill the two token counters stayed frozen for 202.8 seconds, while the four signals together never froze for more than 8.3 seconds. A detector watching only tokens would have to sit above the longest prefill it will ever serve. A supervised stop leaves the other rank running; stop it before starting a new pair, because `cluster switch` refuses an incomplete pair. These notes were informed by the field runbook in Mia PR #70; in both incidents there, killing the containers, confirming the GPU with a short CUDA probe and restarting recovered the kit without a power cycle.

The reference hosts keep 16 GiB of swap for host pages. `vm.swappiness=0` stops new page-outs but leaves pages that are already swapped where they are, and touching old swapped pages during a long prefill was reported as the trigger of a GB10 UVM livelock (same source). Cycle residual swap while both containers are down: `sudo swapoff -a && sudo swapon -a`. Keep the swap file itself; with no swap at all the worker was killed on allocation spikes. On 2026-09-17 the reference pair compared `vm.swappiness` 60 and 0 at chunk 2048, cycling swap before the 0 arm. At 60 no engine process had swapped pages; 0.38 GiB on the head and 0.28 GiB on the peer belonged to other processes, such as a search container and the desktop shell. At 0 swap stayed empty, prefill differed by 1.9%, within the variation between restarts, and the head's lowest free memory was 0.45 GiB lower while the peer's was 0.31 GiB higher. No benefit showed for this workload, so the hosts stay at the distribution's 60. Another two-Spark recipe requires 0 persisted in `/etc/sysctl.d` (tonyd2wild OPEN-PROBLEMS §4, no license, no code adopted); measure on your own load before persisting it.

**Host daemons compete for the same unified memory.** The supervisor stops the model when `MemAvailable` falls below the reserve, but the cause can be another process on the host, and then the model stops while the cause remains. On 2026-09-16 the peer rank stopped at 2.49 GiB against a 2.5 GiB reserve. The cause was a monitoring dashboard on the other host that opened a new SSH login to the peer for every metric it sampled, 3.6 logins per second. Each login created a logind session and a polkit check, so `polkitd` grew to 3.40 GiB over six days; each session change made `wireplumber` re-register its Bluetooth audio profiles, which `bluetoothd` rejected as already registered, so both of those grew too (to 0.69 and 1.15 GiB); and each login ran every `/etc/update-motd.d` script, about 660 new processes per second. The host running the dashboard reads its own metrics locally and stayed at 0.03 GiB. Enabling OpenSSH connection reuse for the dashboard's host aliases (`ControlMaster auto`, `ControlPersist`) took the peer from 218 logins a minute to none, with the dashboard's readings unchanged. Before a long run, count logins on each host with `journalctl -u ssh --since -60s | grep -c Accepted` and compare daemon sizes with `ps -eo user,rss,comm --sort=-rss | head`. A `MemoryMax` drop-in on a daemon that leaks also needs `Restart=on-failure`, because these units ship `Restart=no` and would otherwise stay dead once the cap kills them. Finally, a half-dead pair is invisible from the API: the surviving rank keeps answering `/health` with 200, so check `docker ps` on both hosts.

**How the cause was found, and what held afterwards.** The first suspect was Bluetooth: `wireplumber` retried the rejected registration about four times a second, and the peer's D-Bus connection serial stood at `:1.2020856` after six days against `:1.19052` after ten on the head. Stopping Bluetooth on the peer did not stop the churn: new D-Bus connections continued at 7.3 per second, `polkitd` still grew by 3.6 MB in 240 seconds, and the connections came from the display manager's greeter session, itself another symptom. Counting new process and thread IDs separated the hosts: 6,642 in 10 seconds on the peer against 176 on the head, which carries more model load. New-process snapshots then showed the login message scripts starting 30 times in 8 seconds, `journalctl -u ssh` listed 218 accepted logins in 60 seconds, all from the head's fabric address, and the parent of those `ssh` commands was the dashboard. With connection reuse, 30-second counts on the peer fell from about 220 new D-Bus connections to 1, from about 450 KB of `polkitd` growth to none and from about 19,900 new process and thread IDs to 1,191. Later that day all three hosts were power-cycled for recabling and Bluetooth was enabled again on the peer. With the dashboard watching the peer and a third host, `polkitd`, `bluetoothd` and `wireplumber` stayed the same size on both over 300 seconds, and no new rejected registration appeared. The cap put on `polkitd` during the incident (`MemoryMax=512M` with `Restart=on-failure`) stays on the peer as a second layer; its restart has not been exercised. On 2026-09-17 the dashboard was restarted while the model served: after opening its connections it logged in to the peer 0 times in the next 60 seconds.

**A lost peer is not detected while the pair is idle.** Mia issue #193 reports the mirror image of that incident on the same hardware: during sustained TP=2 serving the head host stopped answering on every network and needed a physical reboot. The container was not OOM-killed, and nothing was logged before it: no kernel, NCCL or container errors, and pstore was empty. On the worker, the kernel reported the RoCE link down and NCCL reported retry-exceeded completions; the worker container kept running and held more than 100 GB until an operator stopped it. The reporter does not attribute the lockup to the model and notes earlier unclean crashes on that host. Two points carry over to this kit. Each supervisor runs on the host it guards, so a host lockup takes its supervisor with it: the reserve guards against the model exhausting unified memory, not against the host itself locking up. And an idle half-dead pair is stopped by neither supervisor: memory stays above the reserve, and stall detection needs a running request. A stop reason for a lost peer, with each rank checking the other, is a design note and is not implemented.

`server warmup` sends a request ladder through the ordinary chat endpoint after readiness: a short text turn, a tool call, one synthetic image (when `runtime.vision` is on) and, when `generation.warmup_long_tokens` is set, a prompt of that many tokens sized with the served tokenizer. These are the shapes that were observed compiling kernels while serving ([image input](vision.md#limits-and-open-items)); the pinned launch disables vLLM's own JIT warmup, and one such compile burst pushed the head below its memory reserve. Compiled kernels persist in the runtime cache, but the ladder still reports them on every start: the pinned Triton calls its post-compile hook on the first use of a kernel in a process whether the binary was compiled or loaded from the on-disk cache, and the jit monitor warns from that hook (its TileLang check likewise looks only at the in-process cache). What the ladder moves before the first user request is that per-process load. The record (`records/<stamp>-warmup-r0/result.json`) lists each rung's seconds, prompt tokens and outcome, the kernels the jit monitor reported before and during the ladder, and whether the prefix cache was reset afterwards (only with `api.dev_endpoints = true`; otherwise the warmup prompts stay in the cache until evicted). With `generation.warmup = true`, `cluster switch` runs the ladder on rank 0 after both ranks are ready and stores the result under `warmup` in `result.json`; a ladder failure is recorded and does not fail or roll back the switch. `cluster resume` only re-observes readiness and does not run the ladder; run `server warmup` on the head afterwards if it is wanted. The long rung pays a full prefill on every start (measured with chunk 2048: about 490 s at 256K and 380 s at 200K; with 512: about 500 s at 200K and 206 s at 82K). On the reference hosts every ladder reports the same ten kernels, among them `BuildPrefillChunkMetadataKernel` and the TileLang `mhc_pre_big_fuse_with_norm_tilelang` shape that had once compiled while serving a user request; across five starts on 2026-09-17 the runtime cache (1,840 Triton and 55 TileLang files) gained no file, and the short rungs took 1–2 s.

## Recovery and records

The scripts do not delete failed containers or weights and do not install a restart watchdog. `server stop` stops only a container carrying this launcher's ownership label. Save its logs and rename a stopped container before recreating the same rank name. Reinitialize both ranks together after a distributed failure.

`state/` holds current acquisition/site state. `records/` holds per-run evidence. A paused acquisition is an intentional stop: verification waiting exits with code 2 and does not restart the download. Do not start a new acquisition while a local transfer is in progress.

Detailed numerical and backend limitations are in [validation.md](validation.md). Keep unresolved failures visible when preparing a release.
