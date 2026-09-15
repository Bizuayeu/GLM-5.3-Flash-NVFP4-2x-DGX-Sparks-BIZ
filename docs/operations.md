# Operations

[日本語](operations.ja.md)

**Routine TP=2 deployment is not qualified yet.** A serial full-model reference profile has [experimental evidence](validation.md#full-model-tp2-experimental-scope) and [initial benchmarks](benchmarks.md); the guarded routine launcher remains a candidate implementation.

## Two launchers

The checkout has two launch paths, and evidence from one does not qualify the other.

| Command | Role | Status |
|---|---|---|
| `python -m glm53_setup startup …` | Experimental reference launcher and serial client, driven by [one startup TOML](startup-configuration.md); the two-rank [switch and recovery procedure](launch-safety.md#all-rail-checks-and-two-rank-switch) wraps it | Every full-model measurement in this repository ran through it. Experimental: no routine-deployment qualification |
| `python -m glm53_setup service …` | Guarded routine-deployment candidate with `state/site.json` and the [full-model launch gate](#full-model-launch-gate) | `plan` and `preflight` are for inspection. `start` still selects the lock's base image, whose native GB10 NoPE path is blocked, and requires a qualification receipt that no workflow produces yet; building the reference image does not switch it |

## Artifact storage and paths

This section owns deployment storage paths. The model ID, revision and base-image digest are fixed by [runtime.lock.json](../config/runtime.lock.json); the repository does not distribute weights. Keep operator-specific hostnames, absolute home paths and credentials outside the public source.

| Artifact | Default location on each Linux host | Role |
|---|---|---|
| Base checkpoint | `$HOME/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/<revision>/` | Fixed model/config/tokenizer view. Weight files link into the sibling `blobs/` directory, which holds their data |
| MTP metadata view, required by distributed defaults | `$HOME/.cache/huggingface/local-views/glm53-mtp-compatible/<revision>/` | Links existing tensor data and adjusts quantization metadata for the checkpoint's BF16 MTP; [create the view](speculative-decoding.md#prepare-a-view-on-each-linux-host) without editing the original snapshot |
| LPA projector, required when `lpa.enabled = true` (off in the template) | A private file selected by `[lpa].projector` in the [startup TOML](startup-configuration.md), relative to that TOML or absolute | Separate trained auxiliary weights, not part of the NVIDIA snapshot or source archive. Obtain or train the matching projector using the [LPA procedure](lpa.md); verify its hash on both hosts. Plain inference and batching do not require it |
| Docker base/reference images | Docker-managed storage | Pull the fixed base and build the reference image from this source. Source checkout, image and checkpoint are separate artifacts |
| Local configuration and acquisition state | `<checkout>/state/` | Site startup settings and `download-status.json`; the latter records the actual acquired `snapshot` path |
| Runtime/JIT cache and evidence | `<checkout>/state/tp2-runtime-cache/`, `<checkout>/records/` | Regenerable runtime data and private execution records; not model weights or distribution inputs. Distributed startup points the Triton, TileLang and TorchInductor caches into the runtime cache so compiled kernels survive restarts |

The experimental startup launcher reads the default host Hugging Face cache and mounts it read-only at `/hf` in the container. It resolves the selected snapshot or MTP view within that mount. Preserve the entire model cache's `blobs`/`snapshots` relationship; copying a snapshot directory alone is insufficient. Both hosts need the complete checkpoint on disk; TP=2 partitions loaded tensors, not the downloaded files.

The downloader follows Hugging Face cache environment settings, but the current launcher assumes the default cache root. For this release, leave `HF_HOME`/`HF_HUB_CACHE` unset when acquiring these assets and use the documented default. A successful custom-cache download does not establish that the launcher can find or mount it.

Inspect the expected and recorded locations without starting a download, from the checkout on each Linux host:

```sh
python -c 'from pathlib import Path; from glm53_setup.config import MODEL, REVISION; print(Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION)'
python -c 'import json; from glm53_setup.config import STATE; s = json.loads((STATE / "download-status.json").read_text()); print(s.get("status"), s.get("snapshot", "not recorded"))'
```

The second command requires prior acquisition registration in this checkout. Neither printed path nor `status=complete` substitutes for checksum verification. Startup settings must reference the image actually built and inspected on both machines.

## Acquire and verify once

Use the pinned revision from `config/runtime.lock.json`. `download` reuses Hugging Face cache files and prevents overlapping downloads in the same checkout. `verify-download` runs the official checksum verifier, which may contact Hugging Face for metadata. Offline inference is distinct from offline checksum verification.

For a second host, transfer the model's complete `blobs` and `snapshots` trees together. A snapshot contains links into `blobs`; copying or mounting only the snapshot can break those links. Preserve existing cache files and avoid delete-sync options.

After the transfer, run `download` once to register and check the fixed snapshot in that checkout. Matching cached files are reused; missing files may be fetched. Then run `verify-download` without `--wait`. Do not claim transfer success solely from file sizes.

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

For the explicit experimental reference path, use [one startup TOML](startup-configuration.md). The `service` instructions below describe the separate candidate launcher and its qualification gate.

For physical connection and persistent IPv4 configuration, use the [QSFP hands-on guide](qsfp-network.md).

Copy `examples/site.example.json` to `state/site.json` separately on each host. Replace all illustrative values with observations from that host:

- rank 0 or 1, local fabric IPv4 and head fabric IPv4;
- Ethernet interface, RDMA HCA and that interface's RoCEv2 GID index;
- unused API and rendezvous ports.

The HCA and GID number need not match between hosts. Confirm the GID maps to the local IPv4 and net device. Use MTU 9000 only when both endpoints and the whole path support it. Verify the actual NCCL transport and collective correctness before loading the full model; a successful SSH connection is not an RDMA test.

```sh
python -m glm53_setup service plan
python -m glm53_setup service preflight
```

`plan` prints arguments without starting a container. `preflight` records failures and exits nonzero if any requirement is missing. It also requires a completed, matching download state.

## Full-model launch gate

The candidate `service start` path requires a `state/kernel-validation.json` receipt with a passing `tp2-kernel-validation` result tied to the exact image and model revision. **This release does not yet include a completed producer/workflow for that receipt.** It must come from real two-rank qualification, not manual editing or a one-GPU fixture.

The candidate launcher still selects the lock's base image, whose native NoPE path is blocked. Building the reference image does not select it for serving. Full-model qualification must implement and validate that runtime selection and its receipt binding; see [the ordered setup gate](../SETUP.md#6-qualify-the-full-model--current-blocker).

After that future qualification, rank 1 starts headless first, followed by rank 0 once the worker is waiting for rendezvous. The API binds to the head's loopback address; use an SSH tunnel for a remote client. Internal rendezvous uses the fabric IP. Exposing it as a business service requires a separately reviewed authentication/TLS/access-control layer; this repository does not claim to supply one.

## Recovery and records

The scripts do not delete failed containers or weights and do not install a restart watchdog. `service stop` stops only a container carrying this deployment's ownership label. Save its logs and rename a stopped container before recreating the same rank name. Reinitialize both ranks together after a distributed failure.

`state/` holds current acquisition/site state. `records/` holds per-run evidence. A paused acquisition is an intentional stop: verification waiting exits with code 2 and does not restart the download. Do not start a new acquisition while a local transfer is in progress.

Detailed numerical and backend limitations are in [validation.md](validation.md). Keep unresolved failures visible when preparing a release.
