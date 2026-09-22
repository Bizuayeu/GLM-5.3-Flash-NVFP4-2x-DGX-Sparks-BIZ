# Deployment runbook

[日本語](SETUP.ja.md) · [Project overview](README.md)

The launcher and its client read [one server TOML](docs/server-configuration.md).

**The serial full-model TP=2 reference profile is accepted for routine use (2026-09-22) within its declared scope, on the evidence recorded in [step 6](#6-qualify-the-full-model). Outside that scope — other hardware, multiple active sequences, video input — nothing is qualified, and harness acceptance is recorded per case in [harnesses](docs/harnesses.md#acceptance-matrix-and-status).**

This is the ordered runbook for a human or an AI operator. Exact pins live in [the runtime lock](config/runtime.lock.json); command behavior and recovery belong to [operations](docs/operations.md); test commands and evidence belong to [validation](docs/validation.md). Read all three before execution. The distributed profile accepts text, tool calls and images, with video rejected; qualify text and tool calls first, then [image input](docs/vision.md).

## Shortest path to a smoke test

For two GB10 hosts that already passed [step 1](#1-collect-inputs-and-inspect-both-hosts). This is a map of the sequence, not a replacement for it: the numbered sections stay authoritative, and the qualification items of [step 6](#6-qualify-the-full-model) are what make a working smoke test routine use.

1. Check out the same reviewed commit on both hosts and run the CPU tests there ([step 2](#2-prepare-the-same-checkout-on-both-hosts)).
2. Download the pinned checkpoint once, verify its checksums, copy the cache to the peer and verify that copy too ([step 3](#3-acquire-the-checkpoint-once-and-verify-each-copy)).
3. Build the reference image once, put it on the peer by transfer or a second build, and record and compare both image IDs ([step 4](#4-prepare-images-and-test-the-reference-implementation)).
4. Connect the cable and qualify the fabric with the two-rank NCCL diagnostic ([step 5](#5-connect-and-qualify-the-fabric--cable-required)).
5. Fill the server TOML from [the distributed template](docs/server-configuration.md#distributed-defaults) and place it at the same path on both hosts.
6. Run `server plan` and `server preflight` on both ranks and clear every finding ([step 6](#6-qualify-the-full-model)).
7. Start the pair with `cluster switch` ([switch procedure](docs/launch-safety.md#all-rail-checks-and-two-rank-switch)), or with `server start` on rank 1 and then rank 0 ([commands](docs/server-configuration.md#commands)), and let the warmup ladder finish.
8. Send a short request through the API and read the answer ([step 7](#7-serve-and-accept--only-after-step-6-passes)).

## 1. Collect inputs and inspect both hosts

| Required input | Requirement / decision |
|---|---|
| Two systems | DGX Spark or compatible Linux ARM64 GB10 systems in the 128 GB unified-memory class; record each vendor, model, OS, driver and usable memory. Compatibility is established by tests, not branding. |
| Fabric cable | At least one direct QSFP link suitable for both systems' ConnectX-7 Ethernet/RoCE ports. Confirm cable and port compatibility with each hardware vendor. A QSFP connector alone does not establish compatibility. |
| Management access | Working SSH to both hosts, verified host keys, a known management path that survives fabric changes, and an account permitted to use GPU Docker. Keep keys outside the checkout. |
| Storage | About 205 GB for the pinned checkpoint on each host, plus space for Docker images, build layers, runtime caches, logs and optional fixtures. Measure free space on the actual cache and Docker filesystems. Reserve transfer-archive space if using archives. |
| Software / access | Python 3.11+, venv/pip, Git, Docker with NVIDIA GPU access; access to Hugging Face and the pinned image registry during acquisition. Image build and GPU tests run on Linux ARM64. |
| Local choices | Checkout path, management aliases, rank assignment, cache location, unused fabric subnet/ports, logging location and a bounded test window. |
| Existing workloads | Identify other inference servers, downloads and memory consumers. Do not stop an unrelated job based only on a process name. Resolve resource ownership before loading GLM; `server preflight` refuses to start while another container requests the GPU. |

NVIDIA documents Ethernet-mode QSFP ports up to 200 Gb/s per port and recommends cables capable of at least that rate. Follow its [network guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) and your compatible-system vendor's instructions. This runbook does not require a switch for a direct two-host link or claim dual-cable bandwidth aggregation.

Run read-only inventory on **each** host and save outputs privately:

```sh
date -Is
uname -a
cat /proc/cmdline
cat /etc/os-release
nvidia-smi
free -h
df -h
docker version
docker ps -a
ip -brief address
ip route
rdma link show
ibdev2netdev
```

If a diagnostic is missing, record that fact and install only the required vendor-supported package within the deployment authorization. Do not blanket-upgrade the OS, driver or firmware as a diagnostic step. Compare both inventories; do not assume their interface names, HCA names or GID indices match.

**Kernel:** if `uname -r` shows `7.0.0-1019-nvidia`, or pending updates would install it, choose between keeping `6.17.0-1032-nvidia` and booting with `kho=off` as described in [host kernel and multi-node RoCE](docs/operations.md#host-kernel-and-multi-node-roce), before step 5. With that kernel's defaults, two-host RoCE can fail with `ibv_reg_mr_iova2 ... Cannot allocate memory`.

**Checkpoint:** both hosts accessible; resource and storage budget recorded; kernel and boot parameters recorded; cable status known. Without the cable, continue steps 2–4 when their prerequisites hold and leave step 5 pending.

## 2. Prepare the same checkout on both hosts

Use the same reviewed Git commit on both hosts. Run from its root:

```sh
git rev-parse HEAD
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/huggingface.lock.txt
python -m glm53_setup --version
python -m unittest discover -s tests -t . -v
python tools/check_publication.py
```

Do not copy `state/`, credentials or local records into Git. Each host owns its own state. Use the default `$HOME/.cache/huggingface` for this release: the launcher assumes that location. Custom cache environment variables are not integrated into its mount resolution yet. Store state and reports under this checkout's `state/` and `records/`.

When deploying a source archive to a new checkout, connect its Git-excluded runtime directories to the host's persistent state before running `server preflight` or `cluster switch`. The source archive intentionally omits these directories. Keep the old checkout and its records intact until the new pair is ready:

```sh
ln -sT /srv/glm53/state /srv/glm53/source/state
ln -sT /srv/glm53/records /srv/glm53/source/records
```

Use the actual absolute paths on each host. `-T` makes `ln` fail instead of creating `state/state` inside an existing directory; `readlink -f /srv/glm53/source/state` must print `/srv/glm53/state`, not a path ending in `state/state`. Do not copy credentials or raw records into the source archive. The `state/server.toml` path used by a switch must be the same path that the remote checkout resolves through this link.

**Checkpoint:** same source commit and lock on both nodes; CPU tests pass.

## 3. Acquire the checkpoint once and verify each copy

First review [checkpoint, MTP-view and LPA-projector storage](docs/operations.md#artifact-storage-and-paths). Use each Linux host's default HF cache and keep base weights distinct from auxiliary artifacts. That section provides read-only commands to compare acquisition and launch locations.

The source is [NVIDIA's GLM-5.3-Flash-NVFP4 repository](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4). The downloader reads the exact revision from the lock; never substitute `main`, another quantization, or a similarly named model. Review the pinned snapshot's model license and [third-party notices](THIRD_PARTY_NOTICES.md). Project licensing does not replace model/dependency terms.

On the chosen download host, follow [README asset preparation](README.md#prepare-assets). Record the manifest, snapshot, download status and successful checksum result. A “complete” download state checks presence and sizes; checksum verification is a separate required step.

Transfer the complete model cache (`blobs` plus `snapshots`, preserving links) to the other host after the link is ready; follow [cache transfer and verification](docs/operations.md#acquire-and-verify-once). Save source/destination paths and transfer result. Do not use delete-sync or overwrite another model's cache. Verify both copies against the same pinned revision. Checksum verification may require online metadata even though inference is offline. Verify before loading the model: hashing fills the page cache that shares memory with the GPU ([why](docs/operations.md#acquire-and-verify-once)).

If a download is intentionally paused, preserve partial files and leave it paused until authorized to resume. Do not run a downloader and a cache transfer against the same destination concurrently. A cable delay does not justify duplicating a large Internet download.

**Checkpoint:** both local snapshots independently verified; otherwise record exactly which node remains pending.

## 4. Prepare images and test the reference implementation

Follow [host preparation](docs/operations.md#prepare-each-host): inspect the pinned ARM64 base, build the reference image once, and transfer that built image to the peer when practical. Record actual image IDs on both nodes and compare them; matching mutable tags are insufficient. The base digest and source-hash checks protect against accidentally patching a different vLLM release.

**Building the reference image is required to install this repository's vLLM/GLM runtime patches**, including [canonical sparse candidate ordering](docs/candidate-order.md). Run `python -m glm53_setup build-reference` from the reviewed checkout; no manual vLLM source editing is required. The official base image alone does not contain these changes. After a source update, rebuild and verify the new image ID before replacing containers: an existing image or running container is not updated automatically. A source-hash mismatch must stop the build, not be bypassed.

Run the [single-GPU fixture procedure](docs/validation.md#reproduce-the-single-gpu-fixture) on the first host. Keep its resource limits, selected precision, output and assessment together. A fixture pass checks selected kernels/state behavior; it cannot establish full-model quality or TP=2 correctness. Repeat appropriate component checks on the peer once available.

**Checkpoint:** pinned base inspected; reference image identified; fixture assessment and remaining numerical limitations recorded.

## 5. Connect and qualify the fabric — cable required

Follow the [QSFP and NetworkManager hands-on guide](docs/qsfp-network.md), starting with management SSH, cable/interface identification and one-host-at-a-time configuration.

Have a person physically connect the supported cable. Follow the vendor's networking procedure while preserving management access. Record existing network configuration before a change and its restoration procedure. Do not assign an illustrative subnet until checking existing routes on both hosts.

Inventory the live Ethernet interface, HCA and RoCEv2 GID mapped to each local IPv4. Configure [per-host site settings](docs/operations.md#network-and-site-configuration). Treat every example value as a placeholder. MTU changes must work end-to-end; do not blindly set 9000. Test both directions and distinguish SSH/IP connectivity from RDMA transport.

Before full weights are loaded, follow the [two-rank NCCL diagnostic](docs/nccl-validation.md). Save the command, tool version, rank placement, transport log, payload sizes, data checks and measured bandwidth. Confirm the intended RDMA interfaces and passing data checks. **This release has no production bandwidth threshold; the full model's routine-use acceptance rests on the evidence recorded in [step 6](#6-qualify-the-full-model), not on a number measured here.** Agree on the performance criterion and document it before accepting performance; a ping or an unexamined bandwidth number cannot close it.

**Checkpoint:** correct two-rank collective data and intended transport demonstrated, or explicitly pending/failed with evidence.

## 6. Qualify the full model

The [experimental scope](docs/validation.md#full-model-tp2-experimental-scope) and [initial benchmarks](docs/benchmarks.md) have evidence for one active sequence. Acceptance for routine use was closed on 2026-09-22 on that recorded evidence, not by a separate launcher; the items below are what it rests on, and the table after them says where each one is recorded.

Optional [MTP k=1 and k=3 experiments](docs/speculative-decoding.md) have also passed the basic API and matched benchmark cases; k=3 is preferred for further evaluation. Prepare their separate metadata view on each host before enabling speculation; a flag alone misclassifies the BF16 MTP tensors. Follow the documented memory/performance comparison and preserve the MTP-off baseline.

Inspect without launching:

```sh
python -m glm53_setup server plan --rank 0
python -m glm53_setup server preflight --rank 0
```

Both require a matching download state, the built reference image on that host and a filled-in server TOML; the [launch checks](docs/operations.md#full-model-launch-checks) list what preflight covers, including the refusal to start beside another GPU container. Do not change the lock's base digest into the reference tag: it anchors image preparation and source patching.

Never relax a failing check, reuse the one-GPU fixture as two-rank evidence, truncate attention candidates, or silently substitute precision.

Before calling a profile ready for routine use, verify and record at least:

- All language layers load on two ranks; peak memory, reserve and KV allocation measured on each host; no OOM or swap thrashing.
- Short/long text, declared context boundaries, concurrent/serial requests, cancellation and repeated request/state behavior meet documented criteria.
- Tool calls have parseable names/JSON arguments; a harmless tool round-trip returns a valid final answer. A model's tool request is not permission to execute arbitrary commands.
- Precision/backend, quality and latency/throughput meet a declared baseline and acceptance criteria. Do not claim W4A4 behavior from W4A16 evidence.
- Controlled stop/restart and distributed failure recovery succeed within the approved test window; both ranks recover together.

**Recorded evidence (2026-09-22):**

| Item above | Where it is recorded |
|---|---|
| Layers, memory, reserve, KV, no OOM | [benchmarks](docs/benchmarks.md) for the serving profile — weights 91.34 GiB per rank, FP8 KV 3 GiB per rank, lowest available memory on the head 10.50 GiB over the same-night long-input bench — and the [launch checks](docs/operations.md#full-model-launch-checks) each start performs |
| Text, context boundary, repeated requests, cancellation | [benchmarks](docs/benchmarks.md): 199,652-token and 261,461-token requests answered correctly within the declared boundary of 262,144 tokens, and identical requests repeat bit for bit. The profile serves one active sequence (`max_num_seqs = 1`), so concurrent requests queue; that is the declared behaviour. Cancellation is [harnesses](docs/harnesses.md#acceptance-matrix-and-status) H-06 PARTIAL: stopping a background job is verified, interrupting generation itself is untested and is the one open sub-item |
| Tool calls | [harnesses](docs/harnesses.md#acceptance-matrix-and-status) API-03 PASS |
| Precision/backend, quality, throughput | [validation](docs/validation.md) for W4A16 Marlin, [benchmarks](docs/benchmarks.md) for the teacher-forced NLL table and the throughput baselines. W4A4 behaviour is not claimed |
| Controlled stop/restart and pair recovery | `cluster switch` with its warmup ladder: two switches on 2026-09-22 completed without recovery ([benchmarks](docs/benchmarks.md)); the recovery path itself was exercised by the earlier drills in [launch safety](docs/launch-safety.md#all-rail-checks-and-two-rank-switch) |

**Verdict:** routine use is accepted from 2026-09-22 for the serving profile. Anything outside the declared scope — multiple active sequences, video input, other hardware — stays outside it.

## 7. Serve and accept — only after step 6 passes

Start rank 1 first and then rank 0 with `server start`, as described in [server configuration](docs/server-configuration.md#commands). Record both image IDs, source/model revisions, arguments, settings and start logs. Check the API through loopback or a reviewed SSH tunnel, then repeat text and harmless tool acceptance tests through the actual client.

Run the [harness acceptance matrix](docs/harnesses.md) for **the accepted route, the npm ZCode CLI**. The official ZCode Desktop stays BLOCKED and Claude Code is skipped by decision (2026-09-22, both recorded in that document), so neither is a required target. Basic API success alone does not close a client case. Keep client versions, non-secret settings and separate case results. Review [artifact-specific licensing](docs/licensing.md) before distributing a deployment.

Keep this service on a trusted network. The host-network containers expose distributed control ports to reachable peers; loopback API binding alone does not protect rendezvous. Public exposure, authentication/TLS, firewall policy and business availability requirements need their own deployment design. The “BIZ” suffix in the project name states a business-use intent; it is not a production certification or a support commitment.

## Completion checklist and AI handoff

Use **PASS / FAIL / PENDING / NOT RUN** with an evidence path for every item. Never infer PASS from absence of errors.

- [ ] Two host inventories, authorized access, resource ownership and disk budgets recorded.
- [ ] Same reviewed source and pinned artifacts; applicable licenses/notices reviewed.
- [ ] Both checkpoint copies checksum-verified; paused/partial acquisitions accounted for.
- [ ] Exact runtime image IDs match; source patch checks and one-GPU diagnostics recorded.
- [ ] Supported cable connected; per-host IP/interface/HCA/GID/MTU measured and recorded.
- [ ] Two-rank collective correctness and intended RDMA transport verified.
- [ ] Every acceptance item of [step 6](#6-qualify-the-full-model) verified on this pair, with its evidence path recorded.
- [ ] Actual API text/tool acceptance, memory, performance and recovery checks passed.
- [ ] The [accepted harness route](docs/harnesses.md#acceptance-matrix-and-status) completed its required acceptance cases; failures/blockers remain visible.
- [ ] Access boundary, logs, stop/restart procedure and operator handoff accepted.

Keep a private `records/<run-id>/REPORT.md` containing: timestamp/timezone; objective and approved scope; host/rank inventory; Git commit/model revision/image IDs; each step's status, command, exit code and evidence path; decisions and tradeoffs; unexpected events/recovery; the checklist; unresolved blockers and exact next action. Redact secrets from reports and publish only reviewed summaries.

Suggested AI task:

> Read AGENTS.md, SETUP.md and its linked operations/validation documents. Inspect current state on the two authorized hosts before mutating anything. Execute eligible steps in order within the approved scope, preserve unrelated jobs, credentials, weights and past evidence, and keep the private deployment report current. Respect deliberate pauses and verify each result. Where a physical prerequisite or a piece of the recorded evidence is missing, record the blocker and continue independent preparation. Do not record a PASS or declare deployment complete without actual evidence.
