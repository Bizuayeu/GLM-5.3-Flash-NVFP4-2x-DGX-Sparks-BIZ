# Operations — beta

**The full-model TP=2 deployment is not qualified yet.** Asset preparation and one-GPU diagnostics are usable; the guarded launcher is a candidate implementation.

## Acquire and verify once

Use the pinned revision from `config/runtime.lock.json`. `download` reuses Hugging Face cache files and prevents overlapping downloads in the same checkout. `verify-download` runs the official checksum verifier, which may contact Hugging Face for metadata. Offline inference is distinct from offline checksum verification.

For a second host, transfer the model's complete `blobs` and `snapshots` trees together. A snapshot contains links into `blobs`; copying or mounting only the snapshot can break those links. Preserve existing cache files and avoid delete-sync options.

After the transfer, run `download` once to register and check the fixed snapshot in that checkout. Matching cached files are reused; missing files may be fetched. Then run `verify-download` without `--wait`. Do not claim transfer success solely from file sizes.

## Prepare each host

1. Inspect available memory, disk, GPU/driver, active model processes and host state. Stop another model through its own documented procedure before an eventual GLM launch.
2. Run `prepare-image` on each host. It pulls the pinned ARM64 base and records actual package versions, GPU calculation and GLM registration. The base's native NoPE path is not a qualified serving path.
3. Build the reference image once with `build-reference`. Its base digest comes from the lock. To replicate it, use Docker image save/load over the verified local link and compare actual image IDs.
4. Run the [single-GPU validation](validation.md). Keep the image, precision, source hashes and generated records together.

## Network and site configuration

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

The candidate `service start` path requires a `state/kernel-validation.json` receipt with a passing `tp2-kernel-validation` result tied to the exact image and model revision. **This beta does not yet include a completed producer/workflow for that receipt.** It must come from real two-rank qualification, not manual editing or a one-GPU fixture.

The candidate launcher still selects the lock's base image, whose native NoPE path is blocked. Building the reference image does not select it for serving. Full-model qualification must implement and validate that runtime selection and its receipt binding; see [the ordered setup gate](../SETUP.md#6-qualify-the-full-model--current-beta-blocker).

After that future qualification, rank 1 starts headless first, followed by rank 0 once the worker is waiting for rendezvous. The API binds to the head's loopback address; use an SSH tunnel for a remote client. Internal rendezvous uses the fabric IP. Exposing it as a business service requires a separately reviewed authentication/TLS/access-control layer; this repository does not claim to supply one.

## Recovery and records

The scripts do not delete failed containers or weights and do not install a restart watchdog. `service stop` stops only a container carrying this deployment's ownership label. Save its logs and rename a stopped container before recreating the same rank name. Reinitialize both ranks together after a distributed failure.

`state/` holds current acquisition/site state. `records/` holds per-run evidence. A paused acquisition is an intentional stop: verification waiting exits with code 2 and does not restart the download. Do not start a new acquisition while a local transfer is in progress.

Detailed numerical and backend limitations are in [validation.md](validation.md). Keep unresolved failures visible when preparing a release.
