# Launch contracts and operational validation

[日本語](launch-safety.ja.md)

These changes extend P10 (memory), P19/P22 (APC) and E03 (operations). They do not introduce another acceleration claim. Implementation and CPU contracts are present; real-container and full-model regression results must be recorded separately before adoption. The regular `service` qualification gate remains unchanged.

## Model API clients

The common transport uses a nonempty `API_KEY`, otherwise a nonempty `VLLM_API_KEY`. If both are empty or absent it sends no Authorization header. This includes `startup ask`, dependent benchmarks, profiler controls and component readiness checks. The origin must explicitly identify the model API; all redirects are refused, including same-origin redirects. Downloads use a separate unauthenticated transport. Keys are never settings or fingerprint inputs; HTTP failures expose the status code without headers or response bodies. HTTP401/403 are failures, not samples omitted from a successful score.

The pinned vLLM authentication middleware guards `/v1`, `/v2`, `/inference` and `/cohere`. It does **not** authenticate `/health`, `/metrics`, `/tokenize`, `/collective_rpc`, `/reset_prefix_cache` or profiler controls. Sending Bearer credentials does not change that server-side boundary. The listener remains loopback-only. This change adds client authentication, not a new public server authentication layer.

## Allocator and shared launch settings

Optional `runtime.cuda_allocator_conf` maps to `PYTORCH_CUDA_ALLOC_CONF`. Omission preserves the image/runtime default; a string is passed exactly, including an explicitly empty string. The verified baseline image has no allocator environment entry. This does not qualify a hidden-state KV connector.

Resolve environment overrides once on the launch-origin host:

```sh
python -m glm53_setup startup freeze --config state/startup.toml --output state/launch.json
python -m glm53_setup startup plan --config state/startup.toml --launch state/launch.json --rank 0
```

The environment's **presence**, including an empty value, overrides TOML. Copy the same frozen JSON to both ranks and supply `--launch` on start/preflight. Rank hosts do not resolve their own allocator environments. A direct start with a local allocator environment requires a frozen manifest. The manifest contains the resolved profile and its lock-bound fingerprint, with no API key. Existing manifests cannot be overwritten by `freeze`.

## All-rail checks and two-rank switch

Each node retains its primary `hca`, `interface`, `local_ip` and `gid_index` (port 1). Optional `additional_rails` lists records with those fields plus `port`. All rails require the same GID index, unique port/NIC/IP associations, an active Ethernet port/link, matching IPv4-mapped RoCE v2 GID, and an IP assigned to that NIC. Comma-separated device strings are rejected in favor of structured records. NCCL receives the complete exact HCA/port list; socket bootstrap stays on the primary interface. These are configuration checks, not multi-rail traffic qualification.

```sh
python -m glm53_setup cluster switch --config state/startup.toml \
  --hosts spark-head spark-peer --checkout /srv/glm53/source \
  --remote-config /srv/glm53/state/startup.toml \
  --output records/switch-run --experimental
```

Both hosts need the same audited checkout, images and assets. `--remote-config` supplies the Linux path base for relative projector paths; the shared frozen manifest supplies settings. `--ssh-config` can select an SSH config. Static assets/fabric and common source/image/model/profile/allocator identities are checked on both ranks and rechecked before stopping. Weight identities use index hashes, shard sizes and local file identities; this is not a replacement for the original weight-integrity verification. The load-memory gate runs after stop.

Pre-stop failures preserve running containers. Post-stop failures clean up only newly reserved owned attempts, then attempt the recorded old profile. Cleanup uncertainty prevents a competing recovery launch. Results distinguish failure, cleanup and recovery; this is a stop/start procedure with downtime, not an atomic switch. A live rank without a recorded configuration path is rejected before stop because its recovery is not established. Other containers are not stopped.

## APC history qualification

Keep the pinned runtime's actual retention baseline until measured otherwise. In this revision `prefix_cache_retention_interval` defaults to **0**: semantic checkpoints/replay boundaries and shared-prefix junctions; it does not mean dense retention. No new retention override or SWA-only rule is introduced for GLM's KDA/MLA/indexer/MTP groups. A reduction that applies to every group is not assumed safe.

The additional matrix covers append, edits/branches at 10/50/90%, alternating conversations, eviction pressure, actual block/pool/MTP boundaries and exact revisits after LPA. Record actual token prefixes and joint H, recomputation, timing, memory and MTP/LPA activity. Native exact controls precede P22 combinations. Retention changes require a separate A/B/A decision; CPU state checks and earlier P22 tests alone do not qualify this extended matrix.

The requirements were informed by Mia PRs #130/#136/#172/#175; no Mia implementation was copied or mechanically rewritten, and no new AGPL dependency was added. Implementation uses this repository and its pinned Apache-2.0 vLLM. EXL3/DFlash, adaptive-k, four-node serving, cable/IP changes and connector implementation are outside this scope.
