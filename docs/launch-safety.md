# Launch contracts and operational validation

[日本語](launch-safety.ja.md)

These changes extend P10 (memory), P19/P22 (APC) and E03 (operations). They do not introduce another acceleration claim. Implementation and CPU contracts are present; real-container and full-model regression results must be recorded separately before adoption.

## Model API clients

The common transport uses a nonempty `API_KEY`, otherwise a nonempty `VLLM_API_KEY`. If both are empty or absent it sends no Authorization header. This includes `server ask`, dependent benchmarks, profiler controls and component readiness checks. The origin must explicitly identify the model API; all redirects are refused, including same-origin redirects. Downloads use a separate unauthenticated transport. Keys are never settings or fingerprint inputs; HTTP failures expose the status code without headers or response bodies. HTTP 401/403 are failures, not samples omitted from a successful score.

The pinned vLLM authentication middleware guards `/v1`, `/v2`, `/inference` and `/cohere`. It does **not** authenticate `/health`, `/metrics`, `/tokenize`, `/collective_rpc`, `/reset_prefix_cache` or profiler controls. Sending Bearer credentials does not change that server-side boundary. The listener remains loopback-only. This change adds client authentication, not a new public server authentication layer. `api.dev_endpoints = true` mounts the dev routes (cache reset, collective RPC, sleep) on a profile that would not otherwise run in dev mode; they are equally unauthenticated ([server configuration](server-configuration.md#commands)).

## Allocator and shared launch settings

Optional `runtime.cuda_allocator_conf` maps to `PYTORCH_CUDA_ALLOC_CONF`. Omission preserves the image/runtime default; a string is passed exactly, including an explicitly empty string. The verified baseline image has no allocator environment entry. This does not qualify a hidden-state KV connector.

Resolve environment overrides once on the launch-origin host:

```sh
python -m glm53_setup server freeze --config state/server.toml --output state/launch.json
python -m glm53_setup server plan --config state/server.toml --launch state/launch.json --rank 0
```

The environment's **presence**, including an empty value, overrides TOML. Copy the same frozen JSON to both ranks and supply `--launch` on start/preflight. Rank hosts do not resolve their own allocator environments. A direct start with a local allocator environment requires a frozen manifest. The manifest contains the resolved profile and its lock-bound fingerprint, with no API key. Existing manifests cannot be overwritten by `freeze`.

## All-rail checks and two-rank switch

Each node retains its primary `hca`, `interface`, `local_ip` and `gid_index` (port 1). Optional `additional_rails` lists records with those fields plus `port`. All rails require the same GID index, unique port/NIC/IP associations, an active Ethernet port/link, matching IPv4-mapped RoCE v2 GID, and an IP assigned to that NIC. Comma-separated device strings are rejected in favor of structured records. NCCL receives the complete exact HCA/port list; socket bootstrap stays on the primary interface. These are configuration checks, not multi-rail traffic qualification.

The primary single-rail case also selects port 1 explicitly (`=hca:1`). Omitting a port permits all ports on that HCA, which would exceed the checked scope. See [NVIDIA's NCCL HCA selection contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca).

```sh
python -m glm53_setup cluster switch --config state/server.toml \
  --hosts spark-head spark-peer --checkout /srv/glm53/source \
  --remote-config /srv/glm53/state/server.toml \
  --output records/switch-run
```

Both hosts need the same audited checkout, images and assets; the pre-stop source-identity check rejects a mismatch, so update both checkouts before switching. A running pair records the profile path it was started from in `state/startup-rank<N>.json`, and recovery after a failed switch relaunches the old profile from that recorded path: when the profile file is renamed (1.1.0 moved it to `state/server.toml`), copy rather than move, and delete the old file only after the new pair is confirmed ready. `--remote-config` supplies the Linux path base for relative projector paths; the shared frozen manifest supplies settings. `--ssh-config` can select an SSH config. Static assets/fabric and common source/image/model/profile/allocator identities are checked on both ranks and rechecked before stopping. Weight identities use index hashes, shard sizes and local file identities; this is not a replacement for the original weight-integrity verification. The load-memory gate runs after stop.

Pre-stop failures preserve running containers. Post-stop failures clean up only newly reserved owned attempts, then attempt the recorded old profile. Cleanup uncertainty prevents a competing recovery launch. Results distinguish failure, cleanup and recovery; this is a stop/start procedure with downtime, not an atomic switch. A live rank without a recorded configuration path is rejected before stop because its recovery is not established. Other containers are not stopped.

Read-only SSH observations retry transport failures at most three times. Start/stop/reserve operations are never automatically replayed. If readiness observation remains unavailable, the journal says `readiness-unconfirmed`; the newly owned supervisors retain their memory/deadline guards. Use `cluster resume` with the same `--output`, `--hosts`, `--checkout` and optional `--ssh-config` to recheck identities/assets and readiness without starting again. A known rank failure or readiness deadline still follows cleanup/recovery. The journal records structured failure reasons without command payloads or credentials.

The same protection applies while recovering the old profile: `recovery-readiness-unconfirmed` preserves that recovering pair for `cluster resume`. A successful recovery confirmation records `recovered=true` while the original candidate's status remains failed. Recovery and cleanup errors also retain their structured operational reasons.

On the owned two-host test system, an intentionally mismatched projector hash left both live ranks unchanged (`pre-stop-failure-v69`). A later drill set the new profile's minimum available memory to 999 GiB: static checks passed, the old pair stopped, new startup failed its post-stop memory gate, and the old profile returned to API readiness on both ranks (`rollback-fault-v72`, `recovered=true`). The earlier `v70` recovery-observation failure is retained; it prompted the recovery-side observation fix. These are controlled launch/recovery tests, not a long-duration availability guarantee. Single-rail fabric checks and all three allocator environment states also passed on both hosts; multi-rail traffic and the external KV connector remain unqualified.

## APC history qualification

Omitting the retention key preserves the pinned runtime default **0**: semantic checkpoints/replay boundaries and shared-prefix junctions; it does not mean dense retention. Optional `cache.prefix_cache_retention_interval` forwards the native flag as an explicit experiment. A positive interval equal to the actual scheduler block retains KDA checkpoints at each such boundary. Full-attention retention stays dense; `KpoolTailManager` remains a private one-block ring with no APC publication. This is checkpoint preservation, not a blanket reduction of all groups. The fixed runtime rejects positive intervals that do not align to its resolved scheduler block. The distributed TOML explicitly selects `dense` following the scoped history, pressure, A/B/A and serial integration evidence. Distinguish [distributed defaults](server-configuration.md#distributed-defaults) from the runtime behavior when the key is absent.

The value `"dense"` selects the pinned CLI's `None` value and keeps every checkpoint without requiring a new numeric block width when MTP is toggled. Numeric intervals remain available for controlled experiments. In the measured aligned KDA layout, dense retention and an interval equal to the KDA block both use the native dense mask; final integration still requires its own checks.

The additional matrix covers append, edits/branches at 10/50/90%, alternating conversations, eviction pressure, actual block/pool/MTP boundaries and exact revisits after LPA. Record actual token prefixes and joint H, recomputation, timing, memory and MTP/LPA activity. Native exact controls precede P22 combinations. Retention changes require a separate A/B/A decision; CPU state checks and earlier P22 tests alone do not qualify this extended matrix.

`apc-history` runs the full-model functional matrix on an exclusive serial APC/LPA server, with exact priming and exact/auto/restored requests. Its default one cycle is a functional check, not performance adoption. Supply the actual `--block-tokens` from the running server, the pinned corpus hash, and a fresh output directory; it reads only the corpus validation split. SSE measurements distinguish first output from inter-chunk gaps, which are not individual-token ITL under MTP. The `apc-lpa-fixture --history` option checks shared GPU bytes before full-model use. See [fixture evidence](component-validation.md#additional-history-fixtures).

For a retention A/B/A, `apc-history --timing-only --case-ids edit-50 --repeats 5` selects exact-only one-output prefill samples, with one excluded warmup and cache reset/exact priming before each sample. It records a timing subset, not a completed functional matrix. Use the same input tokens, fixed model runtime and cache budget in all three runs; the report checks the selected retention and joint block alignment against worker metadata. `apc-lpa-fixture --retention-interval N --history` tests the native candidate on the small fixture before its full-model use.

The requirements were informed by Mia PRs #130/#136/#172/#175; no Mia implementation was copied or mechanically rewritten, and no new AGPL dependency was added. Implementation uses this repository and its pinned Apache-2.0 vLLM. EXL3/DFlash, adaptive-k, four-node serving, cable/IP changes and connector implementation are outside this scope.
