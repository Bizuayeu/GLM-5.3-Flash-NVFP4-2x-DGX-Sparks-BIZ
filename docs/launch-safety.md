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

Both hosts need the same audited checkout, images and assets; the pre-stop source-identity check rejects a mismatch, so update both checkouts before switching. A running pair records the profile path it was started from in `state/startup-rank<N>.json`, and recovery after a failed switch relaunches the old profile from that recorded path: when the profile file is renamed (1.1.0 moved it to `state/server.toml`), copy rather than move, and delete the old file only after the new pair is confirmed ready. `--remote-config` supplies the Linux path base for relative projector paths; the shared frozen manifest supplies settings. Once the new pair is complete, both ranks also write the `--config` text to that path, so commands that read the file on a rank (`server agreement`, `server mojibake`, benchmarks) see the profile that is running. Each rank accepts only text that parses to the profile it was started with, keeps a differing old file beside it as `<name>.bak-<UTC time>-<first 8 of its fingerprint, or unparsed>` and replaces the file in one rename. Nothing is written when the switch fails or recovers the old pair, and a failed write is recorded under `config` in the journal without rolling the pair back. `--no-send-config` leaves the remote files alone; `cluster resume` writes them only when it is given `--config` and finds the new pair ready. `--ssh-config` can select an SSH config. Static assets/fabric and common source/image/model/profile/allocator identities are checked on both ranks and rechecked before stopping. Weight identities use index hashes, shard sizes and local file identities; this is not a replacement for the original weight-integrity verification. The load-memory gate runs after stop. The [foreign-GPU check](operations.md#full-model-launch-checks) runs at both points: before stop it ignores the running old pair, which carries the launcher's label, and after stop it refuses to start while another GPU container runs.

Pre-stop failures preserve running containers. Post-stop failures clean up only newly reserved owned attempts, then attempt the recorded old profile. Cleanup uncertainty prevents a competing recovery launch. Results distinguish failure, cleanup and recovery; this is a stop/start procedure with downtime, not an atomic switch. A live rank without a recorded configuration path is rejected before stop because its recovery is not established. Other containers are not stopped; a GPU container outside the launcher fails the pre-stop check, so nothing is stopped for it. On 2026-09-17 the reference pair switched with the check in place while its old pair was running (the pre-stop check passed on both ranks), and a component probe container on the third host was reported as foreign while it ran.

The running pair and the new pair are not held to the same image requirements. The switch prepares the running pair, and reserves its restart after a failure, as a recovery target: it must be restorable as it was launched, so it keeps the MoE order marker of its own image while the new launch needs marker 2 ([launch checks](operations.md#full-model-launch-checks)). The mark travels beside the frozen launch, not inside it, so the recovered pair has the fingerprint it had before. What a recovery target was allowed appears under `warnings` in the saved `recovery_assets`.

Read-only SSH observations retry transport failures at most three times. So do the two mutations a rank makes harmless to replay: `stop` is idempotent, and `start` answers `replayed` for an attempt already in flight instead of launching a second supervisor (a finished or cancelled attempt still refuses). By the time a switch starts the new ranks it has stopped the old ones, so one dropped connection at that step used to cost a recovery. Reserving an identity, installing the profile text and the warmup ladder are never replayed. If readiness observation remains unavailable, the journal says `readiness-unconfirmed`; the newly owned supervisors retain their memory/deadline guards. Use `cluster resume` with the same `--output`, `--hosts`, `--checkout` and optional `--ssh-config` to recheck identities/assets and readiness without starting again. A pair that resume finds ready then gets what a completed switch gives it: the warmup ladder, and with `--config` the profile text written on both ranks; as in a switch, a failure of either is recorded and leaves the pair running. A recovered old pair gets neither. A known rank failure or readiness deadline still follows cleanup/recovery. The journal records structured failure reasons without command payloads or credentials.

The same protection applies while recovering the old profile: `recovery-readiness-unconfirmed` preserves that recovering pair for `cluster resume`. A successful recovery confirmation records `recovered=true` while the original candidate's status remains failed. Recovery and cleanup errors also retain their structured operational reasons.

On the owned two-host test system, an intentionally mismatched projector hash left both live ranks unchanged (`pre-stop-failure-v69`). A later drill set the new profile's minimum available memory to 999 GiB: static checks passed, the old pair stopped, new startup failed its post-stop memory gate, and the old profile returned to API readiness on both ranks (`rollback-fault-v72`, `recovered=true`). The earlier `v70` recovery-observation failure is retained; it prompted the recovery-side observation fix. These are controlled launch/recovery tests, not a long-duration availability guarantee. Single-rail fabric checks and all three allocator environment states also passed on both hosts; multi-rail traffic and the external KV connector remain unqualified.

### After a switch: the decode check

A new launch is checked, not assumed: on the four-layer fixture launches fell into two numerical states, and on the pair one launch in six computed in another state on 2026-09-22 with the kernel tables unchanged ([measurements on 1.9.0](benchmarks.md#six-launches-of-the-new-image-the-same-completions-five-times-different-once)). After every switch, on rank 0, run `tools/decode_check.py` once per task type with `TOKENS_OUT` set, and save both ranks' container logs before the next switch discards them:

```sh
for kind in prose count code; do
  PROMPT_KIND=$kind SAMPLES=3 TOKENS_OUT=records/<run>/tokens-$kind.json \
    python3 tools/decode_check.py > records/<run>/decode-$kind.jsonl
done
docker logs <rank0 container> 2>&1 | gzip > records/<run>/logs-rank0.txt.gz   # rank 1 likewise, on its host
```

Run the decode check with nothing else in flight: on a two-sequence profile a request that shares its steps with another request gets a different completion, and a different one on each run ([measurements on 1.10.2](benchmarks.md#measurements-on-1102)). Within a launch the three samples must agree (`distinct_completions` 1). Across launches compare `completion_sha256` with the previous launch of the same profile. When they differ, keep the record: `tools/decode_divergence.py` prints the first diverging token between the two `tokens-*.json` files (one tie-break flip late in the text, or an early systematic drift), and the two logs are the only per-launch evidence there is. Speed and acceptance length inside the profile's usual spread say the difference is a tie, not a fault.

When the profile carries `validation.memory_probe = true`, take the weight digest after the switch, before the decode check: `python3 tools/weight_digest.py --output records/<run>/weights.json --reference records/<previous run>/weights.json` fingerprints every parameter and buffer on both ranks as loaded and names the tensors that differ from the previous launch (exit status 1). A launch in another numerical state with an identical digest computed differently from the same bits; one with a differing digest loaded differently, and the record says where.

## APC history qualification

Omitting the retention key preserves the pinned runtime default **0**: semantic checkpoints/replay boundaries and shared-prefix junctions; it does not mean dense retention. Optional `cache.prefix_cache_retention_interval` forwards the native flag as an explicit experiment. A positive interval equal to the actual scheduler block retains KDA checkpoints at each such boundary. Full-attention retention stays dense; `KpoolTailManager` remains a private one-block ring with no APC publication. This is checkpoint preservation, not a blanket reduction of all groups. The fixed runtime rejects positive intervals that do not align to its resolved scheduler block. The distributed TOML explicitly selects `dense` following the scoped history, pressure, A/B/A and serial integration evidence. Distinguish [distributed defaults](server-configuration.md#distributed-defaults) from the runtime behavior when the key is absent.

The value `"dense"` selects the pinned CLI's `None` value and keeps every checkpoint without requiring a new numeric block width when MTP is toggled. Numeric intervals remain available for controlled experiments. In the measured aligned KDA layout, dense retention and an interval equal to the KDA block both use the native dense mask; final integration still requires its own checks.

The additional matrix covers append, edits/branches at 10/50/90%, alternating conversations, eviction pressure, actual block/pool/MTP boundaries and exact revisits after LPA. Record actual token prefixes and joint H, recomputation, timing, memory and MTP/LPA activity. Native exact controls precede P22 combinations. Retention changes require a separate A/B/A decision; CPU state checks and earlier P22 tests alone do not qualify this extended matrix.

`apc-history` runs the full-model functional matrix on an exclusive serial APC/LPA server, with exact priming and exact/auto/restored requests. Its default one cycle is a functional check, not performance adoption. Supply the actual `--block-tokens` from the running server, the pinned corpus hash, and a fresh output directory; it reads only the corpus validation split. SSE measurements distinguish first output from inter-chunk gaps, which are not individual-token ITL under MTP. The `apc-lpa-fixture --history` option checks shared GPU bytes before full-model use. See [fixture evidence](component-validation.md#additional-history-fixtures).

For a retention A/B/A, `apc-history --timing-only --case-ids edit-50 --repeats 5` selects exact-only one-output prefill samples, with one excluded warmup and cache reset/exact priming before each sample. It records a timing subset, not a completed functional matrix. Use the same input tokens, fixed model runtime and cache budget in all three runs; the report checks the selected retention and joint block alignment against worker metadata. `apc-lpa-fixture --retention-interval N --history` tests the native candidate on the small fixture before its full-model use.

The requirements were informed by Mia PRs #130/#136/#172/#175; no Mia implementation was copied or mechanically rewritten, and no new AGPL dependency was added. Implementation uses this repository and its pinned Apache-2.0 vLLM. EXL3/DFlash, adaptive-k, four-node serving, cable/IP changes and connector implementation are outside this scope.
