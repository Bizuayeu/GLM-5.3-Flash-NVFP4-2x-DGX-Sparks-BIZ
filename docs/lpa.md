# LPA: Late-prefill approximation — experimental

[日本語](lpa.ja.md) · [Baseline](benchmarks.md)

This experiment predicts the normalized attention inputs of later GLM layers, builds their cache and recurrent state with the existing attention implementation, and skips historical MLP rows. Generation still executes all language layers. The checkpoint is unchanged; only a small auxiliary projector is fitted.

The feature name is **LPA (Late-prefill approximation)**. Configuration uses `[lpa]`; commands are `lpa-fixture`, `lpa-corpus` and `lpa-train`. Rebuild images against the [current image contract](server-configuration.md#current-image-contract); old interfaces are not retained.

The starting point was [Kishida's Qwen3 experiment](https://nowokay.hatenablog.com/entry/2026/09/11/120001). The GLM adaptation has its own state and quality tests.

GLM's sparse MLA needs its latent cache, indexer and incomplete-pool tail. KDA also needs convolution and recurrent state. The adapter retains these update paths and processes each token once. Optional `skip_mla_queries=true` omits only unused historical query evaluation in the pinned reference MLA backend; retained queries keep every candidate. Historical mHC and KDA output computation remain.

The projector uses a learned diagonal scale plus a low-rank residual map. Layer scales differ substantially, so an identity residual prior alone is inadequate for the KDA-to-MLA boundary. Artifacts contain a cut, layer count, representation version and tensors; loading validates their shapes, precision and finite values. Use the same pinned checkpoint and arithmetic as the teacher. Cached projector tensors are reused across requests.

## Operating scope

**LPA and prefix-cache reuse are mutually exclusive, so the distributed template ships `lpa.enabled = false` and this is a batch opt-in.** An approximated request publishes nothing to the shared prefix cache, and suppression continues through the exact tail and decode, because blocks computed after an approximated region still depend on approximated history. A workload that resends a growing prompt (any chat or coding harness) therefore never accumulates a reusable prefix while LPA is on: `break_even_tokens` compares one request's prefill cost and cannot see the reuse every later request forfeits. Enable LPA for a long input that is processed once, not for a conversation.

LPA runs together with MTP, fused unpack and asynchronous index checks; the evidence is [P18](benchmarks.md#serial-integration-of-mtp-lpa-fused-unpack-and-async-checks-p18).

- Eager, text-only, TP=2, one active sequence and one controlling client. Sequence-parallel MoE and concurrent controllers are unsupported. Without APC, MTP k=1, 2 or 3 requires explicit `allow_mtp=true`; the [server TOML](server-configuration.md) wires this automatically and refuses k=4 and 5 with LPA. k=2 was checked on the four-layer fixture only ([component validation](component-validation.md#apc-first-lpa-cache-isolation-p22)), and because the LPA workers are baked into the image, it needs an image built since k=2 was allowed. APC uses the separate scheduler-integrated P22 path described below.
- A configurable final prompt window is computed normally. Fully protected short prompts use the ordinary path without loading or running a projector.
- The auxiliary model changes historical state. Running all decode layers does not restore exact target-model probabilities.
- This is an experimental worker extension, outside the routine-use acceptance of the serving profile. Keep its development RPC on loopback.

### APC and shared-state provenance

LPA changes cached KV and KDA state. A cache produced with approximation must not enter an LPA-off request or a different projector configuration. The manual `lpa_configure` RPC therefore rejects prefix caching: it cannot establish the scheduler's cache-publication boundary.

With prefix caching on, P22 restores the exact prefix first and approximates only the uncached remainder, publishing nothing to the shared cache from the first approximation on; the [design contract](apc-lpa-design.md) owns that contract and its status. Capture/oracle RPCs are not enabled with APC.

## Download the trained projector

The tested cut32 projector is available as a separate [GitHub Release asset](https://github.com/Bizuayeu/GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ/releases/tag/lpa-cut32-v1), under **Apache-2.0**. [config/lpa-projector.lock.json](../config/lpa-projector.lock.json) owns the download URL, exact size/hash, format, teacher identity and training provenance. The NVIDIA checkpoint remains a separate download. No retraining is needed to use this projector.

On **each Linux host**, from the checkout:

```sh
mkdir -p state/lpa
curl --fail --location --output state/lpa/glm53-lpa-cut32-v1.tar.gz https://github.com/Bizuayeu/GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ/releases/download/lpa-cut32-v1/glm53-lpa-cut32-v1.tar.gz
curl --fail --location --output state/lpa/SHA256SUMS https://github.com/Bizuayeu/GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ/releases/download/lpa-cut32-v1/SHA256SUMS
(cd state/lpa && sha256sum --check SHA256SUMS)
```

Continue only if the archive checksum passes:

```sh
tar -xzf state/lpa/glm53-lpa-cut32-v1.tar.gz -C state/lpa
python -c 'import hashlib,json,pathlib; p=pathlib.Path; m=json.loads(p("config/lpa-projector.lock.json").read_text()); f=p("state/lpa")/m["artifact"]/m["file"]; assert f.stat().st_size == m["bytes"]; assert hashlib.sha256(f.read_bytes()).hexdigest() == m["sha256"]; print("Projector verified:", f)'
```

Downloading the asset does not enable LPA or restart a server. Enabling it is the explicit operator step below.

### Enable LPA in the server profile

Enable LPA only for the [supported workload](#operating-scope): a long input processed once, not a conversation. On both hosts, edit the same `state/server.toml`:

```toml
[runtime]
lpa_image = "sha256:<image ID verified on both hosts>"   # carries GLM53_LPA_API=2
vision = false          # LPA's measured scope is text-only
fa2_attention = false   # the FA2 prefill path excludes LPA

[lpa]
enabled = true
cut = 32                # matches the distributed projector; keep it
projector = "lpa/glm53-lpa-cut32-v1/projector.pt"   # relative to this TOML, or absolute
projector_sha256 = "<sha256 from config/lpa-projector.lock.json>"
```

- **Image**: `lpa.enabled = true` makes the launcher select `runtime.lpa_image` instead of `reference_image`. Preflight requires the `GLM53_LPA_API=2` marker in that image, and `GLM53_APC_LPA_API=1` as well while `cache.prefix_caching` stays on (the template default, which selects the [APC-first path](server-configuration.md#lpa-with-prefix-caching)). An image built from current source against the [current image contract](server-configuration.md#current-image-contract) carries both; check with `docker image inspect <id>` and use the same ID on both hosts.
- **Projector**: preflight recomputes the SHA-256 of the file named by `[lpa].projector` and rejects a mismatch with `projector_sha256`. Keep `cut = 32`, `tail` and `break_even_tokens` at the template values, which are the measured settings for this projector.
- **FA2**: the launcher refuses `runtime.fa2_attention` with LPA, so LPA runs on the reference attention path, without the FA2 prefill speedup ([server configuration](server-configuration.md#attention-cache-and-checkpoint)).
- **Text-only**: see the [text-only alternative](server-configuration.md#distributed-defaults) for its validated reserve.
- **Check, then switch**: run `python -m glm53_setup server preflight --config state/server.toml --rank N` on each host and confirm `projector_sha256`, `lpa_worker` and `image_id` pass. Any of these edits changes the profile fingerprint, so a running pair needs the normal [two-rank switch](launch-safety.md#all-rail-checks-and-two-rank-switch); `server ask` refuses a profile that no longer matches the running server.
- **Per request**: while LPA is enabled, a request can still compute normally with `"vllm_xargs": {"glm53_lpa_mode": "off"}` to prime the shared prefix cache; see [server configuration](server-configuration.md#lpa-with-prefix-caching).

To turn LPA off again, set `enabled = false` and switch; the projector keys may stay in the file.

### Model card and training provenance

This is a diagonal scale plus shared low-rank basis and per-layer residual maps, fitted by ridge regression to the frozen teacher's attention inputs. The cut layer uses its own input; learned maps cover layers 33–44. The artifact contains fitted tensors and scalar metadata, without corpus text, teacher activation captures, credentials or site settings. Serialized metadata, tensor shapes, finite values and SHA-256 were checked before distribution; the bytes match the projector used in the measurements below.

Training uses the pinned LLM-jp Corpus v3 Japanese/English Wikipedia and filtered C++ subsets listed in the lock. Fit uses only the training split; validation selects the candidate and test documents are held out. Source sampling was bounded rather than a full-shard download, so full source-shard checksums were not verified. Dataset licenses and attribution are preserved separately from the [projector license](licensing.md#lpa-projector). The dataset is not relicensed or bundled. The small evaluation below does not establish general quality, absence of memorization or suitability for other checkpoints/precisions.

## Reproduce the components

Use the same fixed reference image as the teacher for GPU commands; CLI help and corpus sampling do not require Torch.

```sh
python -m glm53_setup lpa-corpus --output records/corpus-ja --documents 512
python -m glm53_setup lpa-corpus --subset en-wiki --output records/corpus-en --documents 128
python -m glm53_setup lpa-corpus --subset code --shard 300 --output records/corpus-code --documents 128
python -m glm53_setup lpa-fixture --fixture /fixture --output /out/oracle --cut 0 --skip-mla-queries --lengths 3 4 5 127 128 129 511 512 513 8705
python -m glm53_setup lpa-train --captures /out/teacher --output /out/projector --cut 32 --rank 256 --ridge 0.001
```

The sampling sizes and rank/ridge values are pilot parameters, not established optima; `--cut 32` reproduces the distributed projector, and the pilot first fitted cut 40 from the same capture before cut 32 was selected. The sampler pins the LLM-jp corpus revision, retains source metadata, limits compressed bytes read and partitions normalized document hashes into train/validation/test. The code sample uses the dataset's per-repository license metadata to retain MIT/Apache/BSD/ISC entries. It does not apply the repository's Apache license to all corpus data. Preserve the emitted attribution and subset licenses; see the [LLM-jp corpus README](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3).

## Teacher collection and experimental control

The reference-image server needs `--worker-extension-cls glm53_setup.runtime.lpa.LPAWorkerExtension` and `VLLM_SERVER_DEV_MODE=1` for the private `/collective_rpc` endpoint. It accepts the named `lpa_configure` and `lpa_report` worker methods. No request may enter between configuration and the controlled request. Use the server's tokenizer and identical template settings to determine the actual prompt length.

`lpa_configure` takes `mode`, `cut`, `prompt_length`, `tail`, and, for prediction, `predictor_path`. Modes are `off`, `capture`, `oracle`, `identity`, and `predict`. The response distinguishes requested mode from effective mode. `capture` records teacher attention inputs; `oracle` reuses that exact prompt's capture to test state injection. `identity` is a diagnostic baseline, not a trained approximation.

`lpa_report(output=...)` writes `rank-N/layer-L.pt` tensors under a fresh directory. The trainer expects a completed collection `result.json` containing `cut` and `cases`, with each case carrying `id`, `split`, and `prompt_tokens`; tensors live at `<id>/rank-0/layer-L.pt`. Keep complete-document splits even when selecting interior code windows. The same wider capture can train a later cut. Fit on train only, select with validation, and reserve test for final evaluation.

`profile=true` records per-layer and attention/MLP CUDA event spans. Use a single output token when isolating prefill, and keep instrumentation off for final wall-time comparisons. Communication inside those modules is included in their spans, not measured as an independent term.

Query omission is request-scoped, disabled by default, and requires the fixed reference backend. A layout mismatch or an unexpected backend call fails explicitly. Cache creation remains upstream of the read-only query calculation. The oracle fixture verifies both normal restoration and the actual number of skipped queries.

## Evidence and limits

The initial four-layer oracle experiments covered pool, convolution, chunk and cache-block boundaries. Full-model oracle trials also covered inputs through 8,192 tokens. Token/text agreement and numerical drift were recorded separately. The one-token fixture control was unstable even without approximation and is not a qualified case.

A pilot projector fitted from LLM-jp Japanese/English Wikipedia and permissively labeled C++ samples supports the experimental cut-32 profile: 13 later layers, a 512-token normal tail, and reference MLA query omission. A cut-24 candidate returned an incorrect long-context lookup value and was rejected.

| Pre-tokenized input | Ordinary path, mean of A/A-restored medians | Approximate path median | Latency reduction |
|---:|---:|---:|---:|
| 2,048 tokens | 6.174 s | 5.118 s | 17.1% |
| 8,192 tokens | 24.763 s | 19.409 s | 21.6% |

These are one-output-token requests, five measurements per condition with warmup excluded, on the serial TP=2 reference profile. Inputs came from validation documents. Configuration/tokenization were outside the timing window. This measures prefill/first-token response latency, not faster decode or a universal halving of latency.

All six active long-context/long-code functional cases and the long-context tool round-trip passed. Eight untouched documents yielded all eight lookup values correctly; one response added an explanation, so initial strict-format success was 7/8. Both paths passed four repeats of that affected case. Streaming, recovery after closing an active stream, and a cold restart followed by a long-context query passed. The baseline also exhibited format variability; the small sample is not a statistical non-inferiority guarantee.

Raw data, teacher captures and site-specific controllers remain private under `records/`. The selected projector is distributed separately as described above. The ordinary deployment/harness qualification gate is unchanged.
