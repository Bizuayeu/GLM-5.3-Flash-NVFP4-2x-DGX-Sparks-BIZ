# Image input (vision)

[日本語](vision.ja.md) · [Server configuration](server-configuration.md) · [Document map](README.md)

The distributed profile accepts **text, tool calls and images** at 262,144 input-plus-output tokens (256K). **Video input is disabled and rejected.** This page records how the image profile was reached at 204,800 tokens on 2026-09-15 and extended to 256K on 2026-09-17, what was measured on the reference hosts (MSI EdgeXpert MS-C931, about 121 GiB usable memory each), and what remains unmeasured. Routine-use acceptance and its scope are owned by [SETUP step 6](../SETUP.md#6-qualify-the-full-model); this page holds the image-specific evidence.

## Settings

[Server configuration](server-configuration.md#distributed-defaults) owns the keys; this is the image-related subset of the template.

| Key | Value | Effect |
|---|---|---|
| `runtime.vision` | `true` | Removes `--language-model-only` on both ranks, so the vision tower loads; adds `--limit-mm-per-prompt '{"video": 0}'` |
| `context.max_model_len` | 262144 | The same length as the text-only alternative |
| `cache.kv_cache_memory_bytes` | 3221225472 (3 GiB per rank in the defaults; the [published option](server-configuration.md#the-published-option-against-the-defaults) uses 6 GiB) | The defaults' boot line reports 301,645 tokens, which is 1.15× `max_model_len`: the pool's maximum concurrency in token units, not a cached-conversation capacity ([`server capacity`](server-configuration.md#kv-capacity-and-ram-requirements)) |
| `cache.mm_processor_cache_gb` | 0.1 | `--mm-processor-cache-gb 0.1` instead of vLLM's 4 GiB |
| `resources.reserve_gib` | 3.0 | See the guard arithmetic in [KV capacity and RAM requirements](server-configuration.md#kv-capacity-and-ram-requirements) |

A client must declare image input and must not declare video. For ZCode, set the model's `limit.context` to 262144 and `modalities.input` to `["text", "image"]` ([ZCode model limits](harnesses.md#zcode-permission-modes-model-limits-and-the-existing-file-guard)). The text-only alternative is described in [server configuration](server-configuration.md#distributed-defaults).

## How the settings were chosen

1. **The vision tower loads unquantized.** The checkpoint ships 347 BF16 vision tensors (1.05 GiB). Both quantization exclusion lists name `model.visual*`, and the pinned vLLM builds the tower without a quantization config, so NVFP4 is not misapplied. The MTP metadata view keeps the tower unchanged. The tower is split across the two ranks.
2. **Video is disabled.** At startup vLLM profiles memory by encoding the largest multimodal item once. This checkpoint's video budget is capped at 30,000 tokens (120,000 patches), against at most 8,000 tokens for one image, so a video would set the startup peak. With the video limit at zero, profiling uses one image and a video request returns HTTP 400. The number of images per prompt keeps the vLLM default, because chat harnesses resend earlier images every turn.
3. **The image preprocessing cache is capped at 0.1 GiB.** vLLM keeps one copy in the API server and one in the engine core. Both run only on rank 0, so the 4 GiB default could take up to 8 GiB of the head's memory. An image larger than the cap is processed uncached, with a warning.
4. **The head carries more than the peer, and that cannot be rebalanced.** Rank 0 alone runs the API server and the engine core. Measured proportional set size (Pss) at idle was 2,165 MiB for the API server and 999 MiB for the engine core, against 1,056 MiB for rank 1's headless process; nearly all of it is private anonymous memory. Tensor parallelism has no setting that shifts GPU memory between ranks, vLLM aligns every rank's KV blocks to the smallest rank, the API server and engine core cannot share one process under `vllm serve`, and moving the head to the other host only moves the load.
5. **JIT caches persist** in the mounted runtime cache (`TRITON_CACHE_DIR`, `TILELANG_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`; [storage paths](operations.md#artifact-storage-and-paths)). Written inside the container layer, they were rebuilt at every start (about 2,000 entries), and compiling while serving once took the head from 4.11 to 2.88 GiB available in about 26 seconds and triggered a supervised stop of rank 0. With the caches kept, the head's lowest available memory during startup rose from 3.34 to 4.18 GiB (one start each).
6. **Reserve 3 GiB, KV 3 GiB per rank, 262,144 tokens** since 1.5.0. The first image profile (2026-09-15) had shrunk context to 204,800 and KV and reserve by 0.5 GiB each for the tower and preprocessing. Eight NCCL channels (1.3.1, [numbers](nccl-validation.md#channel-count)) and tracing the peer's supervised stop of 2026-09-16 to a host daemon ([operations](operations.md#supervision-stall-detection-and-warmup)) then raised the head's floor: at 200K the head kept 6.40 GiB during requests, and startup at 256K measured 5.89 GiB against a proceed condition of the reserve plus 1.6 GiB. A 300K profile with 4 GiB KV left 0.3 GiB of margin by the same estimate and was set aside. The reserve is a margin for one transient, not a floor: no kernel or container OOM kill was recorded, the risk below it is a GPU allocation failing inside a worker, and the supervisor samples every 2 seconds and needs about 9 seconds to stop a container.

## Measurements

### Function and regression (final profile)

| Check | HTTP | Seconds | Result |
|---|---|---:|---|
| Synthetic image (number and background colour) | 200 | 3.74 | Correct; 323 prompt tokens, 288 of them image |
| Same image again | 200 | 2.28 | Correct |
| Same question without the image | 200 | 3.43 | Says no image is present; does not guess the number |
| Text of the same length | 200 | 1.71 | Correct |
| Short arithmetic | 200 | 2.62 | Correct |
| Tool call, then tool result | 200 / 200 | 2.07 / 3.21 | Tool call emitted; answer uses the tool result |
| Video | 400 | 0.58 | `At most 0 video(s) may be provided in one prompt` |

Times are single runs measured from a remote client and include SSH overhead. An earlier start of the same profile with the 4 GiB image cache default also passed the image and tool checks, and the [prefix cache check](harnesses.md#zcode-permission-modes-model-limits-and-the-existing-file-guard) restored 13,824 tokens.

### 200K text request

One request built to 199,652 prompt tokens, with a passphrase in the middle, returned the passphrase with `finish_reason: stop` after 506.1 seconds (whole request, including prefill; one run). It ran on the image profile before the reserve and JIT cache changes. The head's lowest available memory during it was 3.31 GiB, and no supervised stop occurred. The ZCode stream idle timeout of 700,000 ms covers this time.

### Memory (final profile)

| | Idle after startup | After one image | During 200K `/tokenize` | After the regression checks |
|---|---:|---:|---:|---:|
| Rank 0 API server Pss (MiB) | 2,165 | 2,201 | 2,220 | 2,329 |
| Rank 0 engine core Pss (MiB) | 999 | 1,005 | 1,005 | 1,007 |
| Rank 0 worker Pss (MiB) | 6,292 | 6,297 | — | 6,300 |
| Rank 1 headless process Pss (MiB) | 1,056 | 1,056 | — | 1,056 |
| Rank 0 available (GiB) | 4.19 | 4.16 | 4.06 | 4.01 |
| Rank 1 available (GiB) | 5.56 | 5.51 | — | 5.53 |

- Lowest available during startup: 4.18 GiB (rank 0) and 5.36 GiB (rank 1). Over the first half hour of serving, including diagnostic sampling, rank 0's lowest was 3.84 GiB, 1.34 GiB above the reserve.
- One 200K `/tokenize` call took 0.22–0.31 seconds and caused no visible peak (one sample).
- The first image request compiled six Triton kernels while serving; the caches kept them. The resulting dip on rank 0 was below 0.1 GiB.
- The API server's anonymous memory grows with use (+164 MiB over these checks). Of its 2.3 GiB, about 1.26 GiB exceeds rank 1's headless process: the heap (+126 MiB), glibc malloc arenas (+255 MiB), other mappings (+138 MiB) and one anonymous mapping of about 624 MiB whose owner was not identified.

### 256K profile (1.5.0, 2026-09-17)

After the switch the same checks passed on the 256K profile: the synthetic image correct, the same question without the image not guessed, text of the same length, the same image again, short arithmetic, a tool call answered from its result, and video rejected with HTTP 400. Both ranks ran with `--max-model-len 262144` and `--kv-cache-memory-bytes 3221225472`, and the warmup ladder's image rung took 1.24 s. The 256K long-context requests and the memory during them are in [measurements on 1.5.0](benchmarks.md#measurements-on-150); the head's lowest available memory there was 5.82 GiB.

## Limits and open items

- Only one small synthetic image was tested. Images near the 8,000-token limit, many images in one session, and image quality in general are unmeasured.
- The MTP draft is text-only; its acceptance rate on image requests is unmeasured. LPA with images is not validated (LPA ships disabled).
- In one ZCode terminal session (2026-09-15, one run), the model downloaded a screenshot with a shell command, read it with the file-read tool and correctly described text that appeared only in the image. Attaching an image directly to a ZCode prompt has not been checked (Claude Code is out of scope, [harnesses](harnesses.md#acceptance-matrix-and-status)); the measurements above used the API directly.
- A request shape seen for the first time can still compile kernels while serving. `server warmup` sends the shapes that were observed compiling (short text, tool call, image, and the long rung set by `generation.warmup_long_tokens`) and records which kernels compiled during the ladder; `cluster switch` runs it after readiness when `generation.warmup = true`. The cache does not remove those warnings on the next start, because the monitor also reports a kernel loaded from the on-disk cache on its first use in a process; the ten reported kernels are loads ([operations](operations.md#supervision-stall-detection-and-warmup)).
- The API server's unidentified 624 MiB mapping and its slow growth over long sessions are not characterized. `MALLOC_ARENA_MAX` could affect at most the 255 MiB in arenas.
- Images with two sequences: one image-plus-prose pair on the published option ([concurrency scope](validation.md#concurrency-scope)). Long-term reliability is not characterized.
