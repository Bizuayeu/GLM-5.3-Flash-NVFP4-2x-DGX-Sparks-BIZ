# Harness integration and acceptance

[日本語](harnesses.ja.md) · [Setup](../SETUP.md) · [Licensing](licensing.md) · [Validation](validation.md)

A harness is the client that manages conversation, files, tool execution, history and approvals; it is separate from the GPU inference server. **This document is the single source for harness connection design, the acceptance matrix and its run status.** Other documents point here instead of restating status. Full-model TP=2 startup and the basic API group are prerequisites for every client case.

## Connection design

| Route | Connection | Decision |
|---|---|---|
| Basic API checks | Local vLLM Chat Completions / Messages | Separate server defects from client integration defects |
| ZCode | Custom Provider → SSH tunnel → OpenAI-compatible vLLM | npm `zcode-app-cli` is the accepted route (2026-09-22); Desktop BLOCKED |
| Claude Code CLI | Anthropic format → SSH tunnel → vLLM Messages | Not pursued (skipped by decision, 2026-09-22) |

ZCode's [official site](https://zcode.z.ai/en) identifies it as a GLM harness and its [configuration guide](https://zcode.z.ai/en/docs/configuration#custom-providers-anthropic--openai-compatible) describes custom compatible services; neither establishes compatibility with this NVFP4 TP=2 configuration. The pinned vLLM serves `/v1/messages` and `/v1/messages/count_tokens` directly ([Claude Code guide](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/serving/integrations/claude_code.md), [router](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/entrypoints/anthropic/api_router.py)); [Z.ai's Claude Code instructions](https://docs.z.ai/devpack/tool/claude) concern its hosted service, and [Anthropic's gateway guidance](https://code.claude.com/docs/en/llm-gateway) excludes support for routing to non-Claude models. Keep vendor support, technical compatibility and applicable terms distinct.

## ZCode distributions

"ZCode" is not one artifact. Acceptance and outbound-traffic facts differ by distribution, so this document names them separately.

| Distribution | What it is | Role here |
|---|---|---|
| Official Desktop GUI | Electron application from the [official installer](https://zcode.z.ai/en/docs/install); providers are added in Model Settings | Not available (BLOCKED, [feedback #270](https://github.com/zai-org/feedback/issues/270)); not the accepted route |
| Official Desktop bundled CLI | `resources/glm/zcode.cjs` inside the Desktop install | Not usable interactively in the inspected version; headless `--prompt` works (see status) |
| npm `zcode-app-cli` | Unofficial [terminal wrapper](https://github.com/kingsword09/zcode-cli) that vendors the ZCode runtime and adds its own TUI (MIT for its own code; ZCode runtime keeps upstream terms) | Accepted route (2026-09-22) |

## Connection settings

These are the settings used on the accepted route (run `20260915-harness-h-sandbox`). They are version-bound, so recheck them after a client update. Do not bypass `server preflight`. Record client version/distribution hash, server source/image/revision and effective arguments first. Use the API port from your server TOML; the example template uses 8893 on loopback.

Open a dedicated PowerShell for an SSH tunnel (`node-a` is illustrative; add your existing `-F` configuration when needed):

```powershell
ssh -N -L 127.0.0.1:8893:127.0.0.1:8893 node-a
```

Use credentials issued for the local service if authentication is configured. Do not reuse Z.ai or Anthropic cloud credentials for the local experiment.

**Deny the harness write access to its own configuration directory during acceptance** (`.zcode` under the user profile). A harness running with build permission can edit its own configuration on request, report success, and then fail to start; a client-side schema rejection surfaces as an unrelated error such as a missing model configuration. Use the client's tool denylist for that path and keep a copy of the working configuration. Under `yolo` the denylist is not consulted; the [existing-file guard](#zcode-permission-modes-model-limits-and-the-existing-file-guard) covers that directory instead.

The npm CLI reads its configuration from `~/.zcode/cli/config.json`. Add an OpenAI-compatible provider with `baseURL` `http://127.0.0.1:8893/v1` and the model ID `glm-5.3-flash-nvidia` (entry `provider.<id>.models.<model>`), and check the actual request path for a duplicated `/v1`. Supply the local service's key, or a non-secret test value only if an unauthenticated loopback endpoint requires a nonempty field; that value adds no security.

Declare text, tool and image input to match the server (`runtime.vision = true` in both example profiles; in ZCode, `modalities.input = ["text", "image"]`), and never video, which the server rejects. Do not enable cloud fallback, external integrations or additional agents by default. Point both the main and the helper ("lite") model at the local served ID. Set the model's `limit.context` to the server `max_model_len`, keep `limit.output` at 32000, and raise `modelStream.idleTimeoutMs` above the longest prefill you expect (see [model limits](#zcode-permission-modes-model-limits-and-the-existing-file-guard)). Confirm the selected model and actual destination. The UI locale accepts only the values the client documents (`zcode --help` lists them); an unsupported value invalidates the whole configuration file.

## ZCode permission modes, model limits and the existing-file guard

The facts below come from a static read of the Desktop-bundled runtime (`resources/glm/zcode.cjs`, Desktop 3.11.2, runtime 0.16.5, 2026-09-14), the runtime version (0.16.5) the npm distribution also vendors. They are version-bound; recheck after any client update. They do not close an acceptance case.

**Modes.** The configuration enum is `plan`, `build`, `edit`, `yolo`, `auto`. Two normalizers map Claude Code names onto it: the session normalizer maps `bypassPermissions`/`dontAsk` to `yolo` and `acceptEdits` to `edit`; the automation normalizer maps `acceptEdits`/`autoEdit`/`default`/`auto` to `build`. `auto` is reserved and unimplemented: it denies every tool, so `permission.allowMediumRiskInAuto` has no effect. Headless `--prompt` defaults to `yolo`.

**Tool risk descriptors.** `Read` is low risk with no side effects. `Write` and `Edit` share one permission (`edit`, medium risk, workspace scope). `Bash` is high risk with system scope. There is no delete or rename tool; deletion goes through `Bash`. Nothing in the descriptors distinguishes creating a file from changing one.

**Decision order.** Plan-mode transitions → tools that need user interaction → `yolo` allow → `auto` deny → `disallowedTools` → project `deny` rules → project `ask` rules → plan-mode check → project `allow` rules → `allowedTools` → `edit` mode (allow `Write`/`Edit`) → `build` mode (allow read-only tools; ask for critical risk, for high risk unless `autoApproveHighRisk`, and for any other side effect). Consequences: `build` asks before every write and shell command, `edit` is `build` plus unattended file edits, and `yolo` is evaluated **before** project rules and `disallowedTools`, so path rules cannot narrow it.

**Hook merge.** A `PreToolUse` hook returning `hookSpecificOutput.permissionDecision` is merged with the policy result: `deny` always wins, `ask` turns a policy `allow` (including `mode.yolo`) into a prompt (`hook.PreToolUse.ask`), and `allow` turns a policy `ask` into an allow. The main executor path routes a hook `ask` to the permission broker without the project-rule filter. This is the only place where `yolo` can be narrowed. The hook receives JSON on stdin with `tool_name`, `tool_input` (`file_path` absolute and `content` for `Write`/`Edit`; `command` for `Bash`), `cwd`, `permission_mode` and `session_id`.

**Guard.** [examples/zcode-hooks/exists-guard.cjs](../examples/zcode-hooks/exists-guard.cjs) implements "run in `yolo`, but ask before changing what already exists": `Edit` always asks; `Write` asks when the target exists and allows a new file; any `Write`/`Edit` under the client's own `.zcode` directory asks; `Bash` asks when the command matches a destructive pattern (`rm`, `mv`, `cp`, `patch`, `Remove-Item`/`Copy-Item`, `git reset`/`clean`/`apply`, redirection except to `/dev/null` or `NUL`, and similar) and allows otherwise; other tools get no opinion. [examples/zcode-hooks/config.hooks.example.json](../examples/zcode-hooks/config.hooks.example.json) shows the `hooks.events.PreToolUse` entry. Install, verification and limits: [the hook's README](../examples/zcode-hooks/README.md).

**Verified behavior (`20260914-zcode-exists-guard`, Desktop-bundled CLI in headless `--prompt`, local model).** A `Write` to a new file was allowed and the file was created. A `Write` to an existing file became `ask`; headless mode has no permission client, so the tool call was rejected ("No permission client configured") and the file was unchanged. A non-destructive `Bash` command (`ls -la .`) was allowed and answered; `rm doomed.txt` became `ask`, was rejected the same way and the file remained (an earlier attempt at these two failed on `ECONNRESET` and is kept in the record). In the TUI the same `ask` is a prompt.

**Model limits and the compaction budget.** `limit.context` becomes the client's context window (default 200000). `limit.output` is sent as `max_tokens` on every request (default 32000; a loopback tap on one headless prompt showed `max_tokens: 32000`); vLLM counts the template's always-present thinking block against it, so a small value truncates reasoning plus answer with `finish_reason: length`. Auto-compaction (`preflight-v1`) starts at `limit.context` − min(`limit.output`, 21000) − 13000, so a leftover small entry compacts far too early (32768/4096 at about 15.7K tokens). Because vLLM rejects a prompt plus `max_tokens` above `max_model_len`, `limit.output` must stay below 34000 when `limit.context` equals `max_model_len`, at any server length: keep 32000, not the model's nominal 64K–128K, and change `limit.context` together with the server's `max_model_len`. Whether an existing session keeps the model definition it was created with was not verified; start a new session after changing limits.

**Prefix cache and the harness cache display.** ZCode's cache display needs two things. (1) The server must cache the request: LPA is off in both example profiles, so plain requests cache. If a batch job enables LPA on the same server, an approximated request publishes nothing to the shared cache, so send `"vllm_xargs": {"glm53_lpa_mode": "off"}` from the harness; in ZCode that is `provider.<id>.models.<model>.options.extra_body.vllm_xargs.glm53_lpa_mode = "off"`, observed on the wire as a top-level `vllm_xargs` field. (2) The server must report the hits: ZCode computes the rate from `usage.prompt_tokens_details.cached_tokens`, which vLLM returns only with `api.prompt_tokens_details = true` (on in both examples). Reuse comes in whole 4,608-token blocks, less the last matched block under MTP, so short exchanges and the first turn of a session show 0 ([measurements on 1.13.0](benchmarks.md#measurements-on-1130)). `python tools/check_prefix_cache.py --model <served id>` checks both conditions and names the one that fails.

**Stream idle timeout.** `modelStream.idleTimeoutMs` (default 600000 in the bundle) counts from the request until the first SSE event, and the local server streams nothing during prefill. On expiry the client aborts, retries with the same prompt, and adds 30 seconds per retry, up to 11 attempts; a long-context turn whose prefill exceeds the value never completes. The tested configuration uses 700000, chosen on 2026-09-14 from a 606-second extrapolation to 256K on the profile of that date; it covers the longest requests measured since, the 256K capacity request ([measurements on 1.6.0](benchmarks.md#measurements-on-160)) and two ~200K requests together ([measurements on 1.10.2](benchmarks.md#measurements-on-1102)). `0` disables the timer at the cost of stall detection. `network.timeout` is wired to the auxiliary HTTP client; whether it also bounds model streaming was not confirmed.

## Outbound traffic of harness clients

Local inference routing and telemetry suppression are separate settings. The table records what a static read of each distribution's bundle shows; it is not a capture of live traffic, which H-09 covers.

| Distribution (inspected version) | Model-execution traces | Other outbound | Switch |
|---|---|---|---|
| npm `zcode-app-cli` (3.11.2-24, runtime 0.16.5) | Sent only when `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set in the environment; no destination is hardcoded | Update check, hosted model-catalog refresh, official plugin marketplace fetch | `ZCODE_MODEL_TELEMETRY_ENABLED=0` (also `false`, `off`, `disabled`); `ZCODE_DISABLE_UPDATE_CHECK=1`; `ZCODE_DISABLE_MODEL_CATALOG_REFRESH=1`; `plugins.enabled=false` in the client config |
| Official Desktop (3.11.2) | An OTLP trace endpoint under `*.cn-beijing.log.aliyuncs.com` and an event-report endpoint under `zcode.z.ai/api/v1/` are hardcoded; inherited `OTEL_*` and `ZCODE_MODEL_TELEMETRY_ENABLED` are removed from the child environment before the hardcoded values are injected | Update feed under `cdn-zcode.z.ai` | None found in settings or environment. Suppression requires not running Desktop or blocking those hosts at the network level, which is operator policy outside this repository |

Set the environment switches in the process that launches the client and restart existing processes. Network-level blocking is not configured by this repository; if you apply it, scope it to the client executables and record it with the run.

For H-09, snapshot TCP connections of the client and its children from another PowerShell right after startup and during inference, recording each distribution separately:

```powershell
$zcodeProcesses = @(Get-CimInstance Win32_Process)
$zcodeIds = @($zcodeProcesses | Where-Object {
    $_.Name -eq 'ZCode.exe' -or
    ($_.Name -eq 'node.exe' -and $_.CommandLine -match 'zcode')
} | Select-Object -ExpandProperty ProcessId)
do {
    $previousCount = $zcodeIds.Count
    $zcodeIds = @($zcodeIds + @($zcodeProcesses | Where-Object {
        $_.ParentProcessId -in $zcodeIds
    } | Select-Object -ExpandProperty ProcessId) | Sort-Object -Unique)
} while ($zcodeIds.Count -gt $previousCount)
Get-NetTCPConnection -ErrorAction SilentlyContinue |
    Where-Object { $_.OwningProcess -in $zcodeIds } |
    Select-Object OwningProcess, State, RemoteAddress, RemotePort
```

A snapshot misses short-lived connections, UDP and DNS; an empty result is not proof of silence, and an address alone does not identify a domain or payload.

**Recorded sample (2026-09-14, `20260914-zcode-cli-tcp-sample`):** with the npm distribution launched under the switches above (no OTLP endpoint, telemetry flag off, update and catalog refresh disabled, no Z.ai login), one headless prompt was answered by the local model while TCP connections of the client's node processes were sampled every 0.5 s for 20 s. The only remote endpoint observed was the loopback SSH tunnel to the local vLLM API. This is supporting evidence that prompt and file content stay on the local route for that distribution; it is one prompt, TCP only, with `plugins.enabled` still true, and it does not exercise the cloud-fallback part of H-09 or say anything about official Desktop.

## Reasoning profile for acceptance

For this fixed GLM template, keep thinking active: it always starts an assistant thinking block and does not read an off switch, so do not send `thinking=false` / `enable_thinking=false`. The [model card](https://huggingface.co/zai-org/GLM-5.3-Flash) documents `reasoning_effort=low/high/max`, defaulting to max, and recommends `clear_thinking=true` for chat; low effort is a chat test profile, not a claim to reproduce max-effort leaderboard scores. In ZCode, `/effort <level>` (alias `/variant`; `/effort list` shows the choices) switches the session effort; IDs matching `glm-5.3` offer `low`, `high`, `max`, while `medium` and `xhigh` exist only for GPT and Qwen IDs. The level reaches the provider as `reasoning_effort`, and the pinned GLM chat template accepts only `low` and `high`, treating any other value, an omitted field included, as `max`. At `max` thinking has no practical bound, so a normal coding turn spends most of its `max_tokens` and wall time there; use `high` or lower for everyday work. The matching parser/template leak is tracked in [vLLM #54744](https://github.com/vllm-project/vllm/issues/54744); [PR #54825](https://github.com/vllm-project/vllm/pull/54825) was open on 2026-09-26, so do not assume an arbitrary image includes it. The pinned vLLM `385dce36` already contains the XGrammar and speculative-decoding fixes [vLLM #53046](https://github.com/vllm-project/vllm/pull/53046) (`c6e19b3`) and [#52805](https://github.com/vllm-project/vllm/pull/52805) (`12f64b3`); a GitHub compare shows it 0 behind both.

Judge normal acceptance by final answers, structured tool calls, tool results and approval boundaries. Keep exact reasoning-text/token replay and cross-batch bitwise comparisons as separate numerical diagnostics. Preserve mismatches; do not treat every such mismatch as task failure or prove model correctness solely from matching final answers. Use one active sequence for golden checks and assess parallel throughput/quality separately.

## Acceptance matrix and status

Status values: PASS, PARTIAL (some criteria met, listed), BLOCKED (cannot run; evidence and alternative recorded), NOT RUN. Run the API group first, then every shared H case with real requests and artifacts. Source availability or a truncated GPU fixture cannot replace actual requests. Status as of 2026-09-22 (run 2026-09-15; decision 2026-09-22); run IDs refer to private `records/`.

**Run `20260915-harness-h-sandbox` (2026-09-15).** All eleven shared H cases were executed once through the npm `zcode-app-cli` 3.11.2-24 TUI (`yolo` with the existing-file guard hook, main and lite bound to `glm-5.3-flash-nvidia` at the loopback tunnel, `limit.context` 204800, `limit.output` 32000) against a small fixture repository with a seeded bug and a unittest suite. The H rows below are that run; the PARTIAL cells name the untested part.

**Decision (2026-09-22).** The accepted route is the npm `zcode-app-cli` distribution, which is what the operator runs. The official Desktop GUI stays BLOCKED on [feedback #270](https://github.com/zai-org/feedback/issues/270) and is no longer the required target. Claude Code is skipped by decision, because its configuration conflicts with a subscription setup on the same machine, so CC-01, CC-02 and the remaining API-04 items are closed as not pursued. Neither ran an H case. The untested sub-items of the PARTIAL rows are outside the accepted scope.

| ID | Target | Action and acceptance criterion | Status |
|---|---|---|---|
| API-01 | Basic API | Listed/selected served IDs match; short Japanese and English requests receive local-model responses | PASS (`api-acceptance-low-local`) |
| API-02 | Basic API | Chat Completions full response and SSE preserve termination, UTF-8 and reasoning/final-answer boundaries; long Japanese/Korean output is monitored separately by [`server mojibake`](validation.md#full-model-tp2-experimental-scope) | PASS (same run; chat at `reasoning_effort=low`) |
| API-03 | Basic API | Harmless tool request → validated JSON arguments → tool result → final answer; IDs survive multiple rounds | PASS (same run) |
| API-04 | Anthropic-compatible API | Messages full/SSE, tool_use/tool_result and count_tokens have valid structure, termination and usage | PARTIAL: Messages full response and count_tokens pass at model-default effort; SSE, tool_use/tool_result and error cases not pursued |
| ZC-01 | ZCode | Custom model is selectable; observed endpoint and model ID match local configuration | Desktop BLOCKED ([feedback #270](https://github.com/zai-org/feedback/issues/270)): the GUI never ran; the bundled CLI fails to start interactively on a missing `@zcode/tui`, while its headless `--prompt` reaches the local served ID (`20260914-zcode-exists-guard`). npm CLI: provider at the tunnel endpoint, served ID selected, Japanese round trip, headless prompt answered by the local model |
| ZC-02 | ZCode | Declared input capabilities (text, tools and image; no video) respected; no silent substitution with a default cloud model | Desktop BLOCKED, as ZC-01. npm CLI: main and lite bound to the local served ID, catalog refresh disabled |
| CC-01 | Claude Code | Isolated configuration starts; main/helper requests reach the served ID without auth/model-resolution loops | NOT RUN — skipped by decision (2026-09-22) |
| CC-02 | Claude Code | Anthropic tool IDs, streamed JSON, reasoning and stop_reason allow continuing after tool results | NOT RUN — skipped by decision (2026-09-22) |
| H-01 | Shared | Read two small fixture-repo files and explain their actual content; do not claim unread content was inspected | PASS (`20260915-harness-h-sandbox`): two fixture files explained from their actual content; the unread test file was reported as unread |
| H-02 | Shared | Fix one small bug; only authorized files receive the intended diff | PASS (same run): the off-by-one in `calc.py` removed; `git diff` showed that one file only |
| H-03 | Shared | Execute an approved local test and report results consistent with its real exit code/log | PASS (same run): red with two of five tests failing and exit 1, then green with all five passing and exit 0, reported as logged |
| H-04 | Shared | Deny a harmless marker-file creation once; verify no file was created and approval was not bypassed | PARTIAL (same run): the guard turned the `Write` under `.zcode` into a prompt and no approval was bypassed, but the marker was created after a client-side approval, so "deny once, nothing created" is not met |
| H-05 | Shared | Preserve arguments/results/order across read → edit → test tool rounds, with additional agents disabled | PASS (same run): about fifteen tool rounds (read, test, edit, diff, commit, write, remove, config read, curl, background job start and stop) kept arguments, results and order |
| H-06 | Shared | Cancel generation/tool waiting, then accept a fresh request; no infinite retry, orphan job or dead server | PARTIAL (same run): a background job was stopped with no orphan process and the server answered 200 afterwards; interrupting generation itself and the infinite-retry check are untested |
| H-07 | Shared | Restart/resume the test conversation with the same local destination and approval settings | PASS (same run): the session was resumed after a client restart with the same session ID, endpoint, model limits, permission mode and hook |
| H-08 | Shared | Exercise the actual context boundary; explicit compaction/error without silent history loss; do not assume 200k/1M support | PARTIAL (same run): `limit.context` was 204800, the 200K profile of that date, and matched the server `max_model_len` from `/v1/models`; `limit.output` is 32000; behaviour near the limit and compaction are untested. The current profiles serve 262,144; not rerun at 256K |
| H-09 | Shared | Observe inference destinations; an unavailable local endpoint must not cause cloud inference fallback. Record ancillary traffic separately from claims of offline operation | PARTIAL (same run): the model route is the loopback tunnel and no fallback setting exists; outage fallback untested. Ancillary traffic in that run: two user-requested `curl` calls to `api.fxtwitter.com` and `pbs.twimg.com`; client telemetry was not sampled (see the 2026-09-14 TCP sample above) |
| H-10 | Shared | Complete the same small read/fix/test/report task; retain API traces, artifacts, correctness and latency | PARTIAL (same run): the read/fix/test/report task completed with artifacts and correctness kept in the run log and Git history; no API tap and no latency measurement |
| H-11 | Shared | Confirm the supported reasoning profile reaches the local service, no unsupported off flag is sent, and reasoning remains separate from final content; record unsupported effort mapping explicitly | PARTIAL (same run): no off flag is configured or sent and no reasoning text leaked into final answers; `reasoning_effort` on the wire was not tapped |

For H-08, use the actual server limit from the server TOML. Multi-agent, MCP and image workflows are later, separate tests.

## Evidence and result handling

Under private `records/<run-id>/`, save case ID, harness/distribution/version/hash, non-secret settings, pinned server/model identity, expected and actual behavior, PASS/PARTIAL/FAIL/BLOCKED/NOT RUN, and references to request IDs, logs, diffs and test output. When a case changes status, update the matrix above in the same change.

Report each harness, and each ZCode distribution, separately. A pass on one never closes another. The required cases are those of the accepted route; retain them if unsupported, unimplemented or constrained by terms: mark BLOCKED with evidence and an alternative. Consider an upstream fix or small adapter only after identifying a concrete API gap, documenting its license and additional tests.
