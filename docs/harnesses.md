# Harness integration and acceptance

[日本語](harnesses.ja.md) · [Setup](../SETUP.md) · [Licensing](licensing.md) · [Validation](validation.md)

A harness is the client that manages conversation, files, tool execution, history and approvals; it is separate from the GPU inference server. **This document is the single source for harness connection design, the acceptance matrix and its run status.** Other documents point here instead of restating status. Full-model TP=2 startup and the basic API group are prerequisites for every client case.

## Connection design

| Route | Connection | Decision |
|---|---|---|
| Basic API checks | Local vLLM Chat Completions / Messages | Separate server defects from client integration defects |
| ZCode | Custom Provider → SSH tunnel → OpenAI-compatible vLLM | Official Z.ai harness; primary acceptance route |
| Claude Code CLI | Anthropic format → SSH tunnel → vLLM Messages | Experimental compatibility route; not Anthropic support for non-Claude models |

ZCode's [official site](https://zcode.z.ai/en) identifies it as a GLM harness, and its [configuration guide](https://zcode.z.ai/en/docs/configuration#custom-providers-anthropic--openai-compatible) describes custom compatible services, including private deployments. This does not establish compatibility with our NVFP4 TP=2 configuration.

The pinned vLLM [Claude Code guide](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/serving/integrations/claude_code.md) and [router](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/entrypoints/anthropic/api_router.py) provide `/v1/messages` and `/v1/messages/count_tokens`. **Start with direct integration, without another gateway.** This reduces components and license handling, but requires testing vLLM conversion, GLM tool parsing and the chosen client version.

[Z.ai's Claude Code instructions](https://docs.z.ai/devpack/tool/claude) concern its hosted service. [Anthropic's current gateway guidance](https://code.claude.com/docs/en/llm-gateway) explicitly excludes support for routing to non-Claude models. Keep vendor support, technical compatibility and applicable terms distinct.

## ZCode distributions

"ZCode" is not one artifact. Acceptance and outbound-traffic facts differ by distribution, so this document names them separately.

| Distribution | What it is | Role here |
|---|---|---|
| Official Desktop GUI | Electron application from the [official installer](https://zcode.z.ai/en/docs/install); providers are added in Model Settings | Required acceptance target |
| Official Desktop bundled CLI | `resources/glm/zcode.cjs` inside the Desktop install | Not usable interactively in the inspected version (see status) |
| npm `zcode-app-cli` | Unofficial [terminal wrapper](https://github.com/kingsword09/zcode-cli) that vendors the ZCode runtime and adds its own TUI (MIT for its own code; ZCode runtime keeps upstream terms) | Supporting evidence only; never closes an official-target case |

## Candidate setup

These are **planned settings, not a validated deployment recipe**. Do not bypass the beta's startup gate. Record client version/distribution hash, server source/image/revision and effective arguments first. Use the API port from your startup TOML; the example template uses 8893 on loopback.

Open a dedicated PowerShell for an SSH tunnel (`node-a` is illustrative; add your existing `-F` configuration when needed):

```powershell
ssh -N -L 127.0.0.1:8893:127.0.0.1:8893 node-a
```

Use credentials issued for the local service if authentication is configured. Do not reuse Z.ai or Anthropic cloud credentials for the local experiment.

**Deny the harness write access to its own client configuration directory during acceptance** (for ZCode, the `.zcode` directory under the user profile; for Claude Code, the directory named by `CLAUDE_CONFIG_DIR`). A harness running with build permission can edit its own configuration on request, report success, and then fail to start; a client-side schema rejection surfaces as an unrelated error such as a missing model configuration. Use the client's tool denylist for those paths and keep a copy of the working configuration.

### ZCode

Create a dedicated test workspace/provider through Model Settings → Add Provider. Select an OpenAI-compatible local provider; try `http://127.0.0.1:8893/v1` with model ID `glm-5.3-flash-nvidia`, checking the actual request path for duplicated `/v1`. Supply the local service's key, or a non-secret test value only if an unauthenticated loopback endpoint requires a nonempty UI field; that value adds no security.

Declare text/tool capability initially. Do not enable image input, cloud fallback, external integrations or additional agents by default. Point both the main and the helper ("lite") model at the local served ID. Confirm the selected model and actual destination. The UI locale accepts only the values the client documents (`zcode --help` lists them); an unsupported value invalidates the whole configuration file.

### Claude Code CLI

Use a **dedicated PowerShell session and empty test workspace**, isolated from everyday configuration and credentials. Check that the selected CLI supports the configuration variables, including `CLAUDE_CONFIG_DIR`; do not modify global settings.

```powershell
$env:CLAUDE_CONFIG_DIR = "$PWD\.claude-local-test"
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8893'
$env:ANTHROPIC_API_KEY = 'local-test'
$env:ANTHROPIC_AUTH_TOKEN = 'local-test'
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = 'glm-5.3-flash-nvidia'
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = 'glm-5.3-flash-nvidia'
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'glm-5.3-flash-nvidia'
claude --model glm-5.3-flash-nvidia
```

Use the server root URL here: the CLI appends the Messages path. `local-test` is a non-secret placeholder for an unauthenticated loopback test, not a real credential; authenticated services need their actual dedicated credentials. Map main/tier/helper requests to the local served ID and verify their routing. Follow official authentication requirements without modifying or bypassing checks.

This is a CLI plan. Desktop, web and Remote Control have different configuration/support boundaries; consult [Anthropic's per-surface guide](https://code.claude.com/docs/en/llm-gateway-connect). Closing the client does not stop the model server.

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

For this fixed GLM template, keep thinking active. It always starts an assistant thinking block and does not read an off switch; avoid `thinking=false` / `enable_thinking=false`. The [model card](https://huggingface.co/zai-org/GLM-5.3-Flash) documents `reasoning_effort=low/high/max`, defaulting to max, and recommends `clear_thinking=true` for chat. Low effort is a chat test profile, not a claim to reproduce max-effort leaderboard scores. The matching parser/template leak is tracked in [vLLM #54744](https://github.com/vllm-project/vllm/issues/54744); [PR #54825](https://github.com/vllm-project/vllm/pull/54825) was open and unmerged when reviewed. Do not assume an arbitrary image includes it.

Judge normal acceptance by final answers, structured tool calls, tool results and approval boundaries. Keep exact reasoning-text/token replay and cross-batch bitwise comparisons as separate numerical diagnostics. Preserve mismatches; do not treat every such mismatch as task failure or prove model correctness solely from matching final answers. Use one active sequence for golden checks and assess parallel throughput/quality separately.

## Acceptance matrix and status

Status values: PASS, PARTIAL (some criteria met, listed), BLOCKED (cannot run; evidence and alternative recorded), NOT RUN. Run the API group first, then every shared H case separately for each harness with real requests and artifacts. Source availability or a truncated GPU fixture cannot replace actual requests. Status as of 2026-09-14; run IDs refer to private `records/`.

| ID | Target | Action and acceptance criterion | Status |
|---|---|---|---|
| API-01 | Basic API | Listed/selected served IDs match; short Japanese and English requests receive local-model responses | PASS (`api-acceptance-low-local`) |
| API-02 | Basic API | Chat Completions full response and SSE preserve termination, UTF-8 and reasoning/final-answer boundaries | PASS (same run; chat at `reasoning_effort=low`) |
| API-03 | Basic API | Harmless tool request → validated JSON arguments → tool result → final answer; IDs survive multiple rounds | PASS (same run) |
| API-04 | Anthropic-compatible API | Messages full/SSE, tool_use/tool_result and count_tokens have valid structure, termination and usage | PARTIAL: Messages full response and count_tokens pass at model-default effort; Messages SSE, tool_use/tool_result and error cases NOT RUN |
| ZC-01 | ZCode | Custom model is selectable; observed endpoint and model ID match local configuration | Official Desktop GUI NOT RUN. Desktop bundled CLI BLOCKED: interactive start fails with a missing `@zcode/tui` package ([zai-org/feedback #270](https://github.com/zai-org/feedback/issues/270)), no GLM request sent. Supporting evidence from npm `zcode-app-cli` 3.11.2-24: provider at the tunnel endpoint, served ID selected, Japanese round trip and a headless prompt answered by the local model |
| ZC-02 | ZCode | Initial text-only capability respected; no silent substitution with a default cloud model | Same as ZC-01: official NOT RUN / BLOCKED; npm distribution smoke shows main and lite bound to the local served ID with catalog refresh disabled |
| CC-01 | Claude Code | Isolated configuration starts; main/helper requests reach the served ID without auth/model-resolution loops | NOT RUN |
| CC-02 | Claude Code | Anthropic tool IDs, streamed JSON, reasoning and stop_reason allow continuing after tool results | NOT RUN |
| H-01 | Both | Read two small fixture-repo files and explain their actual content; do not claim unread content was inspected | NOT RUN |
| H-02 | Both | Fix one small bug; only authorized files receive the intended diff | NOT RUN |
| H-03 | Both | Execute an approved local test and report results consistent with its real exit code/log | NOT RUN |
| H-04 | Both | Deny a harmless marker-file creation once; verify no file was created and approval was not bypassed | NOT RUN |
| H-05 | Both | Preserve arguments/results/order across read → edit → test tool rounds, with additional agents disabled | NOT RUN |
| H-06 | Both | Cancel generation/tool waiting, then accept a fresh request; no infinite retry, orphan job or dead server | NOT RUN |
| H-07 | Both | Restart/resume the test conversation with the same local destination and approval settings | NOT RUN |
| H-08 | Both | Exercise the actual context boundary; explicit compaction/error without silent history loss; do not assume 200k/1M support | NOT RUN |
| H-09 | Both | Observe inference destinations; an unavailable local endpoint must not cause cloud inference fallback. Record ancillary traffic separately from claims of offline operation | NOT RUN. Supporting evidence for the npm distribution: the recorded TCP sample above showed only the loopback tunnel during one prompt; outage fallback untested; official Desktop and Claude Code not sampled |
| H-10 | Both | Complete the same small read/fix/test/report task; retain API traces, artifacts, correctness and latency | NOT RUN |
| H-11 | Both | Confirm the supported reasoning profile reaches the local service, no unsupported off flag is sent, and reasoning remains separate from final content; record unsupported effort mapping explicitly | NOT RUN (API-02/04 ran at different effort settings; the mapping per client is untested) |

For H-08, use the actual server limit from the startup TOML; alignment with client context assumptions remains untested. Multi-agent, MCP and image workflows are later, separate tests.

## Evidence and result handling

Under private `records/<run-id>/`, save case ID, harness/distribution/version/hash, non-secret settings, pinned server/model identity, expected and actual behavior, PASS/PARTIAL/FAIL/BLOCKED/NOT RUN, and references to request IDs, logs, diffs and test output. When a case changes status, update the matrix above in the same change.

Report ZCode and Claude Code separately, and within ZCode report each distribution separately. A pass on one never closes another. Retain required cases if unsupported, unimplemented or constrained by terms: mark BLOCKED with evidence and an alternative. Consider an upstream fix or small adapter only after identifying a concrete API gap, documenting its license and additional tests.
