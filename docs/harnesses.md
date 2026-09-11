# Harness integration and acceptance

[日本語](harnesses.ja.md) · [Setup](../SETUP.md) · [Licensing](licensing.md)

A harness is the client that manages conversation, files, tool execution, history and approvals; it is separate from the GPU inference server. **Official ZCode and Claude Code are both required acceptance targets. Their client cases remain NOT RUN; basic local API smoke has separate evidence.** Full-model TP=2 startup and API qualification are prerequisites.

## Connection design

| Route | Connection | Decision |
|---|---|---|
| Basic API checks | Local vLLM Chat Completions / Messages | Separate server defects from client integration defects |
| ZCode | Custom Provider → SSH tunnel → OpenAI-compatible vLLM | Official Z.ai harness; primary acceptance route |
| Claude Code CLI | Anthropic format → SSH tunnel → vLLM Messages | Experimental compatibility route; not Anthropic support for non-Claude models |

ZCode's [official site](https://zcode.z.ai/en) identifies it as a GLM harness, and its [configuration guide](https://zcode.z.ai/en/docs/configuration#custom-providers-anthropic--openai-compatible) describes custom compatible services, including private deployments. This does not establish compatibility with our NVFP4 TP=2 configuration.

The pinned vLLM [Claude Code guide](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/serving/integrations/claude_code.md) and [router](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/entrypoints/anthropic/api_router.py) provide `/v1/messages` and `/v1/messages/count_tokens`. **Start with direct integration, without another gateway.** This reduces components and license handling, but requires testing vLLM conversion, GLM tool parsing and the chosen client version.

[Z.ai's Claude Code instructions](https://docs.z.ai/devpack/tool/claude) concern its hosted service. [Anthropic's current gateway guidance](https://code.claude.com/docs/en/llm-gateway) explicitly excludes support for routing to non-Claude models. Keep vendor support, technical compatibility and applicable terms distinct.

## Candidate setup after full-model startup

These are **planned settings, not a validated deployment recipe**. Do not bypass the beta's startup gate. Record client version/distribution hash, server source/image/revision and effective arguments first.

Open a dedicated PowerShell for an SSH tunnel (`node-a` is illustrative; add your existing `-F` configuration when needed):

```powershell
ssh -N -L 127.0.0.1:8891:127.0.0.1:8891 node-a
```

Use credentials issued for the local service if authentication is configured. Do not reuse Z.ai or Anthropic cloud credentials for the local experiment.

### ZCode

Create a dedicated test workspace/provider through Model Settings → Add Provider. Select an OpenAI-compatible local provider; try `http://127.0.0.1:8891/v1` with model ID `glm-5.3-flash-nvidia`, checking the actual request path for duplicated `/v1`. Supply the local service's key, or a non-secret test value only if an unauthenticated loopback endpoint requires a nonempty UI field; that value adds no security.

Declare text/tool capability initially. Do not enable image input, cloud fallback, external integrations or additional agents by default. Confirm the selected model and actual destination.

### Claude Code CLI

Use a **dedicated PowerShell session and empty test workspace**, isolated from everyday configuration and credentials. Check that the selected CLI supports the configuration variables, including `CLAUDE_CONFIG_DIR`; do not modify global settings.

```powershell
$env:CLAUDE_CONFIG_DIR = "$PWD\.claude-local-test"
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8891'
$env:ANTHROPIC_API_KEY = 'local-test'
$env:ANTHROPIC_AUTH_TOKEN = 'local-test'
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = 'glm-5.3-flash-nvidia'
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = 'glm-5.3-flash-nvidia'
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'glm-5.3-flash-nvidia'
claude --model glm-5.3-flash-nvidia
```

Use the server root URL here: the CLI appends the Messages path. `local-test` is a non-secret placeholder for an unauthenticated loopback test, not a real credential; authenticated services need their actual dedicated credentials. Map main/tier/helper requests to the local served ID and verify their routing. Follow official authentication requirements without modifying or bypassing checks.

This is a CLI plan. Desktop, web and Remote Control have different configuration/support boundaries; consult [Anthropic's per-surface guide](https://code.claude.com/docs/en/llm-gateway-connect). Closing the client does not stop the model server.

## Required acceptance matrix

For this fixed GLM template, keep thinking active. It always starts an assistant thinking block and does not read an off switch; avoid `thinking=false` / `enable_thinking=false`. The [model card](https://huggingface.co/zai-org/GLM-5.3-Flash) documents `reasoning_effort=low/high/max`, defaulting to max, and recommends `clear_thinking=true` for chat. Low effort is a chat test profile, not a claim to reproduce max-effort leaderboard scores. The matching parser/template leak is tracked in [vLLM #54744](https://github.com/vllm-project/vllm/issues/54744); [PR #54825](https://github.com/vllm-project/vllm/pull/54825) was open and unmerged when reviewed. Do not assume an arbitrary image includes it.

Judge normal acceptance by final answers, structured tool calls, tool results and approval boundaries. Keep exact reasoning-text/token replay and cross-batch bitwise comparisons as separate numerical diagnostics. Preserve mismatches; do not treat every such mismatch as task failure or prove model correctness solely from matching final answers. Use one active sequence for golden checks and assess parallel throughput/quality separately.

**ZCode/Claude Code client cases are NOT RUN. The local API has initial smoke evidence, but complete API matrix coverage remains pending, including Anthropic streaming and boundary/error cases.** Complete the API group, then run every shared H case separately for each harness. Source availability or a truncated GPU fixture cannot replace actual requests and artifacts.

| ID | Target | Action and acceptance criterion |
|---|---|---|
| API-01 | Basic API | Listed/selected served IDs match; short Japanese and English requests receive local-model responses |
| API-02 | Basic API | Chat Completions full response and SSE preserve termination, UTF-8 and reasoning/final-answer boundaries |
| API-03 | Basic API | Harmless tool request → validated JSON arguments → tool result → final answer; IDs survive multiple rounds |
| API-04 | Anthropic-compatible API | Messages full/SSE, tool_use/tool_result and count_tokens have valid structure, termination and usage |
| ZC-01 | ZCode | Custom model is selectable; observed endpoint and model ID match local configuration |
| ZC-02 | ZCode | Initial text-only capability respected; no silent substitution with a default cloud model |
| CC-01 | Claude Code | Isolated configuration starts; main/helper requests reach the served ID without auth/model-resolution loops |
| CC-02 | Claude Code | Anthropic tool IDs, streamed JSON, reasoning and stop_reason allow continuing after tool results |
| H-01 | Both | Read two small fixture-repo files and explain their actual content; do not claim unread content was inspected |
| H-02 | Both | Fix one small bug; only authorized files receive the intended diff |
| H-03 | Both | Execute an approved local test and report results consistent with its real exit code/log |
| H-04 | Both | Deny a harmless marker-file creation once; verify no file was created and approval was not bypassed |
| H-05 | Both | Preserve arguments/results/order across read → edit → test tool rounds, with additional agents disabled |
| H-06 | Both | Cancel generation/tool waiting, then accept a fresh request; no infinite retry, orphan job or dead server |
| H-07 | Both | Restart/resume the test conversation with the same local destination and approval settings |
| H-08 | Both | Exercise the actual context boundary; explicit compaction/error without silent history loss; do not assume 200k/1M support |
| H-09 | Both | Observe inference destinations; an unavailable local endpoint must not cause cloud inference fallback. Record ancillary traffic separately from claims of offline operation |
| H-10 | Both | Complete the same small read/fix/test/report task; retain API traces, artifacts, correctness and latency |
| H-11 | Both | Confirm the supported reasoning profile reaches the local service, no unsupported off flag is sent, and reasoning remains separate from final content; record unsupported effort mapping explicitly |

For H-08, use the actual server limit. The candidate launcher sets 32,768; alignment with client context assumptions remains untested. Multi-agent, MCP and image workflows are later, separate tests.

## Evidence and result handling

Under private `records/<run-id>/`, save case ID, harness/version/distribution hash, non-secret settings, pinned server/model identity, expected and actual behavior, PASS/FAIL/BLOCKED/NOT RUN, and references to request IDs, logs, diffs and test output.

Report ZCode and Claude Code separately. A pass on one never closes the other. Retain required cases if unsupported, unimplemented or constrained by terms: mark BLOCKED with evidence and an alternative. Consider an upstream fix or small adapter only after identifying a concrete API gap, documenting its license and additional tests.
