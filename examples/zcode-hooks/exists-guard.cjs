#!/usr/bin/env node
// ZCode PreToolUse hook: run in yolo mode, but return to "ask" for changes to
// files that already exist and for destructive shell commands.
//
// Verified against ZCode Desktop 3.11.2 (bundled runtime 0.16.5): the runtime
// merges a hook "ask" over a policy "allow" (including mode.yolo), so this hook
// is the only place where yolo can be narrowed. Project permission rules and
// disallowedTools are evaluated after yolo and do not narrow it.
//
// Decisions:
//   Edit                          -> ask   (always targets an existing file)
//   Write  (target exists)        -> ask
//   Write  (target absent)        -> allow
//   Write/Edit under ~/.zcode     -> ask   (harness config directory; see docs/harnesses.md)
//   Bash   (destructive pattern)  -> ask   (heuristic: string match on the command)
//   Bash   (otherwise)            -> allow
//   any other tool                -> no opinion ({})
//
// Observed payload keys (2026-09-14, headless --prompt): tool_name, tool_input
// {file_path (absolute), content} / {command}, cwd, permission_mode.
// Set ZCODE_GUARD_LOG=<path> to append raw stdin for payload inspection.
"use strict";
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const DESTRUCTIVE = [
  /\brm\b/, /\brmdir\b/, /\bdel\b/, /\berase\b/, /\bunlink\b/,
  /\bmv\b/, /\bmove\b/, /\bren(ame)?\b/,
  /Remove-Item/i, /Move-Item/i, /Rename-Item/i, /Clear-Content/i, /\bri\b/, /\brd\b/,
  /\bgit\s+(rm|clean|reset|checkout\s+--|restore|push\s+.*--force|branch\s+-D)\b/,
  /\btruncate\b/, /\bshred\b/, /\bsed\s+-i\b/, /\btee\b/,
  /(^|[^<>])>{1,2}\s*[^&\s]/, // > and >> redirection (excludes >&2)
  /Set-Content/i, /Out-File/i, /Add-Content/i,
];

const CONFIG_DIR = path.resolve(os.homedir(), ".zcode");

function targetPath(input, cwd) {
  const fp = input.file_path || input.filePath || input.path;
  if (typeof fp !== "string" || !fp) return undefined;
  return path.isAbsolute(fp) ? path.normalize(fp) : path.resolve(cwd, fp);
}

function underConfigDir(abs) {
  const rel = path.relative(CONFIG_DIR, abs);
  return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
}

function decide(payload) {
  const tool = payload.tool_name || payload.toolName;
  const input = payload.tool_input || payload.toolInput || {};
  const cwd = payload.cwd || process.cwd();
  if (tool === "Edit" || tool === "Write") {
    const abs = targetPath(input, cwd);
    if (!abs) return ["ask", `${tool}: target path missing`];
    if (underConfigDir(abs)) return ["ask", `harness config directory: ${abs}`];
    if (tool === "Edit") return ["ask", `edit of existing file: ${abs}`];
    return fs.existsSync(abs) ? ["ask", `overwrite of existing file: ${abs}`] : ["allow", "new file"];
  }
  if (tool === "Bash") {
    const cmd = String(input.command || "");
    const hit = DESTRUCTIVE.find((re) => re.test(cmd));
    return hit ? ["ask", `destructive command pattern ${hit}`] : ["allow", "non-destructive command"];
  }
  return [undefined, "not covered"];
}

let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => {
  if (process.env.ZCODE_GUARD_LOG) {
    try { fs.appendFileSync(process.env.ZCODE_GUARD_LOG, raw + "\n"); } catch {}
  }
  let payload = {};
  try { payload = JSON.parse(raw); } catch { payload = {}; }
  const [decision, reason] = decide(payload);
  if (!decision) { process.stdout.write("{}"); return; }
  process.stdout.write(JSON.stringify({
    hookSpecificOutput: { hookEventName: "PreToolUse", permissionDecision: decision, permissionDecisionReason: reason },
  }));
});
