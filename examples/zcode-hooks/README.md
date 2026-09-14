# ZCode PreToolUse guard

A hook that lets ZCode run in `yolo` while still asking before it changes
something that already exists. Facts about the permission chain, the model
limits and the measurements behind this are in
[harness integration](../../docs/harnesses.md#zcode-permission-modes-model-limits-and-the-existing-file-guard);
this file is only the setup.

## Why a hook

ZCode evaluates `yolo` before project permission rules and before
`disallowedTools`, so path rules cannot narrow it. A `PreToolUse` hook is the
only place where a policy `allow` can be turned back into a prompt. The
permission descriptors carry no notion of "the file already exists": `Write`
and `Edit` share one permission, and deletion goes through `Bash`.

## Install

1. Copy `exists-guard.cjs` outside this repository, for example into the
   client's `hooks` directory next to its configuration.
2. Merge `config.hooks.example.json` into the client configuration
   (`config.json`). Keep a copy of the working file first: a schema-rejected
   entry invalidates the whole configuration and surfaces as an unrelated
   error such as a missing model configuration.
3. Use an absolute interpreter path in `command`. A desktop client started
   from a launcher does not inherit a shell `PATH`, and the runtime fails the
   matched tool call on any hook exit other than 0 or 2, on a spawn error and
   on a timeout.

## Verify

Ask the client to create a new file (it should pass without a prompt), then to
change an existing one (it should prompt). If everything is denied instead,
the hook is not launching; restore the configuration copy and check the
interpreter path. Set `ZCODE_GUARD_LOG` to a writable path to append the raw
hook stdin, including file content, while inspecting the payload shape; unset
it afterwards.

## Limits

The shell gate is a string heuristic over the command line and has misses; the
conservative alternative is to ask for every `Bash` call. The hook only sees
the tools its matcher names, so a future file-writing tool passes unless the
matcher is extended. Protection is by existence, not by content: an approved
overwrite is still a full overwrite.
