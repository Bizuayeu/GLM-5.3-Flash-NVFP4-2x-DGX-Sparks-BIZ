"""Check whether a served model reports and reuses a cached prompt prefix.

The probe sends one long prompt twice and reports what the second call
restored. A zero result separates the two independent causes documented in
docs/harnesses.md: the server may not report cached tokens at all
(`--enable-prompt-tokens-details` absent), or it may compute the prompt
approximately and publish nothing to the shared cache (`lpa.enabled = true`,
which this probe opts out of per request).

Reuse is also block-aligned. A prompt shorter than the runtime's aligned
attention block can never hit, so the default length is generous.

    python tools/check_prefix_cache.py --base-url http://127.0.0.1:8893/v1 --model glm-5.3-flash-nvidia
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

SENTENCE = "Note {0}: a copper kettle cools slowly on the veranda while rain crosses the valley."


def build_prompt(lines):
    return " ".join(SENTENCE.format(index) for index in range(lines))


def ask(base_url, model, prompt, timeout, exact):
    body = {
        "model": model,
        "max_tokens": 8,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a terse assistant. " + prompt},
            {"role": "user", "content": "Reply with the single word PONG."},
        ],
    }
    if exact:
        # Documented per-request opt-out; ignored by a server without the extension.
        body["vllm_xargs"] = {"glm53_lpa_mode": "off"}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["usage"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8893/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--lines",
        type=int,
        default=900,
        help="Filler sentences; the default is far above the aligned block",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--approximate",
        action="store_true",
        help="Send both calls without the exact-computation opt-out",
    )
    args = parser.parse_args(argv)

    prompt = build_prompt(args.lines)
    try:
        calls = [
            ask(args.base_url, args.model, prompt, args.timeout, not args.approximate)
            for _ in range(2)
        ]
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"request failed: {error}", file=sys.stderr)
        return 2

    details = [call.get("prompt_tokens_details") for call in calls]
    cached = [(detail or {}).get("cached_tokens") for detail in details]
    result = {
        "prompt_tokens": calls[0].get("prompt_tokens"),
        "reports_prompt_tokens_details": details[1] is not None,
        "cached_tokens": cached,
        "reused_on_repeat": bool(cached[1]),
    }
    print(json.dumps(result, indent=1))

    if details[1] is None:
        print(
            "The server did not report prompt_tokens_details. Set "
            "api.prompt_tokens_details in the server TOML "
            "(--enable-prompt-tokens-details) before reading any cache display.",
            file=sys.stderr,
        )
        return 1
    if not cached[1]:
        print(
            "Nothing was restored on the repeat. An approximated request publishes "
            "no shared prefix, so check lpa.enabled and whether the prompt exceeds "
            "the runtime's aligned attention block.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
