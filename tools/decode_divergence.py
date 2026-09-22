"""First diverging token between two decode_check.py token records (their TOKENS_OUT files).

    python3 tools/decode_divergence.py records/<run-a>/tokens-prose.json records/<run-b>/tokens-prose.json

Prints, per sample index, the token count, the first index where the ids differ (or "identical") and the text
around it, so a launch difference reads as one tie-break flip or as an early systematic drift.
"""

import json
import sys


def main():
    a, b = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
    for i, (x, y) in enumerate(zip(a["samples"], b["samples"])):
        ia, ib = x["token_ids"], y["token_ids"]
        k = next((j for j in range(min(len(ia), len(ib))) if ia[j] != ib[j]), None)
        if k is None and len(ia) == len(ib):
            print(f"sample {i}: identical ({len(ia)} tokens)")
            continue
        k = len(ia) if k is None else k
        ta, tb = x["text"], y["text"]
        c = next(
            (j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]),
            min(len(ta), len(tb)),
        )
        print(
            f"sample {i}: first differing token {k}/{len(ia)} "
            f"(ids {ia[k : k + 3]} vs {ib[k : k + 3]}), char {c}"
        )
        print("   A:", repr(ta[max(0, c - 60) : c + 60]))
        print("   B:", repr(tb[max(0, c - 60) : c + 60]))


if __name__ == "__main__":
    main()
