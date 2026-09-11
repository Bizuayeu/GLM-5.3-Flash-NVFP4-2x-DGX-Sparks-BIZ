"""Take a bounded, attributed pilot sample from the pinned LLM-jp corpus."""

import argparse
import gzip
import hashlib
import io
import json
import urllib.request
from pathlib import Path

REVISION = "e928f19330f5271b29f382fe5a01245dec7d57c2"
BASE = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/"
SUBSETS = {"ja-wiki": "ja/ja_wiki", "en-wiki": "en/en_wiki", "code": "code/code_stack"}
PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}


def accepted_code_license(meta):
    licenses = set(meta.get("max_stars_repo_licenses") or [])
    return bool(licenses) and licenses <= PERMISSIVE


class BoundedReader:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.consumed = stream, limit, 0

    def read(self, size=-1):
        if self.consumed >= self.limit:
            raise ValueError("Pilot compressed-byte budget exhausted")
        size = min(size if size >= 0 else self.limit, self.limit - self.consumed)
        result = self.stream.read(size)
        self.consumed += len(result)
        return result


def document_key(row):
    # Exact duplicate text receives the same split even across different URLs.
    text = " ".join(row["text"].split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(key):
    bucket = int(key[:8], 16) % 10
    return "validation" if bucket == 8 else "test" if bucket == 9 else "train"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", choices=SUBSETS, default="ja-wiki")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument(
        "--source-byte-limit",
        type=int,
        default=64 * 1024**2,
        help="Compressed transfer cap for the pilot sample",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=512,
        help="Pilot size, not the full training requirement",
    )
    args = parser.parse_args(argv)
    if args.documents < 1 or args.shard < 0 or args.source_byte_limit < 1:
        parser.error(
            "documents/source-byte-limit must be positive and shard nonnegative"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    source = (
        BASE + REVISION + "/" + SUBSETS[args.subset] + f"/train_{args.shard}.jsonl.gz"
    )
    manifest = {
        "status": "sampling",
        "source": source,
        "revision": REVISION,
        "subset": SUBSETS[args.subset],
        "license": "per-repository: filtered permissive metadata"
        if args.subset == "code"
        else "CC-BY-SA-3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "attribution": "LLM-jp Corpus v3; original contributors retained in each record's meta",
        "sampling": "first distinct documents with at least 512 characters; pilot only",
        "split": "normalized full-text SHA256 modulo 10: 0-7 train, 8 validation, 9 test",
        "documents": 0,
        "counts": {"train": 0, "validation": 0, "test": 0},
        "full_source_checksum_verified": False,
    }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    seen = set()
    # Stream stops once the sample is complete; the entire large shard is not downloaded.
    with urllib.request.urlopen(source, timeout=60) as response:
        manifest["resolved_url"] = response.url
        bounded = BoundedReader(response, args.source_byte_limit)
        with gzip.GzipFile(fileobj=bounded) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as lines:
                with (args.output / "documents.jsonl").open(
                    "x", encoding="utf-8"
                ) as out:
                    for line_number, line in enumerate(lines, 1):
                        row = json.loads(line)
                        if args.subset == "code" and not accepted_code_license(
                            row.get("meta") or {}
                        ):
                            continue
                        if (
                            not isinstance(row.get("text"), str)
                            or len(row["text"]) < 512
                        ):
                            continue
                        key = document_key(row)
                        if key in seen:
                            continue
                        seen.add(key)
                        split = split_for(key)
                        record = {
                            "id": key,
                            "split": split,
                            "text": row["text"],
                            "meta": row.get("meta"),
                            "source_line": line_number,
                            "subset": SUBSETS[args.subset],
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        manifest["counts"][split] += 1
                        manifest["documents"] += 1
                        if manifest["documents"] == args.documents:
                            break
        manifest["compressed_bytes_read"] = bounded.consumed
    if args.subset == "code":
        manifest["license_url"] = "https://huggingface.co/datasets/bigcode/the-stack"
        manifest["license_filter"] = sorted(PERMISSIVE)
    manifest["status"] = (
        "complete" if manifest["documents"] == args.documents else "insufficient"
    )
    manifest["sample_sha256"] = hashlib.sha256(
        (args.output / "documents.jsonl").read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: manifest[k] for k in ("status", "documents", "counts", "sample_sha256")}
        )
    )
    if manifest["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
