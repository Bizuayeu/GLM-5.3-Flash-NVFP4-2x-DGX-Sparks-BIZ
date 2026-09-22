"""Hash the indexer's kernels inside both serving workers and compare the ranks, and an earlier launch.

Through the memory probe's ``kernel_hashes`` method (`validation.memory_probe = true`, dev route
``/collective_rpc``) each rank runs the kpool indexer's computations on fixed inputs in its own
process (the fp32 head gate, the fused FWHT quantisation, the pool cache's writes, DeepGEMM's paged
MQA logits and the stable top-k) and hashes every output. On 2026-09-23 one rank's copy of the
replicated indexer was the first call to differ between two launches, while thirteen fresh
processes on one GB10 computed these kernels bit-identically; so the question is asked of the
serving processes themselves, after every switch. Exit status 1 when the ranks disagree or a hash
differs from the reference record (the keys are printed).

    python3 tools/kernel_hashes.py --output records/<run>/kernels.json [--reference records/<earlier>/kernels.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # run as a script from any directory

from glm53_setup import server, server_config  # noqa: E402
from glm53_setup.config import ROOT
from glm53_setup.runtime.memory_probe import kernel_hash_differences


def reference_differences(reference, ranks):
    """Per rank, the keys whose hash differs from the reference record's same rank."""
    before = {r["rank"]: r["hashes"] for r in reference["ranks"]}
    result = {}
    for row in ranks:
        old = before.get(row["rank"], {})
        result[str(row["rank"])] = sorted(
            k for k, v in row["hashes"].items() if old.get(k) != v
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "state/server.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, help="An earlier record of the same profile"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    profile = server_config.load(args.config)
    current, info = server.running_head(profile)
    ranks = sorted(
        server.collective_rpc(profile, "kernel_hashes", seed=args.seed),
        key=lambda r: r["rank"],
    )
    record = {
        "fingerprint": server_config.fingerprint(profile),
        "image": info["Image"],
        "container": current if isinstance(current, str) else current.get("name"),
        "ranks": ranks,
        "across_ranks": kernel_hash_differences(ranks),
    }
    across = record["across_ranks"]
    print(
        f"ranks {across['ranks']}: "
        + (
            "agree on every hash"
            if across["agree"]
            else f"differ on {across['differing']}"
        )
    )
    code = 0 if across["agree"] else 1
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        record["reference"] = str(args.reference)
        record["against_reference"] = reference_differences(reference, ranks)
        for rank, keys in record["against_reference"].items():
            print(
                f"rank {rank} against the reference: "
                + (f"differs on {keys}" if keys else "same")
            )
            code = 1 if keys else code
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
