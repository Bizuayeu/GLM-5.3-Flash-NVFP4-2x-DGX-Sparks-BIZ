"""Record the weights a serving pair loaded, per rank, and compare them with an earlier launch.

Through the memory probe's ``weight_digest`` method (`validation.memory_probe = true`, dev
route ``/collective_rpc``) each rank fingerprints every parameter and buffer as it sits on the
GPU after loading; the record keeps the rows and their per-layer digests. Taken after every
switch, before requests, and compared with the previous launch of the profile, it answers the
first question a launch in another numerical state raises: did it compute from the same bits?
Exit status 1 when a rank's tensors differ from the reference (their names are in the record).

    python3 tools/weight_digest.py --output records/<run>/weights.json [--reference records/<earlier>/weights.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # run as a script from any directory

from glm53_setup import server, server_config  # noqa: E402
from glm53_setup.config import DEFAULT_PROFILE
from glm53_setup.runtime.memory_probe import digest_differences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, help="An earlier record of the same profile"
    )
    args = parser.parse_args(argv)
    profile = server_config.load(args.config)
    current, info = server.running_head(profile)
    ranks = server.collective_rpc(profile, "weight_digest", tensors=True)
    record = {
        "fingerprint": server_config.fingerprint(profile),
        "image": info["Image"],
        "container": current if isinstance(current, str) else current.get("name"),
        "ranks": sorted(ranks, key=lambda r: r["rank"]),
    }
    code = 0
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        before = {r["rank"]: r["rows"] for r in reference["ranks"]}
        record["reference"] = str(args.reference)
        record["comparison"] = {}
        for rank in record["ranks"]:
            result = digest_differences(before.get(rank["rank"], []), rank["rows"])
            record["comparison"][rank["rank"]] = result
            moved = result["differing"] or result["missing"] or result["added"]
            print(
                f"rank {rank['rank']}: {result['same']} same, {len(result['differing'])} differing,"
                f" {len(result['missing'])} missing, {len(result['added'])} added"
                + (f"; layers {result['layers']}" if result["layers"] else "")
            )
            code = 1 if moved else code
        record["comparison"] = {str(k): v for k, v in record["comparison"].items()}
    else:
        for rank in record["ranks"]:
            summary = rank["summary"]
            print(
                f"rank {rank['rank']}: {summary['tensors']} tensors, overall {summary['overall']}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
