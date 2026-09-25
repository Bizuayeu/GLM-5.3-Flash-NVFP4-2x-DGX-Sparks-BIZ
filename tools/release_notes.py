"""Print one version's CHANGELOG section; the release workflow publishes it."""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # run as a script from any directory

from glm53_setup.config import ROOT, version  # noqa: E402


def section(changelog, version):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"not a release version: {version!r}")
    match = re.search(
        rf"(?ms)^## {re.escape(version)} [^\n]*\n(.*?)(?=^## |\Z)", changelog
    )
    if match is None or not match[1].strip():
        raise ValueError(f"CHANGELOG.md has no notes for {version}")
    return match[1].strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument(
        "--match-project",
        action="store_true",
        help="fail unless pyproject.toml carries the same version",
    )
    args = parser.parse_args()
    if args.match_project:
        if version() != args.version:
            raise SystemExit("tag and pyproject.toml disagree on the version")
    notes = section((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), args.version)
    sys.stdout.buffer.write(notes.encode("utf-8"))


if __name__ == "__main__":
    main()
