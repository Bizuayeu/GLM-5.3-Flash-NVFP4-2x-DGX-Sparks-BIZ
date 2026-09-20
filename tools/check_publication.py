"""Audit checkout/public-export contents without displaying matched sensitive text."""

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "README.ja.md",
    "SETUP.md",
    "SETUP.ja.md",
    "CHANGELOG.md",
    "CHANGELOG.ja.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "config/runtime.lock.json",
    "pyproject.toml",
    ".gitignore",
    ".dockerignore",
}
PRIVATE_DIRS = {"state", "records", "upstream", ".ssh", ".venv", ".claude-local-test"}
PRIVATE_SUFFIXES = {
    ".safetensors",
    ".gguf",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".bundle",
    ".tar",
    ".gz",
    ".zip",
    ".log",
    ".jsonl",
}
SENSITIVE = [
    re.compile(r"-----BEGIN " + r"(?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    re.compile(r"\bhf_" + r"[A-Za-z0-9]{25,}\b"),
    re.compile(r"\bgh[pousr]_" + r"[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?i)[a-z]:[/\\]Users[/\\][a-z0-9_.-]+"),
    re.compile(r"/home/" + r"[a-z0-9_.-]+/"),
]
# README headline table and the benchmark document that owns its numbers.
VERSION = r"(\d+\.\d+\.\d+)"
HEADLINES = {
    "README.md": (
        "docs/benchmarks.md",
        rf"(?m)^#+ Headline measurements \({VERSION}\)",
        rf"(?m)^## Measurements on {VERSION}$",
    ),
    "README.ja.md": (
        "docs/benchmarks.ja.md",
        rf"(?m)^#+ 主要な測定値（{VERSION}）",
        rf"(?m)^## {VERSION}での測定$",
    ),
}


def headline_problems(name, readme, benchmarks):
    """The README headline must carry the newest version the benchmarks measured."""
    _, headline, measured = HEADLINES[name]
    versions = re.findall(measured, benchmarks)
    if not versions:
        return [f"no versioned measurements beside {name}"]
    latest = max(versions, key=lambda v: tuple(map(int, v.split("."))))
    shown = re.search(headline, readme)
    if shown is None:
        return [f"missing headline measurements: {name}"]
    if shown[1] != latest:
        return [
            f"stale headline measurements: {name} has {shown[1]}, latest is {latest}"
        ]
    return []


# Other recipes: README's table owns their links and licences; documents cite by name.
RECIPE_LINK = re.compile(r"https://(?:github\.com|huggingface\.co)/([\w.-]+/[\w.-]+)")
RESTATED_LICENSE = re.compile(
    r"[(（](?:AGPL-3\.0|MIT|Apache-2\.0|no license|ライセンスなし)[,;、]"
    r"|[,、] ?(?:MIT|no license|ライセンスなし)[)）]"
)


def recipe_problems(readme, documents):
    """Recipes come from README's table rows; other documents hold no copy.

    A link to some other repository of the same author is evidence, not a copy.
    """
    recipes = {
        recipe
        for line in readme.splitlines()
        if line.startswith("| [")
        for recipe in RECIPE_LINK.findall(line)
    }
    owners = sorted({recipe.split("/")[0] for recipe in recipes})
    cited = re.compile("|".join(map(re.escape, owners + ["Mia PR"])))
    problems = []
    for name, text in sorted(documents.items()):
        if recipes & set(RECIPE_LINK.findall(text)):
            problems.append(f"recipe link outside README: {name}")
        if owners and any(
            cited.search(line) and RESTATED_LICENSE.search(line)
            for line in text.splitlines()
        ):
            problems.append(f"recipe license restated outside README: {name}")
    return problems


def public_files(root, export_tree=False):
    if export_tree:
        return {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts
        }
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        capture_output=True,
        check=True,
    )
    return set(result.stdout.decode("utf-8").strip("\0").split("\0")) - {""}


def audit(root, files):
    problems = [f"missing required file: {name}" for name in sorted(REQUIRED - files)]
    for name in sorted(files):
        path = root / name
        parts = Path(name).parts
        if (
            set(parts) & PRIVATE_DIRS
            or path.suffix.lower() in PRIVATE_SUFFIXES
            or path.name == "site.json"
            or path.name.endswith(".local.json")
            or (path.name.startswith(".env") and path.name != ".env.example")
            or (name.startswith("docs/") and "PLAN" in path.name)
        ):
            problems.append(f"private/generated path: {name}")
        if not path.is_file():
            problems.append(f"missing working file: {name}")
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            problems.append(f"linked/outside file: {name}")
            continue
        if path.stat().st_size > 2_000_000:
            problems.append(f"unexpected large file: {name}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"unexpected binary file: {name}")
            continue
        if any(pattern.search(content) for pattern in SENSITIVE):
            problems.append(f"sensitive-text candidate: {name}")
        if path.suffix.lower() != ".md":
            continue
        # The repository uses inline Markdown links. Fenced examples are not links.
        prose = re.sub(r"(?ms)^(```|~~~).*?^\1[^\n]*$", "", content)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", prose):
            target = target.strip().strip("<>")
            url = urlsplit(target)
            if url.scheme or not url.path:
                continue
            resolved = (path.parent / unquote(url.path)).resolve()
            if not resolved.is_relative_to(root.resolve()):
                problems.append(f"outside Markdown link: {name}")
            else:
                relative = resolved.relative_to(root.resolve()).as_posix()
                if relative not in files and not any(
                    f.startswith(relative + "/") for f in files
                ):
                    problems.append(f"non-public Markdown target: {name} -> {relative}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--export-tree",
        action="store_true",
        help="Inspect an exported tree without Git",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    files = public_files(root, args.export_tree)
    problems = audit(root, files)
    if "config/runtime.lock.json" in files:
        lock = json.loads(
            (root / "config/runtime.lock.json").read_text(encoding="utf-8")
        )
        if not re.fullmatch(
            r"[^\s]+@sha256:[0-9a-f]{64}", lock["image"]
        ) or not re.fullmatch(r"[0-9a-f]{40}", lock["revision"]):
            problems.append("runtime artifacts must be digest/revision pinned")
    for name, (benchmarks, *_) in HEADLINES.items():
        if name in files and benchmarks in files:
            problems += headline_problems(
                name,
                (root / name).read_text(encoding="utf-8"),
                (root / benchmarks).read_text(encoding="utf-8"),
            )
    if "README.md" in files:
        problems += recipe_problems(
            (root / "README.md").read_text(encoding="utf-8"),
            {
                name: (root / name).read_text(encoding="utf-8")
                for name in files
                if name.startswith("docs/") and name.endswith(".md")
            },
        )
    if "pyproject.toml" in files:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]
        if not re.fullmatch(r"\d+\.\d+\.\d+", project["version"]):
            problems.append("expected release version")
        if project["license"] != "Apache-2.0":
            problems.append("unexpected project license")
    for problem in problems:
        print(problem)
    print(f"Publication audit: {len(files)} files, {len(problems)} issues")
    raise SystemExit(bool(problems))


if __name__ == "__main__":
    main()
