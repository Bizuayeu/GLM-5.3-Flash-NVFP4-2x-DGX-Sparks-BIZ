"""What the source-pinned vLLM patches share: anchors, the hash gate and the command.

A single-target ``runtime/patch_*`` module names one file of the pinned vLLM
(its target), the SHA-256 of that file as pinned, and a pure ``patch_text(text)``
that refuses a drifted or already patched source. ``patch_kpool_ring`` changes
two hash-pinned files through ``prepare_files`` and ``main_files``;
``patch_apc_lpa`` and ``patch_nope_reference`` change several files with flags
of their own and use only ``replace_once`` and ``default_package``. The image build runs each module as
``python3 -m glm53_setup.runtime.patch_X``; this module holds the parts that
are the same for all of them, so a patch module states only what it changes.
"""

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path


def replace_once(text, old, new):
    """Replace an anchor that occurs exactly once; anything else is source drift."""
    if text.count(old) != 1:
        raise ValueError("Patch anchor is not unique; refusing source drift")
    return text.replace(old, new, 1)


def replace_exactly(text, old, new, count):
    """Like ``replace_once`` for an anchor the source repeats a known number of times."""
    if text.count(old) != count:
        raise ValueError("Patch anchor count differs; refusing source drift")
    return text.replace(old, new)


def default_package():
    """The installed vLLM package, when ``--package`` names none."""
    return Path(sysconfig.get_paths()["purelib"]) / "vllm"


def prepare(package, target, sha256, mismatch, patch_text):
    """Read the pinned file, refuse any other content, and return the patched bytes."""
    original = (package / target).read_bytes()
    if hashlib.sha256(original).hexdigest() != sha256:
        raise ValueError(mismatch)
    return patch_text(original.decode("utf-8")).encode("utf-8")


def prepare_files(package, sources, patches, mismatch):
    """Check every file against its pinned hash before patching any.

    ``sources`` maps each target to its SHA-256, ``patches`` to its pure
    ``patch_text``; a mismatch raises ``mismatch`` followed by the target.
    """
    originals = {name: (package / name).read_bytes() for name in sources}
    for name, data in originals.items():
        if hashlib.sha256(data).hexdigest() != sources[name]:
            raise ValueError(mismatch + name)
    return {
        name: patches[name](data.decode("utf-8")).encode("utf-8")
        for name, data in originals.items()
    }


def main_files(argv, *, doc, sources, prepare, record):
    """The ``--package``/``--check`` command of a patch that changes several files.

    As ``main``, but the record beside the package maps each target to its
    source and patched hashes, and the printed line adds ``check_only``.
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    package = args.package or default_package()
    outputs = prepare(package)
    row = {
        name: {
            "source_sha256": sources[name],
            "patched_sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in outputs.items()
    }
    if not args.check:
        for name, data in outputs.items():
            (package / name).write_bytes(data)
        (package.parent / record).write_text(json.dumps(row, indent=2))
    print(json.dumps({"files": row, "check_only": args.check}))


def main(argv, *, doc, target, sha256, prepare, record):
    """The ``--package``/``--check`` command every single-target patch exposes.

    ``--check`` reports the hashes without writing. Otherwise the patched file
    replaces the pinned one and ``record`` is written beside the package, so an
    image carries the evidence of what was applied to it.
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    package = args.package or default_package()
    patched = prepare(package)
    row = {
        "source_sha256": sha256,
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "check_only": args.check,
    }
    if not args.check:
        (package / target).write_bytes(patched)
        (package.parent / record).write_text(json.dumps(row, indent=2))
    print(json.dumps(row))
