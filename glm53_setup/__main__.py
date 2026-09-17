"""One checkout-local CLI; GPU dependencies are imported only by GPU commands."""

import argparse
import importlib
import sys

from .config import version

COMMANDS = {
    "download": "download",
    "verify-download": "verify_download",
    "prepare-image": "images",
    "build-reference": "build_reference",
    "server": "server",
    "cluster": "cluster",
    "fixture-build": "validation.make_fixture",
    "fixture-run": "validation.run_fixture",
    "fixture-assess": "validation.summarize_fixture",
    "inspect-runtime": "validation.inspect_runtime",
    "probe-attention": "validation.probe_attention",
    "test-reference": "validation.reference_check",
    "patch-reference": "runtime.patch_nope_reference",
    "lpa-fixture": "validation.run_lpa",
    "apc-lpa-fixture": "validation.run_apc_lpa_fixture",
    "apc-lpa-benchmark": "validation.benchmark_apc_lpa",
    "apc-history": "validation.apc_history",
    "lpa-corpus": "validation.lpa_corpus",
    "lpa-train": "validation.train_lpa",
    "freedombench": "validation.freedombench",
    "profile-assess": "validation.profile_trace",
    "indexer-overlap": "validation.indexer_overlap",
    "quant-error": "validation.quant_error",
    "agreement-fixture": "validation.run_agreement_fixture",
    "agreement-compare": "validation.compare_agreement",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m glm53_setup", description=__doc__)
    parser.add_argument("command", choices=COMMANDS, nargs="?")
    parser.add_argument("--version", action="version", version=version())
    if not argv or argv[0] in ("-h", "--help", "--version"):
        parser.parse_args(argv)
        if not argv:
            parser.print_help()
        return
    if argv[0] not in COMMANDS:
        parser.error("unknown command: " + argv[0])
    target = COMMANDS[argv[0]]
    module = importlib.import_module("glm53_setup." + target)
    module.main(argv[1:])


if __name__ == "__main__":
    main()
