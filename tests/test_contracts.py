"""Facts written in more than one place, held to one value."""

import ast
import re
import unittest
from pathlib import Path

from glm53_setup import server, warmup
from glm53_setup import server_config as config
from glm53_setup.runtime import lpa
from glm53_setup.validation import run_apc_lpa_fixture, run_lpa
from tools import decode_check

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "glm53_setup/runtime"


def package_imports(path):
    """The glm53_setup names a module imports, as absolute ``module:name``."""
    package = "glm53_setup.runtime"
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0]
                module = f"{base}.{node.module}" if node.module else base
            elif (node.module or "").startswith("glm53_setup"):
                module = node.module
            else:
                continue
            found |= {f"{module}:{alias.name}" for alias in node.names}
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.startswith("glm53_setup")}
    return found


class MountedRuntimeTests(unittest.TestCase):
    # A mounted file runs over an image built earlier, so its package imports
    # resolve to that image's modules. A new import, or a name an older image's
    # module lacks, fails the launch; removing one is always safe.
    ALLOWED = {
        "lpa.py": {
            "glm53_setup.config:REVISION",
            "glm53_setup.config:TEACHER_PRECISION",
            "glm53_setup.runtime.apc_worker:report",
            "glm53_setup.runtime.graph_policy:lpa_execution_supported",
            "glm53_setup.runtime.lpa_query:ReferenceQueryMask",
        },
        "memory_probe.py": {
            "glm53_setup.runtime:fa2_attention",
            "glm53_setup.runtime.reference_attention:unpack_latent",
            "glm53_setup.runtime.fa2_attention:DECODE_MAX_ROWS",
        },
        "inductor_pin.py": set(),
        "fused_unpack.py": set(),
        "fa2_attention.py": {"glm53_setup.runtime.reference_attention:unpack_latent"},
        "reference_attention.py": {
            "glm53_setup.runtime.fused_unpack:unpack_latent_cuda",
            "glm53_setup.runtime.fa2_attention:sparse_nope_fa2",
            "glm53_setup.runtime.fa2_attention:use_fa2",
        },
    }

    def mounted(self):
        # FA2 excludes LPA, so every mount shows up across two launches.
        fa2 = config.load(ROOT / "examples/server.example.toml")
        fa2["runtime"]["fa2_attention"] = True
        fa2["runtime"]["inductor_deterministic"] = True
        fa2.setdefault("validation", {})["memory_probe"] = True
        with_lpa = config.load(ROOT / "examples/server.example.toml")
        with_lpa["lpa"]["enabled"] = True
        with_lpa["runtime"]["fa2_attention"] = False
        with_lpa["runtime"]["enforce_eager"] = True
        with_lpa["context"]["max_num_seqs"] = 1
        with_lpa["cache"]["prefix_caching"] = False
        names = set()
        for profile in (fa2, with_lpa):
            args = server.command(
                profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
            )
            for i, arg in enumerate(args):
                if arg == "-v":
                    source = Path(re.sub(r":/[^:]*(:ro)?$", "", args[i + 1]))
                    if source.parent == RUNTIME and source.suffix == ".py":
                        names.add(source.name)
        return names

    def test_every_mounted_runtime_file_is_listed(self):
        self.assertEqual(self.mounted(), set(self.ALLOWED))

    def test_mounted_files_import_only_what_older_images_carry(self):
        for name, allowed in self.ALLOWED.items():
            with self.subTest(name=name):
                imported = package_imports(RUNTIME / name)
                self.assertLessEqual(imported, allowed, "new import in a mounted file")


class ImagePackageDirTests(unittest.TestCase):
    def test_the_mount_target_is_where_the_image_copies_the_package(self):
        dockerfile = (ROOT / "docker/Dockerfile.reference").read_text(encoding="utf-8")
        copied = re.search(r"(?m)^COPY glm53_setup (\S+)$", dockerfile)
        self.assertEqual(copied[1], server.IMAGE_PACKAGE_DIR)


class LpaMtpDepthTests(unittest.TestCase):
    # The settings, the two LPA workers and the fixture CLIs must accept the
    # same depths; a depth the settings pass and a worker refuses fails every
    # LPA request after an otherwise healthy start.
    def test_settings_accept_exactly_the_worker_depths(self):
        profile = config.load(ROOT / "examples/server.example.toml")
        profile["lpa"]["enabled"] = True
        profile["runtime"]["fa2_attention"] = False
        profile["runtime"]["enforce_eager"] = True
        profile["context"]["max_num_seqs"] = 1
        profile["cache"]["prefix_caching"] = False
        profile["mtp"]["enabled"] = True
        accepted = set()
        for depth in range(1, 6):
            profile["mtp"]["num_speculative_tokens"] = depth
            try:
                config.validate(profile)
            except ValueError:
                continue
            accepted.add(depth)
        self.assertEqual(accepted, set(lpa.LPA_MTP_DEPTHS))

    def test_the_fixture_clis_offer_the_same_depths(self):
        for runner in (run_lpa, run_apc_lpa_fixture):
            with self.subTest(runner=runner.__name__):
                action = next(
                    a for a in runner.parser()._actions if "--mtp" in a.option_strings
                )
                self.assertEqual(set(action.choices), set(lpa.LPA_MTP_DEPTHS))

    def test_the_baked_apc_worker_keeps_the_same_literal(self):
        # apc_worker.py is baked into the image and is not mounted; importing the
        # constant from the mounted lpa.py would tie the two across images, so it
        # keeps a literal that this test holds to the owner.
        tree = ast.parse((RUNTIME / "apc_worker.py").read_text(encoding="utf-8"))
        literals = [
            ast.literal_eval(node.comparators[0])
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "num_speculative_tokens"
        ]
        self.assertEqual(literals, [tuple(lpa.LPA_MTP_DEPTHS)])


DOCS = ("docs/server-configuration.md", "docs/server-configuration.ja.md")


def doc_table(name, header):
    """The rows of the table that starts at ``header``, as first cell -> rest."""
    text = (ROOT / name).read_text(encoding="utf-8")
    lines = text[text.index(header) :].splitlines()[2:]
    rows = {}
    for line in lines:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0]] = " | ".join(cells[1:])
    return rows


class MarkerTableTests(unittest.TestCase):
    """The image-contract table states what the code and the Dockerfile do."""

    HEADERS = ("| Marker |", "| marker |")
    # Switch defaults the runtime reads; the table lists capability markers only.
    SWITCH_DEFAULTS = {
        "GLM53_CANONICAL_CANDIDATES=1",
        "GLM53_CANONICAL_MOE_ORDER=1",
        "GLM53_STABLE_INDEXER_TOPK=1",
    }

    def enabled_profile(self):
        profile = config.load(ROOT / "examples/server.example.toml")
        profile["runtime"].update(
            pipeline_parallel_size=2,
            expert_parallel=True,
            decode_graphs=True,
            canonical_moe_order=True,
            stable_indexer_topk=True,
            prefix_page_dedup=True,
            fa2_attention=True,
        )
        profile["validation"]["component_worker"] = True
        profile["cache"].update(fused_unpack=True, prefix_caching=True)
        profile["lpa"]["enabled"] = True
        return profile

    def test_each_language_lists_every_marker_with_the_check_that_needs_it(self):
        dockerfile = (ROOT / "docker/Dockerfile.reference").read_text(encoding="utf-8")
        env = re.findall(r"^ENV (GLM53_\w+=\S+)$", dockerfile, re.MULTILINE)
        profile = self.enabled_profile()
        for name, header in zip(DOCS, self.HEADERS, strict=True):
            rows = {
                marker.strip("`"): cell
                for marker, cell in doc_table(name, header).items()
            }
            with self.subTest(document=name):
                self.assertEqual(set(rows), set(env) - self.SWITCH_DEFAULTS)
            for marker, cell in rows.items():
                without = [m for m in env if m != marker]
                failing = [
                    check
                    for check, ok in config.image_capability_checks(
                        profile, {"Config": {"Env": without}}
                    ).items()
                    if not ok
                ]
                with self.subTest(document=name, marker=marker):
                    if failing:
                        self.assertIn(f"`{failing[0]}`", cell)
                    else:
                        self.assertRegex(cell, r"^(Never|要求しない)")


class TemplateTableTests(unittest.TestCase):
    """The template summary quotes the distributed TOML's values."""

    HEADERS = ("| Item | Default |", "| 項目 | 既定値 |")

    def test_the_summary_values_are_the_example_values(self):
        p = config.load(ROOT / "examples/server.example.toml")
        expected = [
            f"{p['context']['max_model_len']:,}",
            f"chunk {p['context']['max_num_batched_tokens']}",
            f"{p['cache']['kv_cache_memory_bytes'] / 2**30:g} GiB",
            f"{p['cache']['mm_processor_cache_gb']:g} GiB",
            f"MTP k={p['mtp']['num_speculative_tokens']}",
            f"nccl_channels = {p['runtime']['nccl_channels']}",
            f"max_tokens={p['generation']['max_tokens']}",
            f"{p['resources']['container_memory_gib']} GiB",
            f"{p['resources']['minimum_available_gib']} GiB",
            f"{p['resources']['reserve_gib']:g} GiB",
            f"run_seconds={p['resources']['run_seconds']}",
            f"stall_seconds={p['resources']['stall_seconds']}",
            f"warmup_long_tokens={p['generation']['warmup_long_tokens']}",
        ]
        for name, header in zip(DOCS, self.HEADERS, strict=True):
            table = " ".join(doc_table(name, header).values())
            # The Japanese page writes units without the space.
            table = re.sub(r"(\d)(GiB)", r"\1 \2", table)
            for value in expected:
                with self.subTest(document=name, value=value):
                    self.assertIn(value, table)


class SpecMetricTests(unittest.TestCase):
    def test_warmup_reads_the_counters_the_decode_check_reads(self):
        # tools/decode_check.py runs without the package, so it keeps its own table.
        for name, key in warmup.SPEC_METRICS.items():
            self.assertEqual(decode_check.COUNTERS.get(name), key)


if __name__ == "__main__":
    unittest.main()
