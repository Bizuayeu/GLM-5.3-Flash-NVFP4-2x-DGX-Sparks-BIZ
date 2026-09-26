"""Contracts for the profile validator and the serve-argument assembly.

``validate`` was 234 lines: a recursive schema walk with the per-category
optional keys buried inside its closure, followed by some thirty cross-field
rules in one run. The rules are order-dependent -- the first raise is the
sentence the operator reads -- so they are split by position, never by theme,
and the order lives in one readable sequence.
"""

import ast
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]


def profile():
    return config.load(ROOT / "examples/server.example.toml")


class OptionalKeyTests(unittest.TestCase):
    def test_the_optional_keys_are_readable_without_entering_the_walker(self):
        self.assertIn("server.runtime", config.OPTIONAL_KEYS)
        self.assertIn("decode_graphs", config.OPTIONAL_KEYS["server.runtime"])
        self.assertIn("memory_probe", config.OPTIONAL_KEYS["server.validation"])
        for path, keys in config.OPTIONAL_KEYS.items():
            with self.subTest(path=path):
                self.assertTrue(path.startswith("server"))
                self.assertIsInstance(keys, frozenset | set)

    def test_every_optional_key_may_be_absent_and_is_still_type_checked(self):
        for path, keys in config.OPTIONAL_KEYS.items():
            section = path.split(".", 1)[1] if "." in path else None
            if section is None or section not in profile():
                continue
            for key in keys:
                with self.subTest(section=section, key=key):
                    without = profile()
                    without[section].pop(key, None)
                    config.validate(without)

    def test_every_default_belongs_to_an_optional_key(self):
        for section, defaults in config.OPTIONAL_DEFAULTS.items():
            with self.subTest(section=section):
                self.assertLessEqual(
                    defaults.keys(), config.OPTIONAL_KEYS["server." + section]
                )

    def test_an_absent_key_reads_as_its_default_and_a_present_one_as_written(self):
        value = profile()
        value["runtime"].pop("vision", None)
        self.assertIs(config.optional(value, "runtime", "vision"), False)
        value["runtime"]["vision"] = True
        self.assertIs(config.optional(value, "runtime", "vision"), True)

    def test_keys_whose_absence_leaves_the_choice_to_the_image_have_no_default(self):
        # Absent, the launcher sets no environment and the image's patch decides.
        for key in ("canonical_moe_order", "stable_indexer_topk"):
            self.assertNotIn(key, config.OPTIONAL_DEFAULTS["runtime"])


class CheckOrderTests(unittest.TestCase):
    def test_the_validator_runs_its_checks_in_one_declared_order(self):
        names = [check.__name__ for check in config.VALIDATORS]
        self.assertEqual(names[0], "check_schema")
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            with self.subTest(check=name):
                self.assertTrue(name.startswith("check_"))

    def test_an_earlier_rule_wins_over_a_later_one(self):
        # A profile that breaks the schema and a cross-field rule must report
        # the schema, because that is what the operator mistyped first.
        broken = profile()
        broken["context"]["typo"] = 1
        broken["generation"]["temperature"] = -1
        with self.assertRaises(ValueError) as caught:
            config.validate(broken)
        self.assertIn("Unknown/missing settings", str(caught.exception))

    def test_every_check_is_callable_on_its_own(self):
        good = profile()
        for check in config.VALIDATORS:
            with self.subTest(check=check.__name__):
                check(good)

    def test_the_validator_calls_each_check_exactly_once(self):
        # load() validates on the way in, so the profile is built first.
        current = profile()
        expected = [check.__name__ for check in config.VALIDATORS]
        calls = []
        wrapped = tuple(
            (lambda name: lambda p: calls.append(name))(name) for name in expected
        )
        with patch.object(config, "VALIDATORS", wrapped):
            config.validate(current)
        self.assertEqual(calls, expected)


class CheckMessageTests(unittest.TestCase):
    CASES = [
        ("check_optional_shapes", ("runtime", "vision"), "x", "runtime.vision"),
        ("check_optional_shapes", ("runtime", "nccl_channels"), 0, "nccl_channels"),
        ("check_pinned_identity", ("schema_version",), 2, "schema_version"),
        ("check_magnitudes", ("resources", "run_seconds"), -1, "run_seconds"),
        ("check_generation", ("generation", "temperature"), -1, "temperature"),
        ("check_speculation", ("mtp", "num_speculative_tokens"), 6, "MTP depth"),
        ("check_lpa", ("lpa", "cut"), 45, "LPA cut"),
        (
            "check_lpa",
            ("mtp", "num_speculative_tokens"),
            4,
            "LPA with MTP",
            {("lpa", "enabled"): True},
        ),
        ("check_identifiers", ("api", "served_model_name"), "bad name", "api."),
    ]

    def test_each_rule_reports_from_the_check_that_owns_it(self):
        # An optional fifth element sets what the rule needs besides the edit.
        by_name = {check.__name__: check for check in config.VALIDATORS}
        for name, path, value, fragment, *preset in self.CASES:
            with self.subTest(check=name, key=".".join(path)):
                bad = profile()
                for (section, key), setting in (preset[0] if preset else {}).items():
                    bad[section][key] = setting
                target = bad
                for step in path[:-1]:
                    target = target[step]
                target[path[-1]] = value
                with self.assertRaises(ValueError) as caught:
                    by_name[name](bad)
                self.assertIn(fragment, str(caught.exception))


class ServeArgumentTests(unittest.TestCase):
    def test_the_assembly_steps_are_declared_in_one_sequence(self):
        names = [step.__name__ for step in config.SERVE_STEPS]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            with self.subTest(step=name):
                self.assertTrue(name.startswith("apply_"))

    def test_the_steps_together_reproduce_the_assembled_arguments(self):
        for edits in ({}, {"vision": True}, {"expert_parallel": True}):
            with self.subTest(edits=edits):
                current = profile()
                current["runtime"].update(edits)
                if edits:
                    current["runtime"]["decode_graphs"] = False
                    current["runtime"]["enforce_eager"] = True
                    current["context"]["max_num_seqs"] = 1
                expected = config.serve_args(current, 0, "/model")
                built = config.serve_template(config.site(current, 0), "/model")
                for step in config.SERVE_STEPS:
                    step(built, current)
                self.assertEqual(built, expected)

    def test_the_settings_layer_does_not_import_the_host_module(self):
        # host runs docker and reads /proc; the argv template and the site
        # checks the settings need are pure, and live here and in fabric.
        tree = ast.parse(Path(config.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level and node.module
        }
        self.assertNotIn("host", imported)

    def test_a_step_that_needs_no_work_leaves_the_arguments_alone(self):
        plain = profile()
        plain["profiling"]["enabled"] = False
        args = config.serve_args(plain, 0, "/model")
        before = copy.deepcopy(args)
        config.apply_profiling(args, plain)
        self.assertEqual(args, before)


if __name__ == "__main__":
    unittest.main()
