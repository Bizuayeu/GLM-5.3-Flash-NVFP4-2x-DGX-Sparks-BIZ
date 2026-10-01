import copy
import importlib
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from glm53_setup import server
from glm53_setup import server_config as config
from glm53_setup.runtime import shm_spin

ROOT = Path(__file__).resolve().parents[1]
TARGET = "fakevllm.shm_broadcast"


class HookTests(unittest.TestCase):
    def setUp(self):
        # A stand-in for vllm.distributed.device_communicators.shm_broadcast (385dce36).
        self.directory = tempfile.TemporaryDirectory()
        package = Path(self.directory.name) / "fakevllm"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "shm_broadcast.py").write_text(
            textwrap.dedent(
                """
                class SpinCondition:
                    def __init__(self, is_reader, context, notify_address, busy_loop_s=1):
                        self.busy_loop_s = busy_loop_s if is_reader else 0
                """
            ),
            encoding="utf-8",
        )
        sys.path.insert(0, self.directory.name)
        self.meta_path = list(sys.meta_path)

    def tearDown(self):
        sys.meta_path[:] = self.meta_path
        sys.path.remove(self.directory.name)
        for name in ("fakevllm", TARGET):
            sys.modules.pop(name, None)
        self.directory.cleanup()

    def test_does_nothing_unless_a_spin_is_set(self):
        self.assertEqual(shm_spin.install({}, target=TARGET), "off")
        self.assertEqual(sys.meta_path, self.meta_path)
        module = importlib.import_module(TARGET)
        self.assertEqual(module.SpinCondition(True, None, "").busy_loop_s, 1)

    def test_sets_the_reader_default_when_the_module_is_first_imported(self):
        state = shm_spin.install({shm_spin.ENV: "0.002"}, target=TARGET)
        self.assertEqual(state, "waiting")
        self.assertNotIn(TARGET, sys.modules)
        module = importlib.import_module(TARGET)
        # The reader constructor passes no value, so the default is what it gets.
        self.assertEqual(module.SpinCondition(True, None, "").busy_loop_s, 0.002)
        self.assertEqual(module.SpinCondition(False, None, "").busy_loop_s, 0)
        # One shot: the finder leaves the import machinery once it has set.
        self.assertEqual(sys.meta_path, self.meta_path)
        self.assertIsNone(shm_spin.SetAfterImport(TARGET, 1.0).find_spec("json", None))

    def test_sets_at_once_when_the_module_is_already_imported(self):
        module = importlib.import_module(TARGET)
        self.assertEqual(shm_spin.install({shm_spin.ENV: "1"}, target=TARGET), "set")
        self.assertEqual(module.SpinCondition.__init__.__defaults__, (1.0,))
        self.assertEqual(sys.meta_path, self.meta_path)

    def test_refuses_a_spin_condition_other_than_the_pinned_one(self):
        # Another default means another vLLM: stop rather than guess.
        module = importlib.import_module(TARGET)
        module.SpinCondition.__init__.__defaults__ = (0.5,)
        with self.assertRaisesRegex(RuntimeError, "SpinCondition"):
            shm_spin.install({shm_spin.ENV: "0.002"}, target=TARGET)

    def test_the_pth_file_is_one_import_line(self):
        lines = shm_spin.PTH.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["import glm53_setup.runtime.shm_spin"])


class SettingTests(unittest.TestCase):
    def setUp(self):
        self.profile = config.load(ROOT / "examples/server.example.toml")

    def command(self, profile):
        return server.command(
            profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
        )

    def test_absent_key_adds_no_hook_and_keeps_the_fingerprint(self):
        # The templates set the key from 1.26.0; a profile without it launches as before.
        self.assertEqual(self.profile["runtime"].pop("shm_spin_seconds"), 0.002)
        command = self.command(self.profile)
        self.assertFalse(any("shm_spin" in arg for arg in command))
        self.assertFalse(any(arg.startswith(shm_spin.ENV) for arg in command))
        with_key = copy.deepcopy(self.profile)
        with_key["runtime"]["shm_spin_seconds"] = 1
        config.validate(with_key)
        self.assertNotEqual(
            config.fingerprint(with_key), config.fingerprint(self.profile)
        )

    def test_the_key_mounts_the_hook_and_sets_its_variable_on_every_rank(self):
        self.profile["runtime"]["shm_spin_seconds"] = 0.002
        config.validate(self.profile)
        for rank in (0, 1):
            self.assertEqual(
                config.environment(self.profile, rank)[shm_spin.ENV], "0.002"
            )
        command = self.command(self.profile)
        self.assertIn(f"{shm_spin.ENV}=0.002", command)
        for target in (
            f":{server.IMAGE_PACKAGE_DIR}/runtime/shm_spin.py:ro",
            f":{server.SITE_PACKAGES}/glm53-shm-spin.pth:ro",
        ):
            self.assertTrue(any(v.endswith(target) for v in command), target)
        # No image support is needed: the hook is the checkout's mounted copy.
        self.assertNotIn(
            "shm_spin",
            json.dumps(
                config.image_capability_checks(
                    self.profile, {"Config": {"Env": ["GLM53_REFERENCE_ATTENTION=1"]}}
                )
            ),
        )

    def test_values_outside_the_measured_span_are_refused(self):
        for good in (0.002, 0.5, 1, 1.0):
            with self.subTest(good=good):
                profile = copy.deepcopy(self.profile)
                profile["runtime"]["shm_spin_seconds"] = good
                config.validate(profile)
        for bad in (0, 0.001, 1.5, -1, float("nan"), float("inf"), True, "1", None):
            with self.subTest(bad=bad):
                profile = copy.deepcopy(self.profile)
                profile["runtime"]["shm_spin_seconds"] = bad
                with self.assertRaisesRegex(ValueError, "shm_spin_seconds"):
                    config.validate(profile)


if __name__ == "__main__":
    unittest.main()
