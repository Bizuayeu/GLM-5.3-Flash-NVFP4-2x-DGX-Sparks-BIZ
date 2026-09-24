import importlib
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from glm53_setup import server
from glm53_setup import server_config as config
from glm53_setup.runtime import inductor_pin

ROOT = Path(__file__).resolve().parents[1]
UNSET = object()


class PinTests(unittest.TestCase):
    def setUp(self):
        # A stand-in for torch._inductor.config: one entry with torch's shape.
        self.directory = tempfile.TemporaryDirectory()
        package = Path(self.directory.name) / "fakeinductor"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "config.py").write_text(
            textwrap.dedent(
                """
                from types import SimpleNamespace
                UNSET = object()
                _config = {"deterministic": SimpleNamespace(env_value_force=UNSET)}
                """
            ),
            encoding="utf-8",
        )
        sys.path.insert(0, self.directory.name)
        self.meta_path = list(sys.meta_path)

    def tearDown(self):
        sys.meta_path[:] = self.meta_path
        sys.path.remove(self.directory.name)
        for name in ("fakeinductor", "fakeinductor.config"):
            sys.modules.pop(name, None)
        self.directory.cleanup()

    def test_does_nothing_unless_the_deterministic_mode_is_asked_for(self):
        for environ in ({}, {"TORCHINDUCTOR_DETERMINISTIC": "0"}):
            self.assertEqual(
                inductor_pin.install(environ, target="fakeinductor.config"), "off"
            )
        self.assertEqual(sys.meta_path, self.meta_path)

    def test_pins_the_entry_when_the_config_module_is_first_imported(self):
        # Installed at interpreter start, before vLLM sets the TORCHINDUCTOR_*
        # variables the config module reads on import: it must not import it.
        state = inductor_pin.install(
            {"TORCHINDUCTOR_DETERMINISTIC": "1"}, target="fakeinductor.config"
        )
        self.assertEqual(state, "waiting")
        self.assertNotIn("fakeinductor.config", sys.modules)
        module = importlib.import_module("fakeinductor.config")
        self.assertIs(module._config["deterministic"].env_value_force, True)
        # One shot: the finder leaves the import machinery once it has pinned.
        self.assertEqual(sys.meta_path, self.meta_path)
        # Other imports pass through untouched.
        self.assertIsNone(
            inductor_pin.PinAfterImport("fakeinductor.config").find_spec("json", None)
        )

    def test_pins_at_once_when_the_module_is_already_imported(self):
        module = importlib.import_module("fakeinductor.config")
        state = inductor_pin.install(
            {"TORCHINDUCTOR_DETERMINISTIC": "1"}, target="fakeinductor.config"
        )
        self.assertEqual(state, "pinned")
        self.assertIs(module._config["deterministic"].env_value_force, True)
        self.assertEqual(sys.meta_path, self.meta_path)

    def test_the_pth_file_is_one_import_line(self):
        # site executes a .pth line that starts with "import"; nothing else runs.
        lines = inductor_pin.PTH.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["import glm53_setup.runtime.inductor_pin"])


class LaunchTests(unittest.TestCase):
    def test_the_key_mounts_the_pin_and_its_pth_and_nothing_without_it(self):
        profile = config.load(ROOT / "examples/server.example.toml")
        off = server.command(
            profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
        )
        self.assertFalse(any("inductor_pin" in v for v in off))
        profile["runtime"]["inductor_deterministic"] = True
        config.validate(profile)
        command = server.command(
            profile, ROOT / "state/server.toml", 0, "c", ROOT / "state/test-hf"
        )
        for target in (
            ":/opt/glm53/glm53_setup/runtime/inductor_pin.py:ro",
            ":/usr/local/lib/python3.12/dist-packages/glm53-inductor-pin.pth:ro",
        ):
            self.assertTrue(any(v.endswith(target) for v in command), target)


if __name__ == "__main__":
    unittest.main()
