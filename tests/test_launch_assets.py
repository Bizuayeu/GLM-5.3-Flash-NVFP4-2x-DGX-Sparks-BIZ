import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import launch_assets, server_config
from glm53_setup.config import ROOT


class LaunchAssetTests(unittest.TestCase):
    def test_missing_tokenizer_is_a_static_failure_before_any_image_or_stop_action(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            with (
                patch.object(
                    launch_assets.server, "preflight", return_value={"passed": True}
                ),
                patch.object(launch_assets.server, "model_path", return_value=model),
                patch.object(launch_assets.host, "run") as run,
            ):
                with self.assertRaisesRegex(ValueError, "Tokenizer assets"):
                    launch_assets.inspect({}, Path("profile.toml"), 0)
                run.assert_not_called()

    def test_what_a_recovery_target_was_allowed_reaches_the_switch_record(self):
        profile = server_config.load(ROOT / "examples/server.example.toml")
        warning = "moe_order_marker_1_accepted_for_recovery"
        image = [{"Id": "sha256:" + "0" * 64, "Config": {"Env": []}}]
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                (model / name).write_text("{}", encoding="utf-8")
            (model / "a.safetensors").write_bytes(b"w")
            (model / "model.safetensors.index.json").write_text(
                '{"weight_map": {"t": "a.safetensors"}}', encoding="utf-8"
            )
            for warnings, recovery in (([warning], True), ([], False)):
                passed = {"passed": True, "warnings": warnings}
                with (
                    patch.object(
                        launch_assets.server, "preflight", return_value=passed
                    ) as preflight,
                    patch.object(
                        launch_assets.server, "model_path", return_value=model
                    ),
                    patch.object(
                        launch_assets.host, "run", return_value=json.dumps(image)
                    ),
                ):
                    result = launch_assets.inspect(
                        profile, Path("profile.toml"), 1, recovery=recovery
                    )
                self.assertIs(preflight.call_args.kwargs["recovery"], recovery)
                self.assertEqual(result["local"]["warnings"], warnings)
                self.assertNotIn("warnings", result["common"])
