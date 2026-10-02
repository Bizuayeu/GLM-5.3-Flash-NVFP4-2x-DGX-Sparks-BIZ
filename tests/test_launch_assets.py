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

    def test_a_refused_gid_names_where_the_entry_is_now(self):
        hint = {"rail": 0, "configured_gid_index": 3, "roce_v2_gid_indices": [4]}
        failed = {
            "passed": False,
            "checks": {"rail_0_roce_v2_gid": False},
            "foreign_gpu_containers": [],
            "gid_hints": [hint],
        }
        with patch.object(launch_assets.server, "preflight", return_value=failed):
            with self.assertRaisesRegex(ValueError, '"roce_v2_gid_indices": \\[4\\]'):
                launch_assets.inspect({}, Path("profile.toml"), 1)

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


class WeightIndexTests(unittest.TestCase):
    """What the weight index and its shards must be before any image is read."""

    def refused(self, weight_map, shards=None):
        """Run inspect over a model holding ``shards`` (name -> bytes); return the error."""
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (model / name).write_text("{}", encoding="utf-8")
            for name, data in (shards or {}).items():
                (model / name).write_bytes(data)
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8"
            )
            with (
                patch.object(
                    launch_assets.server, "preflight", return_value={"passed": True}
                ),
                patch.object(launch_assets.server, "model_path", return_value=model),
                patch.object(launch_assets.host, "run") as run,
                self.assertRaises(Exception) as caught,
            ):
                launch_assets.inspect({}, Path("profile.toml"), 0)
            run.assert_not_called()
        return caught.exception

    def test_an_empty_weight_index_is_refused(self):
        error = self.refused({})
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "Empty model weight index")

    def test_a_shard_name_that_is_not_a_plain_safetensors_file_is_refused(self):
        for name in ("../x.safetensors", "a\\b.safetensors", "a.bin"):
            with self.subTest(name=name):
                error = self.refused({"t": name})
                self.assertIsInstance(error, ValueError)
                self.assertEqual(str(error), "Invalid model shard filename")

    def test_an_empty_shard_is_refused(self):
        error = self.refused({"t": "a.safetensors"}, {"a.safetensors": b""})
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "Missing or empty model shard")

    # Until 1.26.2 path.stat() ran before is_file(): an absent shard raised
    # FileNotFoundError and the "Missing" half of the message was unreachable.
    def test_a_missing_shard_is_refused_with_the_shard_message(self):
        error = self.refused({"t": "a.safetensors"})
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "Missing or empty model shard")
