import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import launch_assets


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
