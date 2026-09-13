import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import cluster, startup, startup_config


class ClusterOwnershipTests(unittest.TestCase):
    def test_cancelled_reserved_job_cannot_launch_after_transport_returns(self):
        profile = startup_config.load(
            Path(__file__).resolve().parents[1] / "examples/startup.example.toml"
        )
        launch = {
            "manifest": startup.freeze(profile, {}),
            "config_path": "/srv/glm53/state/startup.toml",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            identity = cluster.rpc("reserve", 0, launch)
            with patch.object(cluster, "inspect_attempt", return_value=None):
                cluster.rpc("stop", 0, identity)
            with patch.object(startup, "main") as start:
                cluster.main(["job", "--record", identity["record"]])
                start.assert_not_called()
            finished = startup.read_json(Path(identity["record"]) / "finished.json")
            self.assertEqual(finished["status"], "cancelled before launch")
            with self.assertRaises(ValueError):
                cluster.rpc("start", 0, identity)

    def test_swapped_attempt_record_is_not_allowed_to_stop_a_container(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cluster, "ROOT", Path(tmp)),
        ):
            identity = cluster.rpc("reserve", 0, {"manifest": {"fingerprint": "test"}})
            identity["name"] = "unrelated-workload"
            with (
                patch.object(cluster, "inspect_attempt") as inspect,
                self.assertRaises(ValueError),
            ):
                cluster.rpc("stop", 0, identity)
            inspect.assert_not_called()

    def test_ssh_alias_cannot_be_a_command_line_option(self):
        with self.assertRaises(ValueError):
            cluster.SSHBackend(["-oProxyCommand=bad", "peer"], "/srv/model", None, 30)
