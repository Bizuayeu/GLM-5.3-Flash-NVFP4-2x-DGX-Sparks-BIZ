import copy
import json
import tempfile
import unittest
from pathlib import Path

from glm53_setup import fabric, host, server_config


class FabricTests(unittest.TestCase):
    def setUp(self):
        self.site = {
            "rank": 0,
            "head_ip": "10.53.0.1",
            "local_ip": "10.53.0.1",
            "interface": "fabric0",
            "hca": "roce0",
            "gid_index": 3,
            "api_port": 8891,
            "master_port": 29553,
            "additional_rails": [
                {
                    "hca": "roce1",
                    "port": 2,
                    "interface": "fabric1",
                    "local_ip": "10.54.0.1",
                    "gid_index": 3,
                }
            ],
        }

    def test_all_ports_explicit_and_common_gid_required(self):
        self.assertEqual(host.fabric_env(self.site)["NCCL_IB_HCA"], "=roce0:1,roce1:2")
        self.assertEqual(host.fabric_env(self.site)["NCCL_SOCKET_IFNAME"], "=fabric0")
        for change in [
            {"gid_index": 4},
            {"hca": "roce1,"},
            {"hca": ""},
            {"port": 0},
            {"interface": "fabric0"},
            {"local_ip": "10.53.0.1"},
        ]:
            site = copy.deepcopy(self.site)
            site["additional_rails"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                host.validate_site(site)

    def test_second_rail_faults_cannot_hide_behind_first_rail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dev/infiniband").mkdir(parents=True)
            files = {}
            for rail in fabric.rails(self.site):
                port = (
                    root
                    / "sys/class/infiniband"
                    / rail["hca"]
                    / "ports"
                    / str(rail["port"])
                )
                files.update(
                    {
                        root / "sys/class/net" / rail["interface"] / "operstate": "up",
                        port / "state": "4: ACTIVE",
                        port / "phys_state": "5: LinkUp",
                        port / "link_layer": "Ethernet",
                        port / "gids/3": "::ffff:" + rail["local_ip"],
                        port / "gid_attrs/types/3": "RoCE v2",
                        port / "gid_attrs/ndevs/3": rail["interface"],
                    }
                )
            for path, value in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)

            def run(*args):
                ip = "10.53.0.1" if args[-1] == "fabric0" else "10.54.0.1"
                return json.dumps([{"addr_info": [{"local": ip}]}])

            def check():
                return fabric.checks(
                    self.site, run, sys_root=root / "sys", dev_root=root / "dev"
                )

            self.assertTrue(all(check().values()))
            port = root / "sys/class/infiniband/roce1/ports/2"
            for path, bad in [
                (port / "gids/3", ""),
                (port / "gids/3", "::"),
                (port / "gids/3", "::ffff:10.54.0.99"),
                (port / "gid_attrs/types/3", "RoCE v1"),
                (port / "gid_attrs/ndevs/3", "fabric0"),
                (port / "state", "1: DOWN"),
            ]:
                path.write_text(bad)
                result = check()
                self.assertTrue(
                    all(v for k, v in result.items() if k.startswith("rail_0_"))
                )
                self.assertFalse(all(result.values()), (path, bad))
                path.write_text(files[path])
            missing = copy.deepcopy(self.site)
            missing["additional_rails"][0]["hca"] = "absent"
            self.assertFalse(
                all(
                    fabric.checks(
                        missing, run, sys_root=root / "sys", dev_root=root / "dev"
                    ).values()
                )
            )

    def test_optional_toml_rails_preserve_single_input(self):
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        profile["nodes"][0]["additional_rails"] = self.site["additional_rails"]
        server_config.validate(profile)
