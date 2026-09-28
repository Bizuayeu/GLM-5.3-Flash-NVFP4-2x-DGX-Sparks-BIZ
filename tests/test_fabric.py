import copy
import json
import tempfile
import unittest
from pathlib import Path

from glm53_setup import fabric, server_config


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
        self.assertEqual(
            fabric.fabric_env(self.site)["NCCL_IB_HCA"], "=roce0:1,roce1:2"
        )
        self.assertEqual(fabric.fabric_env(self.site)["NCCL_SOCKET_IFNAME"], "=fabric0")
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
                fabric.validate_site(site)

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

    def test_shifted_gid_is_found_but_still_refused(self):
        # 2026-09-27: after the peer's power loss, rail 0's IPv4 RoCE v2 GID
        # moved from index 3 to 4 with 3 left empty (stage5/INCIDENT.md).
        with tempfile.TemporaryDirectory() as tmp:
            sys_root = Path(tmp) / "sys"
            for rail in fabric.rails(self.site):
                port = (
                    sys_root
                    / "class/infiniband"
                    / rail["hca"]
                    / "ports"
                    / str(rail["port"])
                )
                index = 4 if rail["hca"] == "roce0" else 3
                entries = {
                    f"gids/{index}": "::ffff:" + rail["local_ip"],
                    f"gid_attrs/types/{index}": "RoCE v2",
                    f"gid_attrs/ndevs/{index}": rail["interface"],
                    # Same address as RoCE v1, and a link-local v2: not candidates.
                    "gids/2": "::ffff:" + rail["local_ip"],
                    "gid_attrs/types/2": "IB/RoCE v1",
                    "gid_attrs/ndevs/2": rail["interface"],
                    "gids/1": "fe80::1",
                    "gid_attrs/types/1": "RoCE v2",
                    "gid_attrs/ndevs/1": rail["interface"],
                }
                for name, value in entries.items():
                    (port / name).parent.mkdir(parents=True, exist_ok=True)
                    (port / name).write_text(value)
            checks = {"rail_0_roce_v2_gid": False, "rail_1_roce_v2_gid": True}
            self.assertEqual(
                fabric.gid_hints(self.site, checks, sys_root=sys_root),
                [
                    {
                        "rail": 0,
                        "hca": "roce0",
                        "port": 1,
                        "local_ip": "10.53.0.1",
                        "configured_gid_index": 3,
                        "roce_v2_gid_indices": [4],
                    }
                ],
            )
            self.assertEqual(
                fabric.gid_hints(
                    self.site, {"rail_0_roce_v2_gid": True}, sys_root=sys_root
                ),
                [],
            )
            missing = copy.deepcopy(self.site)
            missing["hca"] = "absent"
            self.assertEqual(
                fabric.gid_hints(missing, checks, sys_root=sys_root)[0][
                    "roce_v2_gid_indices"
                ],
                [],
            )

    def test_optional_toml_rails_preserve_single_input(self):
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        profile["nodes"][0]["additional_rails"] = self.site["additional_rails"]
        server_config.validate(profile)
