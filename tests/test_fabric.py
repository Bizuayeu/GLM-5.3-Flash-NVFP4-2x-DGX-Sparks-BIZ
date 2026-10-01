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
            gid = fabric.rail_check(0, "roce_v2_gid")
            checks = {gid: False, fabric.rail_check(1, "roce_v2_gid"): True}
            # The reader takes the writer's keys: checks() feeds gid_hints() as is.
            measured = fabric.checks(
                self.site, lambda *a: "[]", sys_root=sys_root, dev_root=sys_root
            )
            self.assertFalse(measured[gid])
            self.assertEqual(
                fabric.gid_hints(self.site, measured, sys_root=sys_root)[0]["rail"], 0
            )
            # One link-local address: no likely_cause, only the general fixes.
            (hint,) = fabric.gid_hints(self.site, checks, sys_root=sys_root)
            fixes = hint.pop("fixes")
            self.assertEqual(
                hint,
                {
                    "rail": 0,
                    "hca": "roce0",
                    "port": 1,
                    "local_ip": "10.53.0.1",
                    "configured_gid_index": 3,
                    "roce_v2_gid_indices": [4],
                },
            )
            self.assertIn("gid_index", fixes[0])
            self.assertIn("fabric0", fixes[1])
            self.assertEqual(
                fabric.gid_hints(self.site, {gid: True}, sys_root=sys_root),
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

    def test_second_link_local_names_stable_privacy(self):
        # MiaAI-Lab recipe #291: NetworkManager's stable-privacy address adds a
        # second IPv6 link-local, and the IPv4 entries move past its pair.
        with tempfile.TemporaryDirectory() as tmp:
            sys_root = Path(tmp) / "sys"
            port = sys_root / "class/infiniband/roce0/ports/1"
            entries = {}
            for index, (gid, kind) in enumerate(
                [
                    ("fe80::1", "IB/RoCE v1"),
                    ("fe80::1", "RoCE v2"),
                    ("fe80::2", "IB/RoCE v1"),
                    ("fe80::2", "RoCE v2"),
                    ("::ffff:10.53.0.1", "IB/RoCE v1"),
                    ("::ffff:10.53.0.1", "RoCE v2"),
                ]
            ):
                entries[f"gids/{index}"] = gid
                entries[f"gid_attrs/types/{index}"] = kind
                entries[f"gid_attrs/ndevs/{index}"] = "fabric0"
            for name, value in entries.items():
                (port / name).parent.mkdir(parents=True, exist_ok=True)
                (port / name).write_text(value)
            checks = {
                fabric.rail_check(0, "roce_v2_gid"): False,
                fabric.rail_check(1, "roce_v2_gid"): True,
            }
            (hint,) = fabric.gid_hints(self.site, checks, sys_root=sys_root)
            self.assertEqual(hint["roce_v2_gid_indices"], [5])
            self.assertEqual(hint["likely_cause"], "nm_stable_privacy")
            self.assertTrue(any("eui64" in fix for fix in hint["fixes"]))
            self.assertTrue(any("mlx5" in fix for fix in hint["fixes"]))
            # Still refused: the hint does not touch the check.
            self.assertFalse(checks[fabric.rail_check(0, "roce_v2_gid")])

    def test_gid_cause_needs_two_link_locals_and_a_moved_ipv4_entry(self):
        two = {
            0: "fe80::1",
            1: "fe80::1",
            2: "fe80::2",
            3: "fe80::2",
            4: "::ffff:10.53.0.1",
            5: "::ffff:10.53.0.1",
        }
        one = {1: "fe80::1", 2: "::ffff:10.53.0.1", 4: "::ffff:10.53.0.1"}
        for gids, indices, cause in (
            (two, [5], "nm_stable_privacy"),
            # v1 and v2 rows of one address are one link-local, not two.
            (one, [4], None),
            # No IPv4 RoCE v2 entry anywhere: an address or link fault.
            (two, [], None),
            ({**two, 6: "", 7: "not a gid"}, [5], "nm_stable_privacy"),
        ):
            with self.subTest(gids=gids, indices=indices):
                self.assertEqual(fabric.gid_cause(gids, indices), cause)

    def test_fabric_addresses_exclude_loopback_unspecified_and_multicast(self):
        for value, ok in (
            ("10.53.0.1", True),
            ("127.0.0.1", False),
            ("0.0.0.0", False),
            ("224.0.0.1", False),
        ):
            with self.subTest(value=value):
                self.assertEqual(fabric.is_fabric_ipv4(value), ok)

    def test_optional_toml_rails_preserve_single_input(self):
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        profile["nodes"][0]["additional_rails"] = self.site["additional_rails"]
        server_config.validate(profile)
