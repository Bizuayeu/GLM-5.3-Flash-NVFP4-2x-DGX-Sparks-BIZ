"""Three or more nodes on a switchless QSFP ring: schema, addressing and launch (TP=3 plan, Stage 2)."""

import contextlib
import copy
import io
import json
import re
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import fabric, server
from glm53_setup import server_config as config
from glm53_setup.runtime.tp_padding import HEADS

ROOT = Path(__file__).resolve().parents[1]
RING = ROOT / "examples/server.tp3.example.toml"
DEFAULTS = ROOT / "examples/server.example.toml"
IMAGE = "sha256:" + "1" * 64


def ring():
    profile = config.load(RING)
    profile["runtime"]["reference_image"] = IMAGE
    profile["runtime"]["lpa_image"] = IMAGE
    return profile


def reference_env():
    dockerfile = (ROOT / "docker/Dockerfile.reference").read_text(encoding="utf-8")
    return re.findall(r"^ENV (GLM53_\w+=\S+)$", dockerfile, re.MULTILINE)


def link(profile, rank, peer):
    return next(
        item for item in profile["nodes"][rank]["links"] if item["peer"] == peer
    )


def value(args, flag):
    return args[args.index(flag) + 1]


def with_host_addresses(profile, ranks=(0, 1, 2)):
    for rank in ranks:
        profile["nodes"][rank]["host_address"] = f"10.40.0.{rank + 1}"
        profile["nodes"][rank]["host_interface"] = "glmhost"
    return profile


class RingExampleTests(unittest.TestCase):
    def test_the_example_is_the_distributed_defaults_on_three_nodes(self):
        profile, defaults = config.load(RING), config.load(DEFAULTS)
        self.assertEqual(len(profile["nodes"]), 3)
        self.assertEqual(
            {k: v for k, v in profile.items() if k != "nodes"},
            {k: v for k, v in defaults.items() if k != "nodes"},
        )
        self.assertEqual(profile["context"]["max_model_len"], 262144)
        self.assertEqual(profile["context"]["max_num_seqs"], 1)
        self.assertEqual(profile["mtp"]["num_speculative_tokens"], 3)

    def test_every_rank_launches_tp3_and_only_the_head_serves(self):
        profile = ring()
        for rank in (0, 1, 2):
            with self.subTest(rank=rank):
                args = config.serve_args(profile, rank, "/model")
                self.assertEqual(value(args, "--nnodes"), "3")
                self.assertEqual(value(args, "--tensor-parallel-size"), "3")
                self.assertEqual(value(args, "--node-rank"), str(rank))
                self.assertEqual("--headless" in args, rank != 0)
                # 16 vision heads do not split three ways: the tower runs per rank.
                self.assertEqual(value(args, "--mm-encoder-tp-mode"), "data")

    def test_without_host_addresses_a_rank_meets_the_head_on_their_link(self):
        profile = ring()
        expected = {
            # rank: (master address, advertised address, socket interface)
            0: ("10.53.1.1", "10.53.1.1", "fabric1"),
            1: ("10.53.1.1", "10.53.1.2", "fabric0"),
            2: ("10.53.2.1", "10.53.2.2", "fabric1"),
        }
        for rank, (master, own, interface) in expected.items():
            with self.subTest(rank=rank):
                args = config.serve_args(profile, rank, "/model")
                env = config.environment(profile, rank)
                self.assertEqual(value(args, "--master-addr"), master)
                self.assertEqual(env["VLLM_HOST_IP"], own)
                self.assertEqual(env["GLOO_SOCKET_IFNAME"], interface)
                self.assertEqual(env["NCCL_SOCKET_IFNAME"], "=" + interface)

    def test_host_addresses_are_the_master_and_the_advertised_addresses(self):
        profile = with_host_addresses(ring())
        config.validate(profile)
        for rank in (0, 1, 2):
            with self.subTest(rank=rank):
                args = config.serve_args(profile, rank, "/model")
                env = config.environment(profile, rank)
                self.assertEqual(value(args, "--master-addr"), "10.40.0.1")
                self.assertEqual(env["VLLM_HOST_IP"], f"10.40.0.{rank + 1}")
                self.assertEqual(env["GLOO_SOCKET_IFNAME"], "glmhost")
                self.assertEqual(env["NCCL_SOCKET_IFNAME"], "=glmhost")

    def test_each_rank_picks_its_own_address_and_the_heads(self):
        # Only rank 1 carries a host address: it advertises it, and still meets
        # the head at the head's end of their link.
        profile = with_host_addresses(ring(), ranks=(1,))
        config.validate(profile)
        self.assertEqual(
            value(config.serve_args(profile, 1, "/model"), "--master-addr"),
            "10.53.1.1",
        )
        self.assertEqual(config.environment(profile, 1)["VLLM_HOST_IP"], "10.40.0.2")
        self.assertEqual(config.environment(profile, 2)["VLLM_HOST_IP"], "10.53.2.2")
        # Only the head carries one: every rank meets it there.
        profile = with_host_addresses(ring(), ranks=(0,))
        config.validate(profile)
        for rank in (0, 1, 2):
            self.assertEqual(
                value(config.serve_args(profile, rank, "/model"), "--master-addr"),
                "10.40.0.1",
            )
        self.assertEqual(config.environment(profile, 1)["VLLM_HOST_IP"], "10.53.1.2")

    def test_nccl_uses_every_link_of_the_rank_and_the_subnet_aware_routing(self):
        profile = ring()
        for rank in (0, 1, 2):
            env = config.environment(profile, rank)
            hcas = ",".join(
                f"{item['hca']}:1" for item in profile["nodes"][rank]["links"]
            )
            with self.subTest(rank=rank):
                self.assertEqual(env["NCCL_IB_HCA"], "=" + hcas)
                self.assertEqual(env["NCCL_IB_GID_INDEX"], "3")
                self.assertEqual(env["NCCL_IB_SUBNET_AWARE_ROUTING"], "1")
                self.assertEqual(env["GLM53_TP_PAD_MULTIPLE"], "3")
        self.assertEqual(
            config.environment(profile, 0)["NCCL_IB_HCA"], "=roce1:1,roce0:1"
        )

    def test_two_node_launches_set_neither_knob(self):
        profiles = [
            config.load(DEFAULTS),
            config.load(ROOT / "examples/server.axl.example.toml"),
        ]
        pp2 = config.load(DEFAULTS)
        pp2["runtime"]["pipeline_parallel_size"] = 2
        pp2["mtp"]["enabled"] = False
        pp2["cache"].update(prefix_caching=False, fused_unpack=False)
        config.validate(pp2)
        for profile in [*profiles, pp2]:
            for rank in (0, 1):
                env = config.environment(profile, rank)
                args = config.serve_args(profile, rank, "/model")
                self.assertNotIn("GLM53_TP_PAD_MULTIPLE", env)
                self.assertNotIn("NCCL_IB_SUBNET_AWARE_ROUTING", env)
                self.assertNotIn("--mm-encoder-tp-mode", args)

    def test_the_encoder_mode_follows_vision(self):
        profile = ring()
        profile["runtime"]["vision"] = False
        self.assertNotIn(
            "--mm-encoder-tp-mode", config.serve_args(profile, 0, "/model")
        )

    def test_the_runtime_cache_is_per_node_count_and_tp2_keeps_its_name(self):
        self.assertEqual(server.runtime_cache_dir().name, "tp2-runtime-cache")
        self.assertEqual(server.runtime_cache_dir(3).name, "tp3-runtime-cache")
        command = server.command(ring(), RING, 2, "test", cache=Path("/cache"))
        self.assertIn(f"{server.runtime_cache_dir(3)}:{config.RUNTIME_CACHE}", command)

    def test_the_padded_launch_needs_the_padding_image_and_the_reference_attention(
        self,
    ):
        profile = ring()
        env = reference_env()
        checks = config.image_capability_checks(profile, {"Config": {"Env": env}})
        self.assertTrue(checks["tp_padding_support"])
        self.assertTrue(checks["reference_attention"])
        # 22 heads per rank: the SM120 decode kernel refuses them (plan Stage 0 P2),
        # the reference path takes any count, and preflight always requires it.
        self.assertNotEqual(HEADS % 3, 0)
        without = [m for m in env if m != "GLM53_TP_PAD_API=1"]
        checks = config.image_capability_checks(profile, {"Config": {"Env": without}})
        self.assertFalse(checks["tp_padding_support"])
        two = config.image_capability_checks(
            config.load(DEFAULTS), {"Config": {"Env": env}}
        )
        self.assertNotIn("tp_padding_support", two)


class RingValidationTests(unittest.TestCase):
    def refused(self, profile, message):
        with self.assertRaisesRegex(ValueError, message):
            config.validate(profile)

    def test_a_missing_link_is_refused(self):
        profile = ring()
        profile["nodes"][2]["links"].pop(1)
        self.refused(profile, "every other node once")
        profile = ring()
        for rank, peer in ((1, 2), (2, 1)):
            profile["nodes"][rank]["links"] = [
                item for item in profile["nodes"][rank]["links"] if item["peer"] != peer
            ]
        self.refused(profile, "every other node once")

    def test_a_link_to_itself_or_twice_to_one_peer_is_refused(self):
        for peer in (1, 3, "2"):
            profile = ring()
            profile["nodes"][1]["links"][1]["peer"] = peer
            with self.subTest(peer=peer):
                self.refused(profile, "every other node once")

    def test_mismatched_ends_of_a_link_are_refused(self):
        for rank, peer, key, address in (
            (1, 2, "peer_ip", "10.53.3.3"),
            (2, 1, "local_ip", "10.53.3.3"),
            # Consistent on both sides, but not one /30.
            (0, 2, "local_ip", "10.53.2.5"),
        ):
            profile = ring()
            link(profile, rank, peer)[key] = address
            if key == "local_ip":
                link(profile, peer, rank)["peer_ip"] = address
            with self.subTest(rank=rank, key=key):
                self.refused(profile, "one /30")

    def test_the_links_of_a_rank_share_one_gid_index(self):
        profile = ring()
        profile["nodes"][2]["links"][1]["gid_index"] = 4
        self.refused(profile, "common GID index")
        # Another rank may use another index.
        profile = ring()
        for item in profile["nodes"][2]["links"]:
            item["gid_index"] = 4
        config.validate(profile)
        self.assertEqual(config.environment(profile, 2)["NCCL_IB_GID_INDEX"], "4")

    def test_malformed_links_are_refused(self):
        for change in (
            {"interface": "wlP9s9"},
            {"hca": "roce 0"},
            {"peer_ip": "127.0.0.1"},
            {"gid_index": -1},
            {"port": 1},
        ):
            profile = ring()
            profile["nodes"][0]["links"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                config.validate(profile)
        profile = ring()
        del profile["nodes"][0]["links"][0]["gid_index"]
        self.refused(profile, "peer, hca, interface, local_ip, peer_ip, gid_index")

    def test_three_nodes_need_links_on_every_node(self):
        profile = config.load(DEFAULTS)
        third = copy.deepcopy(profile["nodes"][1])
        third["local_ip"] = "10.53.0.3"
        profile["nodes"].append(third)
        self.refused(profile, "one direct link per pair")
        profile = ring()
        profile["nodes"][2] = {
            "local_ip": "10.53.2.2",
            "interface": "fabric1",
            "hca": "roce1",
            "gid_index": 3,
        }
        self.refused(profile, "Every node lists its links, or none does")

    def test_a_node_with_links_carries_no_single_rail_keys(self):
        profile = ring()
        profile["nodes"][0]["local_ip"] = "10.53.1.1"
        self.refused(profile, r"Unknown/missing settings in server\.nodes\[0\]")
        profile = ring()
        profile["nodes"][0]["additional_rails"] = []
        self.refused(profile, "additional_rails")

    def test_host_addresses_come_with_an_interface_and_are_distinct(self):
        profile = ring()
        profile["nodes"][1]["host_address"] = "10.40.0.2"
        self.refused(profile, "host_address and host_interface")
        profile = with_host_addresses(ring())
        profile["nodes"][2]["host_address"] = "10.40.0.1"
        self.refused(profile, "host_address")
        profile = with_host_addresses(ring())
        profile["nodes"][2]["host_address"] = "10.53.2.2"
        self.refused(profile, "host_address")
        profile = with_host_addresses(ring())
        profile["nodes"][2]["host_interface"] = "wlP9s9"
        self.refused(profile, "Wi-Fi")
        two = config.load(DEFAULTS)
        two["nodes"][1].update(host_address="10.40.0.2", host_interface="glmhost")
        self.refused(two, "host_address belongs to nodes with links")

    def test_a_management_wifi_host_address_is_an_explicit_test_setting(self):
        # Ring data goes over the links' HCAs and host_interface carries only the
        # sockets; the first TP=3 boot ran them on the management Wi-Fi (plan
        # Stage 4). Allowed only when the node says so; the links stay refused.
        def wifi(opt_in):
            profile = ring()
            for rank, address in enumerate(
                ("192.168.68.56", "192.168.68.59", "192.168.68.57")
            ):
                profile["nodes"][rank].update(
                    host_address=address, host_interface="wlP9s9"
                )
                if opt_in:
                    profile["nodes"][rank]["host_interface_wifi_test"] = True
            return profile

        self.refused(wifi(False), "Wi-Fi")
        profile = wifi(True)
        config.validate(profile)
        env = config.environment(profile, 2)
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "=wlP9s9")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "wlP9s9")
        self.assertEqual(env["VLLM_HOST_IP"], "192.168.68.57")
        self.assertEqual(env["NCCL_IB_HCA"], "=roce1:1,roce0:1")
        # One node without the key refuses the profile.
        profile = wifi(True)
        del profile["nodes"][1]["host_interface_wifi_test"]
        self.refused(profile, "Wi-Fi")
        # The key is true or absent, and only beside a host_interface.
        profile = wifi(True)
        profile["nodes"][0]["host_interface_wifi_test"] = "yes"
        self.refused(profile, "host_interface_wifi_test")
        profile = ring()
        profile["nodes"][0]["host_interface_wifi_test"] = True
        self.refused(profile, "host_interface_wifi_test")
        two = config.load(DEFAULTS)
        two["nodes"][1]["host_interface_wifi_test"] = True
        self.refused(two, "nodes with links")
        # The site check refuses a Wi-Fi socket interface without the key.
        site = config.site(wifi(True), 2)
        del site["host_interface_wifi_test"]
        with self.assertRaisesRegex(ValueError, "Wi-Fi"):
            fabric.validate_site(site)

    def test_fewer_than_two_nodes_are_refused(self):
        profile = config.load(DEFAULTS)
        profile["nodes"].pop()
        self.refused(profile, "two or more nodes")

    def test_two_nodes_may_also_be_written_as_links(self):
        profile = ring()
        profile["nodes"] = [
            {"links": [link(profile, 0, 1)]},
            {"links": [link(profile, 1, 0)]},
        ]
        config.validate(profile)
        args = config.serve_args(profile, 1, "/model")
        self.assertEqual(value(args, "--tensor-parallel-size"), "2")
        env = config.environment(profile, 1)
        self.assertNotIn("NCCL_IB_SUBNET_AWARE_ROUTING", env)
        self.assertNotIn("GLM53_TP_PAD_MULTIPLE", env)

    def test_the_rank_is_bounded_by_the_node_count(self):
        profile = ring()
        with self.assertRaisesRegex(ValueError, "rank must be from 0 to 2"):
            config.site(profile, 3)
        with self.assertRaisesRegex(ValueError, "rank must be from 0 to 1"):
            config.site(config.load(DEFAULTS), 2)


class RingExclusionTests(unittest.TestCase):
    def test_two_node_experiments_are_refused_on_three_nodes(self):
        cases = {
            "PP2": lambda p: p["runtime"].update(pipeline_parallel_size=2),
            "EP": lambda p: p["runtime"].update(expert_parallel=True),
            "EP observer": lambda p: p["validation"].update(expert_worker=True),
            "LPA": lambda p: p["lpa"].update(enabled=True),
        }
        for name, change in cases.items():
            profile = ring()
            # Leave each experiment's own rules satisfied, so the node count refuses it.
            profile["mtp"]["enabled"] = False
            profile["cache"].update(prefix_caching=False, fused_unpack=False)
            profile["runtime"]["fa2_attention"] = False
            change(profile)
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "two nodes"),
            ):
                config.validate(profile)

    def test_a_derived_checkpoint_launches_on_three_nodes(self):
        # Stage 6 (2026-10-01): the overlays split heads by TP with assert num_heads % tp_size,
        # which holds at the padded 66 heads; the derived config is padded by the same knob.
        profile = ring()
        profile["runtime"]["derived_checkpoint"] = {
            "path": "/derived",
            "requant_target": "l",
            "overlays": [
                {
                    "target": "kda.py",
                    "source": "/overlays/kda.py",
                    "sha256": "0" * 64,
                    "base_sha256": "0" * 64,
                    "marker": "m",
                }
            ],
        }
        config.validate(profile)
        env = config.environment(profile, 2)
        self.assertEqual(env["GLM53_TP_PAD_MULTIPLE"], "3")

    def test_the_tp2_kv_limit_does_not_apply_to_three_nodes(self):
        profile = ring()
        profile["cache"]["kv_cache_memory_bytes"] = 6 * 2**30
        config.validate(profile)
        two = config.load(DEFAULTS)
        two["cache"]["kv_cache_memory_bytes"] = 6 * 2**30
        with self.assertRaisesRegex(ValueError, "derived_checkpoint"):
            config.validate(two)


class RingPreflightTests(unittest.TestCase):
    def preflight(self, profile, rank, checks):
        cache = Path.home() / ".cache/huggingface"
        model = server.model_path(profile, cache)
        env = reference_env()

        def docker(*args):
            if args[:3] == ("docker", "image", "inspect"):
                return json.dumps([{"Id": IMAGE, "Config": {"Env": env}}])
            raise AssertionError(args)

        view = {
            "source_revision": server.load_lock()["revision"],
            "weight_bytes_modified": False,
        }
        metadata = {
            "text_config": {"num_hidden_layers": server.MODEL_LAYERS},
            server.MTP_VIEW_KEY: view,
        }
        with (
            patch.object(server, "read_json", return_value=metadata),
            patch.object(
                server.host,
                "snapshot_from_state",
                return_value=server.model_path(
                    {**profile, "mtp": {**profile["mtp"], "enabled": False}}, cache
                ),
            ),
            patch.object(
                server.host, "fabric_checks", return_value=checks
            ) as fabric_checks,
            patch.object(server.host, "run", side_effect=docker),
            patch.object(server.host, "running_containers", return_value=[]),
        ):
            result = server.preflight(profile, RING, rank, check_memory=False)
        return result, fabric_checks.call_args.args[0], model

    def test_the_cpu_part_of_preflight_passes_on_every_rank(self):
        profile = ring()
        for rank in (0, 1, 2):
            with self.subTest(rank=rank):
                result, site, _ = self.preflight(
                    profile, rank, {"rail_0_link_up": True}
                )
                self.assertTrue(result["passed"], result["checks"])
                self.assertTrue(result["checks"]["tp_padding_support"])
                self.assertTrue(result["checks"]["reference_attention"])
                self.assertEqual(site["rank"], rank)
                self.assertEqual(site["nnodes"], 3)
        result, _, _ = self.preflight(profile, 2, {"rail_1_link_up": False})
        self.assertFalse(result["passed"])

    def test_fabric_checks_cover_every_link_and_the_host_address(self):
        profile = with_host_addresses(ring())
        site = config.site(profile, 2)
        seen = []

        def run(*args):
            seen.append(args)
            return "[]"

        checks = fabric.checks(
            site, run, sys_root=Path("/nonexistent"), dev_root=Path("/nonexistent")
        )
        for index in (0, 1):
            for name in ("link_up", "port_active", "roce_v2_gid", "address_assigned"):
                self.assertIn(fabric.rail_check(index, name), checks)
        self.assertIn("host_address_assigned", checks)
        self.assertFalse(checks["host_address_assigned"])
        self.assertNotIn(
            "host_address_assigned",
            fabric.checks(
                config.site(ring(), 2),
                run,
                sys_root=Path("/nonexistent"),
                dev_root=Path("/nonexistent"),
            ),
        )


class RingPlanTests(unittest.TestCase):
    def test_plan_shows_each_rank_and_how_to_probe_its_links(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            server.main(["plan", "--config", str(RING), "--rank", "2"])
        plan = json.loads(printed.getvalue())
        self.assertIn("--headless", plan["command"])
        probes = {probe["peer"]: probe for probe in plan["link_probes"]}
        self.assertEqual(sorted(probes), [0, 1])
        # The lower rank of the pair is the probe's head, on that link's address.
        self.assertEqual(probes[0]["head"], "10.53.2.1")
        self.assertEqual(probes[0]["probe_rank"], 1)
        self.assertEqual(probes[1]["head"], "10.53.3.1")
        env = probes[1]["environment"]
        self.assertEqual(env["NCCL_IB_HCA"], "=roce0:1")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "=fabric0")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "fabric0")
        self.assertEqual(env["NCCL_IB_GID_INDEX"], "3")

    def test_a_two_node_plan_is_unchanged(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            server.main(["plan", "--config", str(DEFAULTS), "--rank", "1"])
        self.assertNotIn("link_probes", json.loads(printed.getvalue()))


class RingTomlTests(unittest.TestCase):
    def test_the_example_parses_as_plain_toml(self):
        with RING.open("rb") as stream:
            data = tomllib.load(stream)
        self.assertEqual([sorted(node) for node in data["nodes"]], [["links"]] * 3)
