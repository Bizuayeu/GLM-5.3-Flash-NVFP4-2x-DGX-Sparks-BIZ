import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import fabric, host, server_config


class HostContractTests(unittest.TestCase):
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
        }
        self.lock = {
            "image": "vllm/vllm-openai@sha256:" + "a" * 64,
            "model": "nvidia/GLM-5.3-Flash-NVFP4",
            "revision": "b" * 40,
        }

    def test_p0_preserves_target_semantics_and_local_api(self):
        args = server_config.serve_template(self.site, "/hf/model")
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        for flag in [
            "--enforce-eager",
            "--language-model-only",
            "--no-enable-prefix-caching",
        ]:
            self.assertIn(flag, args)
        for flag in [
            "--speculative-config",
            "--enable-expert-parallel",
            "--trust-remote-code",
        ]:
            self.assertNotIn(flag, args)

    def test_worker_uses_own_address_and_headless(self):
        self.site.update(rank=1, local_ip="10.53.0.2")
        args = server_config.serve_template(self.site, "/hf/model")
        self.assertIn("--headless", args)
        self.assertEqual(fabric.fabric_env(self.site)["VLLM_HOST_IP"], "10.53.0.2")
        self.assertEqual(fabric.fabric_env(self.site)["NCCL_NET"], "IB")
        self.assertEqual(fabric.fabric_env(self.site)["NCCL_IB_HCA"], "=roce0:1")

    def test_bad_rank_interface_or_address_refused(self):
        for values in [
            {"rank": 2},
            {"interface": ""},
            {"interface": "wlP9s9"},
            {"head_ip": "127.0.0.1"},
            {"local_ip": "10.53.0.2"},
            {"gid_index": -1},
            {"api_port": 70000},
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                fabric.validate_site({**self.site, **values})

    def test_free_blocks_count_only_orders_of_two_mib_and_above(self):
        # Recorded on the reference head (4 KiB pages, orders 0-13) while serving.
        text = (
            "Node 0, zone      DMA      1      2      2      2      3      3"
            "      2      3     16      9      4      2      4     11 \n"
            "Node 0, zone   Normal   2613   1163   1355    778    666    769"
            "    956    217     34      1      4      2      0      0 \n"
        )
        # Order >= 9: DMA 466 MiB plus Normal 34 MiB.
        self.assertEqual(host.free_blocks_gib(text, 4096, 2 * 1024**2), 500 / 1024)
        # 64 KiB pages reach 2 MiB at order 5.
        line = "Node 0, zone   Normal   7 7 7 7 7 1 2\n"
        self.assertEqual(
            host.free_blocks_gib(line, 64 * 1024, 2 * 1024**2), (2 + 2 * 4) / 1024
        )
        for bad in ("", "Node 0, zone Normal\n", "Node 0, zone Normal 1 x 3\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                host.free_blocks_gib(bad, 4096, 2 * 1024**2)

    def test_meminfo_reads_named_field_in_gib(self):
        text = "MemTotal:  127600792 kB\nMemFree:    1129420 kB\n"
        self.assertEqual(host.meminfo_gib(text, "MemFree"), 1129420 / 1024**2)
        with self.assertRaises(ValueError):
            host.meminfo_gib(text, "MemAvailable")

    def test_foreign_gpu_containers_skip_owned_and_gpu_free_containers(self):
        def container(name, requests, labels=None):
            return {
                "Name": "/" + name,
                "Config": {"Labels": labels},
                "HostConfig": {"DeviceRequests": requests},
            }

        gpus = [{"Driver": "", "Count": -1, "Capabilities": [["gpu"]]}]
        cdi = [{"Driver": "cdi", "DeviceIDs": ["nvidia.com/gpu=all"]}]
        inspections = [
            # A pair of any fingerprint is this launcher's own, e.g. the old
            # pair still running while cluster switch checks the new profile.
            container("glm53-startup-r0-old", gpus, {"owner": "old-fingerprint"}),
            container("searxng", None),
            container("other-cdi", cdi, {}),
            container("other-gpus", gpus),
            # An empty value is not ownership: inspect_owned rejects it too.
            container("empty-owner", gpus, {"owner": ""}),
        ]
        self.assertEqual(
            host.foreign_gpu_containers(inspections, "owner"),
            ["empty-owner", "other-cdi", "other-gpus"],
        )

    def test_running_container_inspection_skips_inspect_when_none_run(self):
        with patch.object(host, "run", return_value="\n") as run:
            self.assertEqual(host.running_containers(), [])
            run.assert_called_once_with("docker", "ps", "-q")

    def test_running_containers_drop_ones_that_exit_or_vanish_meanwhile(self):
        def inspected(container, running=True):
            return json.dumps([{"Id": container, "State": {"Running": running}}])

        gone = subprocess.CalledProcessError(1, ["docker", "inspect", "b2"])
        readings = [
            "a1\nb2\nc3\n",
            inspected("a1"),
            gone,
            "a1\nc3\n",
            inspected("c3", running=False),
        ]
        with patch.object(host, "run", side_effect=readings) as run:
            self.assertEqual(
                host.running_containers(), [{"Id": "a1", "State": {"Running": True}}]
            )
            run.assert_any_call("docker", "inspect", "b2")

    def test_running_containers_fail_closed_when_a_listed_one_cannot_be_read(self):
        broken = subprocess.CalledProcessError(1, ["docker", "inspect", "a1"])
        with patch.object(host, "run", side_effect=["a1\n", broken, "a1\n"]):
            with self.assertRaises(subprocess.CalledProcessError):
                host.running_containers()

    def test_container_memory_sample_reads_cgroup_and_process_status(self):
        # What told device-side growth from process growth on 2026-09-18: the
        # cgroup and the workers' RSS stayed flat while MemAvailable fell.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope = root / "cgroup/system.slice/docker-abc.scope"
            scope.mkdir(parents=True)
            (scope / "memory.current").write_text(str(3 * 1024**3))
            (scope / "cgroup.procs").write_text("11\n12\n13\n")
            for pid, rss, anon in ((11, 1048576, 524288), (12, 2097152, 1048576)):
                status = root / f"proc/{pid}"
                status.mkdir(parents=True)
                (status / "status").write_text(
                    f"Name:\tx\nVmRSS:\t{rss} kB\nRssAnon:\t{anon} kB\n"
                )
            sample = host.container_memory_sample(
                "abc", cgroup_root=root / "cgroup", proc_root=root / "proc"
            )
        self.assertEqual(sample["container_cgroup_gib"], 3.0)
        self.assertEqual(sample["container_rss_gib"], 3.0)  # pid 13 vanished: skipped
        self.assertEqual(sample["container_anon_gib"], 1.5)
        missing = host.container_memory_sample(
            "nope", cgroup_root=Path("/nonexistent"), proc_root=Path("/nonexistent")
        )
        self.assertEqual(missing, {"container_memory_error": "FileNotFoundError"})

    def test_memory_sample_records_unreadable_observation_instead_of_raising(self):
        meminfo = "MemFree: 1048576 kB\n"
        buddyinfo = "Node 0, zone Normal 0 0 0 0 0 0 0 0 0 3\n"
        with (
            patch.object(host.mmap, "PAGESIZE", 4096),
            patch.object(host.Path, "read_text", side_effect=[meminfo, buddyinfo]),
        ):
            sample = host.memory_sample()
        self.assertEqual(sample["mem_free_gib"], 1.0)
        self.assertEqual(sample["free_2mib_gib"], 3 * 2 * 1024**2 / 1024**3)
        for readings, error in [
            ([meminfo, ""], "ValueError"),
            (["MemFree:\n", buddyinfo], "IndexError"),
            (OSError("gone"), "OSError"),
        ]:
            with patch.object(host.Path, "read_text", side_effect=readings):
                self.assertEqual(host.memory_sample(), {"memory_sample_error": error})

    def test_snapshot_requires_matching_complete_state(self):
        state = {
            "status": "complete",
            "model": self.lock["model"],
            "revision": self.lock["revision"],
            "snapshot": "/tmp/model",
        }
        self.assertEqual(host.snapshot_from_state(state, self.lock), Path("/tmp/model"))
        for change in [{"status": "downloading"}, {"revision": "c" * 40}]:
            with self.assertRaises(ValueError):
                host.snapshot_from_state({**state, **change}, self.lock)


if __name__ == "__main__":
    unittest.main()
