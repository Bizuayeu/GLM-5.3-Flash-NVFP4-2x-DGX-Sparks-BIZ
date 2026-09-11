import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm53_setup import service


class ServiceContractTests(unittest.TestCase):
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
        args = service.serve_args(self.site, "/hf/model")
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
        args = service.serve_args(self.site, "/hf/model")
        self.assertIn("--headless", args)
        self.assertEqual(service.fabric_env(self.site)["VLLM_HOST_IP"], "10.53.0.2")
        self.assertEqual(service.fabric_env(self.site)["NCCL_NET"], "IB")
        self.assertEqual(service.fabric_env(self.site)["NCCL_IB_HCA"], "=roce0")

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
                service.validate_site({**self.site, **values})

    def test_hf_mount_preserves_blob_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            snapshot = (
                cache
                / "hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots"
                / self.lock["revision"]
            )
            snapshot.mkdir(parents=True)
            args = service.docker_args(self.site, self.lock, snapshot, cache, root)
            self.assertIn(f"{cache.resolve()}:/hf:ro", args)
            self.assertNotIn("--privileged", args)
            self.assertNotIn("--rm", args)
            self.assertIn(self.lock["image"], args)
            with self.assertRaises(ValueError):
                service.docker_args(self.site, self.lock, root / "outside", cache, root)

    def test_snapshot_requires_matching_complete_state(self):
        state = {
            "status": "complete",
            "model": self.lock["model"],
            "revision": self.lock["revision"],
            "snapshot": "/tmp/model",
        }
        self.assertEqual(
            service.snapshot_from_state(state, self.lock), Path("/tmp/model")
        )
        for change in [{"status": "downloading"}, {"revision": "c" * 40}]:
            with self.assertRaises(ValueError):
                service.snapshot_from_state({**state, **change}, self.lock)

    def test_stale_kernel_receipt_cannot_authorize_launch(self):
        receipt = {
            "passed": True,
            "image": self.lock["image"],
            "model_revision": self.lock["revision"],
            "kind": "tp2-kernel-validation",
        }
        self.assertTrue(service.valid_kernel_receipt(receipt, self.lock))
        for change in [{"passed": False}, {"image": "old"}, {"kind": "gpu-smoke"}]:
            self.assertFalse(
                service.valid_kernel_receipt({**receipt, **change}, self.lock)
            )


if __name__ == "__main__":
    unittest.main()
