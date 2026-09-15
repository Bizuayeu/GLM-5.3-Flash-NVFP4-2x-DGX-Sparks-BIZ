import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import server
from glm53_setup import server_config as config


class AllocatorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.profile = config.load(self.root / "examples/server.example.toml")

    def test_origin_resolves_unset_value_and_empty_identically_for_both_ranks(self):
        for env, expected in [
            ({}, None),
            ({"PYTORCH_CUDA_ALLOC_CONF": ""}, ""),
            (
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128"
                },
                "expandable_segments:True,max_split_size_mb:128",
            ),
        ]:
            frozen = config.resolve_launch(self.profile, env)
            for rank in (0, 1):
                with patch.dict(
                    "os.environ", {"PYTORCH_CUDA_ALLOC_CONF": f"wrong-host-{rank}"}
                ):
                    command = server.command(
                        frozen, self.root / "state/server.toml", rank, "test"
                    )
                values = [
                    x for x in command if x.startswith("PYTORCH_CUDA_ALLOC_CONF=")
                ]
                self.assertEqual(
                    values,
                    [] if expected is None else ["PYTORCH_CUDA_ALLOC_CONF=" + expected],
                )
        self.assertNotIn("cuda_allocator_conf", self.profile["runtime"])

    def test_env_presence_overrides_toml_but_absence_preserves_it(self):
        self.profile["runtime"]["cuda_allocator_conf"] = "backend:cudaMallocAsync"
        self.assertEqual(
            config.resolve_launch(self.profile, {})["runtime"]["cuda_allocator_conf"],
            "backend:cudaMallocAsync",
        )
        self.assertEqual(
            config.resolve_launch(self.profile, {"PYTORCH_CUDA_ALLOC_CONF": ""})[
                "runtime"
            ]["cuda_allocator_conf"],
            "",
        )
        for value in [True, None, "bad\nvalue"]:
            self.profile["runtime"]["cuda_allocator_conf"] = value
            with self.assertRaises(ValueError):
                config.validate(self.profile)

    def test_frozen_manifest_survives_other_host_env_and_detects_tampering(self):
        manifest = server.freeze(self.profile, {"PYTORCH_CUDA_ALLOC_CONF": ""})
        with patch.dict("os.environ", {"PYTORCH_CUDA_ALLOC_CONF": "different"}):
            loaded = server.thaw(manifest)
        self.assertEqual(loaded["runtime"]["cuda_allocator_conf"], "")
        manifest["profile"]["runtime"]["cuda_allocator_conf"] = (
            "backend:cudaMallocAsync"
        )
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            server.thaw(manifest)
