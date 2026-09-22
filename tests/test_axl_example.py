import copy
import unittest
from difflib import SequenceMatcher
from pathlib import Path

from glm53_setup import server
from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ROOT / "examples/server.example.toml"
AXL = ROOT / "examples/server.axl.example.toml"
SIX_GIB = 6 * 2**30
IMAGE = "sha256:" + "1" * 64
VLLM_MODELS = "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia"


def launch(path, rank):
    """The docker command and the environment of an example with the image supplied."""
    profile = config.load(path)
    profile["runtime"]["reference_image"] = IMAGE
    profile["runtime"]["lpa_image"] = IMAGE
    command = server.command(profile, path, rank, "test", cache=Path("/cache"))
    return command, config.environment(profile, rank)


class AxlExampleTests(unittest.TestCase):
    def test_the_axl_example_serves_two_sequences_from_twice_the_kv(self):
        profile = config.load(AXL)
        config.validate(profile)
        derived = config.derived_checkpoint(profile)
        self.assertEqual(derived["requant_target"], "l")
        self.assertEqual(
            [o["target"] for o in derived["overlays"]], ["kda.py", "model.py"]
        )
        self.assertTrue(profile["runtime"]["prefix_page_dedup"])
        self.assertEqual(profile["context"]["max_num_seqs"], 2)
        self.assertEqual(profile["cache"]["kv_cache_memory_bytes"], SIX_GIB)
        for rank in (0, 1):
            args = config.serve_args(profile, rank, "/hf/mtp-view")
            self.assertEqual(args[args.index("--max-num-seqs") + 1], "2")
            self.assertEqual(
                args[args.index("--kv-cache-memory-bytes") + 1], str(SIX_GIB)
            )
            self.assertEqual(
                config.environment(profile, rank)["GLM53_PREFIX_PAGE_DEDUP"], "1"
            )
        # The two examples differ only where the option differs.
        defaults = config.load(DEFAULTS)
        self.assertEqual(profile["api"], defaults["api"])
        self.assertEqual(profile["nodes"], defaults["nodes"])
        self.assertEqual(profile["mtp"], defaults["mtp"])

    def test_the_two_examples_launch_differently_in_five_points(self):
        # docs/server-configuration.md, "The published option against the defaults":
        # the option adds the derived checkpoint (one mount), its two overlays (two
        # mounts), the dedup switch (one environment variable), the second sequence
        # and twice the KV. The model argument and the fingerprint label follow from
        # the derived checkpoint; nothing else in the docker command or the
        # environment moves, on either rank.
        for rank in (0, 1):
            with self.subTest(rank=rank):
                defaults, defaults_env = launch(DEFAULTS, rank)
                axl, axl_env = launch(AXL, rank)
                edits = []
                for tag, i1, i2, j1, j2 in SequenceMatcher(
                    None, defaults, axl, autojunk=False
                ).get_opcodes():
                    if tag != "equal":
                        # Sorted: the matcher may pair a flag with the value of
                        # the neighbouring one, the tokens are the same.
                        edits.append((sorted(defaults[i1:i2]), sorted(axl[j1:j2])))
                label = edits[0]
                self.assertTrue(label[0][0].startswith("glm53.experiment.startup="))
                self.assertTrue(label[1][0].startswith("glm53.experiment.startup="))
                self.assertEqual(
                    edits[1:],
                    [
                        (
                            [],
                            sorted(
                                [
                                    "-v",
                                    "-v",
                                    "-v",
                                    "/srv/glm53/weights/GLM-5.3-Flash-NVFP4-l-split"
                                    ":/derived:ro",
                                    "/srv/glm53/source/overlays/kda-quant-split.py"
                                    f":{VLLM_MODELS}/kda.py:ro",
                                    "/srv/glm53/source/overlays/mla-quant-split.py"
                                    f":{VLLM_MODELS}/model.py:ro",
                                ]
                            ),
                        ),
                        ([], ["-e", "GLM53_PREFIX_PAGE_DEDUP=1"]),
                        (
                            [
                                "/hf/local-views/glm53-mtp-compatible/"
                                "423acf37583782c51c142d145aef733d72943d93"
                            ],
                            ["/derived"],
                        ),
                        (["1"], ["2"]),
                        ([str(3 * 2**30)], [str(SIX_GIB)]),
                    ],
                )
                self.assertEqual(axl[axl.index("--max-num-seqs") + 1], "2")
                self.assertEqual(
                    axl[axl.index("--kv-cache-memory-bytes") + 1], str(SIX_GIB)
                )
                self.assertEqual(
                    {k: v for k, v in axl_env.items() if defaults_env.get(k) != v},
                    {"GLM53_PREFIX_PAGE_DEDUP": "1"},
                )
                self.assertEqual([k for k in defaults_env if k not in axl_env], [])

    def test_a_kv_budget_above_three_gib_needs_the_derived_checkpoint(self):
        # Measured on the reference pair (2026-09-22): the pinned weights load 95.76 GiB
        # per rank and leave the head 5.5 GiB at 3 GiB of KV; the repacked ones load
        # 91.34 GiB and leave 10.5 GiB. Six GiB of KV crosses the 3 GiB reserve on the
        # pinned weights and stays above it on the repacked ones.
        defaults = config.load(DEFAULTS)
        defaults["cache"]["kv_cache_memory_bytes"] = SIX_GIB
        with self.assertRaisesRegex(ValueError, "derived_checkpoint"):
            config.validate(defaults)
        defaults["cache"]["kv_cache_memory_bytes"] = 3 * 2**30
        config.validate(defaults)
        axl = config.load(AXL)
        axl["runtime"]["derived_checkpoint"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "derived_checkpoint"):
            config.validate(axl)
        pinned_size = copy.deepcopy(axl)
        pinned_size["cache"]["kv_cache_memory_bytes"] = 3 * 2**30
        config.validate(pinned_size)  # a smaller KV needs no derived checkpoint


if __name__ == "__main__":
    unittest.main()
