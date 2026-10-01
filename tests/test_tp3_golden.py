"""TP=3 launches stay byte for byte what 1.24.0 produced.

tests/tp3_launch_golden.json was written by ``snapshot()`` from the tree of 856e4bf (1.24.0),
before the optional ``runtime.shm_spin_seconds``. The three-node example, and the AXL settings on
its nodes (the serving shape), must still give the same site, environment, vLLM arguments, docker
command, image checks and fingerprint on every rank; a profile without the new key launches as before.
1.26.0 (2026-10-02): the TP=3 and AXL templates set runtime.shm_spin_seconds = 0.002, which adds
GLM53_SHM_SPIN_SECONDS and its two read-only mounts on every rank and moves both fingerprints. The
``*_no_spin`` entries drop the key again and equal the 1.24.0 golden's ``ring`` and ``ring_axl``
byte for byte.
"""

import copy
import json
import re
import unittest
from pathlib import Path

from glm53_setup import server
from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/tp3_launch_golden.json"
IMAGE = "sha256:" + "1" * 64
CACHE = Path("/cache")


def variants():
    ring = config.load(ROOT / "examples/server.tp3.example.toml")
    axl = config.load(ROOT / "examples/server.axl.example.toml")
    axl["nodes"] = ring["nodes"]
    result = {"ring": ring, "ring_axl": axl}
    for name in ("ring", "ring_axl"):
        without = copy.deepcopy(result[name])
        del without["runtime"]["shm_spin_seconds"]
        result[name + "_no_spin"] = without
    for profile in result.values():
        profile["runtime"]["reference_image"] = IMAGE
        profile["runtime"]["lpa_image"] = IMAGE
        config.validate(profile)
    return result


def reference_env():
    dockerfile = (ROOT / "docker/Dockerfile.reference").read_text(encoding="utf-8")
    return re.findall(r"^ENV (GLM53_\w+=\S+)$", dockerfile, re.MULTILINE)


def normalized(value):
    """Host paths as placeholders, so the golden holds on Windows and Linux."""
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, str):
        value = value.replace(str(server.ROOT), "<ROOT>")
        return value.replace(str(CACHE.resolve()), "<CACHE>").replace("\\", "/")
    return value


def snapshot():
    image = {"Config": {"Env": reference_env()}}
    result = {}
    for name, profile in variants().items():
        result[name] = {
            "fingerprint": config.fingerprint(profile),
            "ranks": [
                {
                    "site": config.site(profile, rank),
                    "environment": config.environment(profile, rank),
                    "serve_args": config.serve_args(profile, rank, "/model"),
                    "command": server.command(
                        profile, ROOT / "state/server.toml", rank, "test", cache=CACHE
                    ),
                }
                for rank in range(len(profile["nodes"]))
            ],
            "capabilities": config.image_capability_checks(profile, image),
        }
    return normalized(result)


class ThreeNodeGoldenTests(unittest.TestCase):
    def test_three_node_launches_match_the_1_24_0_golden(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        current = snapshot()
        self.assertEqual(sorted(current), sorted(golden))
        for name in golden:
            with self.subTest(profile=name):
                self.assertEqual(current[name], golden[name])


if __name__ == "__main__":
    # Regenerate only from a checkout whose TP=3 output is the reference.
    with GOLDEN.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(snapshot(), indent=1) + "\n")
