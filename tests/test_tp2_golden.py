"""TP=2 launches stay byte for byte what 1.21.0 produced.

tests/tp2_launch_golden.json was written by ``snapshot()`` at d2270d8 (1.21.0),
before the launcher learned N nodes. Every two-node profile below must still give
the same site, environment, vLLM arguments, docker command, image checks and
fingerprint on both ranks; a difference here means a TP=2 launch moved.
"""

import json
import re
import unittest
from pathlib import Path

from glm53_setup import server
from glm53_setup import server_config as config

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/tp2_launch_golden.json"
IMAGE = "sha256:" + "1" * 64
CACHE = Path("/cache")


def variants():
    """The shipped two-node examples, and the PP2 and EP shapes of the defaults."""
    defaults = config.load(ROOT / "examples/server.example.toml")
    pp2 = config.load(ROOT / "examples/server.example.toml")
    pp2["runtime"]["pipeline_parallel_size"] = 2
    ep = config.load(ROOT / "examples/server.example.toml")
    ep["runtime"]["expert_parallel"] = True
    for profile in (pp2, ep):
        profile["mtp"]["enabled"] = False
        profile["cache"].update(prefix_caching=False, fused_unpack=False)
    result = {
        "defaults": defaults,
        "axl": config.load(ROOT / "examples/server.axl.example.toml"),
        "pp2": pp2,
        "ep": ep,
    }
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
                for rank in (0, 1)
            ],
            "capabilities": config.image_capability_checks(profile, image),
        }
    return normalized(result)


class TwoNodeGoldenTests(unittest.TestCase):
    def test_two_node_launches_match_the_1_21_0_golden(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        current = snapshot()
        self.assertEqual(sorted(current), sorted(golden))
        for name in golden:
            with self.subTest(profile=name):
                self.assertEqual(current[name], golden[name])


if __name__ == "__main__":
    # Regenerate only from a checkout whose TP=2 output is the reference.
    with GOLDEN.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(snapshot(), indent=1) + "\n")
