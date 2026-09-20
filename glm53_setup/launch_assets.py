"""Read-only launch identities, suitable for comparison before stopping a model."""

import hashlib
import json
from pathlib import Path

from . import host, server
from . import server_config as settings
from .config import ROOT, load_lock


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect(profile, config_path, rank, *, recovery=False):
    checks = server.preflight(
        profile, config_path, rank, check_memory=False, recovery=recovery
    )
    if not checks["passed"]:
        detail = {key: checks[key] for key in ("checks", "foreign_gpu_containers")}
        raise ValueError("Static launch checks failed: " + json.dumps(detail))
    model = server.model_path(profile, Path.home() / ".cache/huggingface")
    if not all(
        (model / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        raise ValueError("Tokenizer assets required by the pinned model are missing")
    index_path = model / "model.safetensors.index.json"
    index = server.read_json(index_path)
    names = sorted(set(index["weight_map"].values()))
    if not names:
        raise ValueError("Empty model weight index")
    weights, identities = {}, {}
    for name in names:
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            or "\\" in name
        ):
            raise ValueError("Invalid model shard filename")
        path = model / name
        stat = path.stat()
        if not path.is_file() or stat.st_size <= 0:
            raise ValueError("Missing or empty model shard")
        weights[name] = stat.st_size
        # Local evidence catches replacement between preflight and stop; mtimes
        # and inode numbers intentionally do not participate in cross-rank equality.
        identities[name] = [
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ino,
        ]
    source = {
        str(p.relative_to(ROOT)): sha(p)
        for directory in ("glm53_setup", "config")
        for p in sorted((ROOT / directory).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".json")
    }
    source["examples/server.example.toml"] = sha(ROOT / "examples/server.example.toml")
    image = json.loads(
        host.run("docker", "image", "inspect", settings.selected_image(profile))
    )[0]
    image_allocator = next(
        (
            s.split("=", 1)[1]
            for s in image["Config"].get("Env", [])
            if s.startswith("PYTORCH_CUDA_ALLOC_CONF=")
        ),
        None,
    )
    common = {
        "profile": settings.fingerprint(profile),
        "source": hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest(),
        "image": image["Id"],
        "revision": load_lock()["revision"],
        "model_config": sha(model / "config.json"),
        "weight_index": sha(index_path),
        "tokenizer_and_templates": {
            p.name: sha(p)
            for p in sorted(model.iterdir())
            if p.is_file()
            and (
                p.name.startswith("tokenizer")
                or p.suffix == ".jinja"
                or p.name
                in (
                    "generation_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                )
            )
        },
        "weights": weights,
        "allocator": profile["runtime"].get("cuda_allocator_conf", image_allocator),
        "projector": profile["lpa"]["projector_sha256"]
        if profile["lpa"]["enabled"]
        else None,
    }
    return {
        "common": common,
        "local": {
            "rank": rank,
            "weights": identities,
            "site": settings.site(profile, rank),
            # Beside the identities, never in common: what one rank was allowed
            # must not decide whether the two ranks agree.
            "warnings": checks.get("warnings", []),
        },
    }
