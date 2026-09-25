"""Build and byte-verify a small checkpoint without altering source weights."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from ..config import MODEL, REVISION
from ..io import write_json


def keep_tensor(name, layers=4):
    prefix = "model.language_model.layers."
    if name.startswith(prefix):
        layer = name[len(prefix) :].split(".", 1)[0]
        if not layer.isdigit():
            raise ValueError("Unexpected layer key: " + name)
        return int(layer) < layers
    return name.startswith(("model.language_model.", "lm_head."))


def fixture_name(name, layers=4, mtp_source=None):
    """Name in the fixture, or None when dropped; the draft layer moves to ``layers``."""
    if keep_tensor(name, layers):
        return name
    prefix = f"model.language_model.layers.{mtp_source}."
    if mtp_source is not None and name.startswith(prefix):
        return f"model.language_model.layers.{layers}." + name[len(prefix) :]
    return None


def fixture_config(source, layers=4, with_mtp=False):
    if type(layers) is not int or layers not in (4, 8):
        raise ValueError("Fixture layers must be 4 or 8")
    config = copy.deepcopy(source)
    text = config["text_config"]
    expected = (["linear_attention"] * 3 + ["deepseek_sparse_attention"]) * (
        layers // 4
    )
    if config["model_type"] != "glm5_next" or text["layer_types"][:layers] != expected:
        raise ValueError("Expected the pinned GLM KDA/KDA/KDA/MLA prefix")
    config["_test_fixture_only"] = True
    config["_fixture_source"] = {
        "model": MODEL,
        "revision": REVISION,
        "layers": list(range(layers)),
    }
    mtp_source = text.get("num_hidden_layers")
    if with_mtp and text["num_nextn_predict_layers"] != 1:
        raise ValueError("Expected one declared MTP layer")
    text["num_hidden_layers"] = layers
    text["num_nextn_predict_layers"] = int(with_mtp)
    for key in ["layer_types", "mlp_layer_types", "indexer_types"]:
        # The draft layer keeps its own per-layer entry directly after the kept ones.
        draft = text[key][mtp_source : mtp_source + 1] if with_mtp else []
        text[key] = text[key][:layers] + draft
    if with_mtp:
        # Same exclusion tools/prepare_mtp_view.py adds for the BF16 draft layer.
        config["quantization_config"]["ignore"].append(f"*.layers.{layers}.*")
        config["_fixture_source"]["mtp_layer"] = {"source": mtp_source, "as": layers}
    for key in ["kda_layers", "full_attn_layers"]:
        text["linear_attn_config"][key] = [
            i for i in text["linear_attn_config"][key] if i < layers
        ]
    return config


def tensor_hash(tensor):
    import torch

    raw = tensor.reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--with-mtp",
        action="store_true",
        help="keep the BF16 draft layer, renumbered to follow the kept layers",
    )
    args = parser.parse_args(argv)
    if args.source.name != REVISION:
        raise ValueError("Use the pinned original snapshot")
    if args.output.exists():
        raise ValueError("Use a fresh output directory; partial fixtures are preserved")
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_config = json.loads((args.source / "config.json").read_text())
    config = fixture_config(source_config, args.layers, args.with_mtp)
    mtp_source = (
        source_config["text_config"]["num_hidden_layers"] if args.with_mtp else None
    )
    index = json.loads((args.source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    renamed = {k: fixture_name(k, args.layers, mtp_source) for k in index}
    selected = {k: v for k, v in index.items() if renamed[k] is not None}
    args.output.mkdir(parents=True)
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "hf_quant_config.json",
    ]:
        if (args.source / filename).exists():
            shutil.copyfile(args.source / filename, args.output / filename)
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    legacy = args.output / "hf_quant_config.json"
    if args.with_mtp and legacy.exists():
        # vLLM reads the ModelOpt exclusions from this file as well.
        content = json.loads(legacy.read_text(encoding="utf-8"))
        content["quantization"]["exclude_modules"].append(f"*.layers.{args.layers}.*")
        legacy.write_text(json.dumps(content, indent=2), encoding="utf-8")
    buffer, manifest, weight_map = {}, {}, {}
    buffer_bytes, total_bytes, shard_number = 0, 0, 0

    def flush():
        nonlocal buffer, buffer_bytes, shard_number
        if not buffer:
            return
        shard_number += 1
        filename = f"fixture-{shard_number:04d}.safetensors"
        save_file(buffer, args.output / filename, metadata={"format": "pt"})
        with safe_open(args.output / filename, framework="pt", device="cpu") as reader:
            for key in buffer:
                if tensor_hash(reader.get_tensor(key)) != manifest[key]["sha256"]:
                    raise ValueError("Output tensor differs from source: " + key)
                weight_map[key] = filename
        print(
            json.dumps(
                {
                    "shard": filename,
                    "verified_tensors": len(buffer),
                    "bytes": buffer_bytes,
                }
            ),
            flush=True,
        )
        buffer, buffer_bytes = {}, 0

    for source_shard in sorted(set(selected.values())):
        with safe_open(
            args.source / source_shard, framework="pt", device="cpu"
        ) as reader:
            for source_key in sorted(
                k for k, f in selected.items() if f == source_shard
            ):
                tensor = reader.get_tensor(source_key).contiguous()
                key = renamed[source_key]
                size = tensor.numel() * tensor.element_size()
                # Bound packing memory; an individual embedding tensor may exceed 512 MiB.
                if buffer and buffer_bytes + size > 512 * 1024**2:
                    flush()
                manifest[key] = {
                    "source_name": source_key,
                    "source_shard": source_shard,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": size,
                    "sha256": tensor_hash(tensor),
                }
                buffer[key] = tensor
                buffer_bytes += size
                total_bytes += size
    flush()
    if weight_map.keys() != {renamed[k] for k in selected}:
        raise RuntimeError("Fixture tensor coverage is incomplete")
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            indent=2,
        )
    )
    (args.output / "fixture-manifest.json").write_text(
        json.dumps(
            {"source_model": MODEL, "source_revision": REVISION, "tensors": manifest},
            indent=2,
        )
    )
    # Written last and atomically: the runners admit a fixture only on this file.
    write_json(
        args.output / "fixture-status.json",
        {
            "status": "complete",
            "layers": args.layers,
            "mtp_layer": args.with_mtp,
            "tensor_count": len(manifest),
            "total_bytes": total_bytes,
            "all_tensor_bytes_verified": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "tensor_count": len(manifest),
                "total_bytes": total_bytes,
            }
        )
    )


if __name__ == "__main__":
    main()
