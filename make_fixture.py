"""Build and byte-verify a four-layer checkpoint without altering source weights."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from download_model import MODEL, REVISION


def keep_tensor(name):
    prefix = "model.language_model.layers."
    if name.startswith(prefix):
        layer = name[len(prefix) :].split(".", 1)[0]
        if not layer.isdigit():
            raise ValueError("Unexpected layer key: " + name)
        return int(layer) < 4
    return name.startswith(("model.language_model.", "lm_head."))


def fixture_config(source):
    config = copy.deepcopy(source)
    text = config["text_config"]
    expected = ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
    if config["model_type"] != "glm5_next" or text["layer_types"][:4] != expected:
        raise ValueError("Expected the pinned GLM KDA/KDA/KDA/MLA prefix")
    config["_test_fixture_only"] = True
    config["_fixture_source"] = {
        "model": MODEL,
        "revision": REVISION,
        "layers": [0, 1, 2, 3],
    }
    text["num_hidden_layers"] = 4
    text["num_nextn_predict_layers"] = 0
    for key in ["layer_types", "mlp_layer_types", "indexer_types"]:
        text[key] = text[key][:4]
    for key in ["kda_layers", "full_attn_layers"]:
        text["linear_attn_config"][key] = [
            i for i in text["linear_attn_config"][key] if i < 4
        ]
    return config


def tensor_hash(tensor):
    import torch

    raw = tensor.reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.name != REVISION:
        raise ValueError("Use the pinned original snapshot")
    if args.output.exists():
        raise ValueError("Use a fresh output directory; partial fixtures are preserved")
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = fixture_config(json.loads((args.source / "config.json").read_text()))
    index = json.loads((args.source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    selected = {k: v for k, v in index.items() if keep_tensor(k)}
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
            for key in sorted(k for k, f in selected.items() if f == source_shard):
                tensor = reader.get_tensor(key).contiguous()
                size = tensor.numel() * tensor.element_size()
                # Bound packing memory; an individual embedding tensor may exceed 512 MiB.
                if buffer and buffer_bytes + size > 512 * 1024**2:
                    flush()
                manifest[key] = {
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
    if weight_map.keys() != selected.keys():
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
    (args.output / "fixture-status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "layers": 4,
                "tensor_count": len(manifest),
                "total_bytes": total_bytes,
                "all_tensor_bytes_verified": True,
            },
            indent=2,
        )
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
