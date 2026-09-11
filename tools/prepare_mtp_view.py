"""Build a separate GLM BF16-MTP metadata view; source tensor bytes stay unchanged."""

import argparse
import copy
import hashlib
import json
import os
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def metadata_configs(config, legacy, revision):
    if config.get("model_type") != "glm5_next":
        raise ValueError("Expected GLM-5.3-Flash configuration")
    text = config["text_config"]
    if text["num_nextn_predict_layers"] != 1:
        raise ValueError("This view supports one declared MTP layer")
    if config["quantization_config"]["quant_algo"] != "NVFP4":
        raise ValueError("Expected NVFP4 target metadata")
    layer = text["num_hidden_layers"]
    pattern = f"*.layers.{layer}.*"
    config, legacy = copy.deepcopy(config), copy.deepcopy(legacy)
    for exclusions in (
        config["quantization_config"]["ignore"],
        legacy["quantization"]["exclude_modules"],
    ):
        if pattern not in exclusions:
            exclusions.append(pattern)
    config["_local_mtp_metadata"] = {
        "source_revision": revision,
        "weight_bytes_modified": False,
        "change": "Exclude only the BF16 MTP layer from global NVFP4",
    }
    return config, legacy, layer


def inspect_mtp(snapshot, layer):
    index = json.loads(
        (snapshot / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    selected = {
        k: v
        for k, v in index.items()
        if k.startswith(f"model.language_model.layers.{layer}.")
    }
    if not selected:
        raise ValueError("Declared MTP tensor set is missing")
    headers = {}
    for shard in set(selected.values()):
        if Path(shard).name != shard:
            raise ValueError("Expected a checkpoint-local shard filename")
        with (snapshot / shard).open("rb") as source:
            length = struct.unpack("<Q", source.read(8))[0]
            if (
                not 0
                < length
                <= min(64 * 1024**2, (snapshot / shard).stat().st_size - 8)
            ):
                raise ValueError("Invalid or unexpectedly large tensor header")
            headers[shard] = json.loads(source.read(length))
    dtypes, size = {}, 0
    for key, shard in selected.items():
        tensor = headers[shard][key]
        dtype = tensor["dtype"]
        if dtype not in ("BF16", "F32"):
            raise ValueError(f"MTP is not wholly BF16/F32: {key} ({dtype})")
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
        size += tensor["data_offsets"][1] - tensor["data_offsets"][0]
    return {"tensor_count": len(selected), "tensor_bytes": size, "dtypes": dtypes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Create the symlink view on the Linux model host")
    lock = json.loads((ROOT / "config/runtime.lock.json").read_text(encoding="utf-8"))
    source, output = args.snapshot.resolve(), args.output.resolve()
    if source.name != lock["revision"]:
        parser.error("Use the pinned, checksum-verified snapshot")
    if (
        output.exists()
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        parser.error("Choose a new view separate from the original snapshot")
    config, legacy, layer = metadata_configs(
        json.loads((source / "config.json").read_text(encoding="utf-8")),
        json.loads((source / "hf_quant_config.json").read_text(encoding="utf-8")),
        lock["revision"],
    )
    report = inspect_mtp(source, layer)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "hf_quant_config.json").write_text(
        json.dumps(legacy, indent=2), encoding="utf-8"
    )
    for path in source.iterdir():
        if path.name in ("config.json", "hf_quant_config.json"):
            continue
        if not path.is_file():
            raise ValueError("Unexpected non-file in the snapshot")
        (output / path.name).symlink_to(os.path.relpath(path, output))
    report.update(
        view=str(output),
        source_revision=lock["revision"],
        added_exclusion=f"*.layers.{layer}.*",
        weight_bytes_modified=False,
        config_sha256=hashlib.sha256((output / "config.json").read_bytes()).hexdigest(),
        source_config_sha256=hashlib.sha256(
            (source / "config.json").read_bytes()
        ).hexdigest(),
    )
    (output / "mtp-view.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
