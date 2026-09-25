"""Tabulate weight-space error of an NVFP4 requantized checkpoint, tensor by tensor."""

import argparse
import json
import math
from pathlib import Path

from ..io import write_json

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GROUP = 16


def e4m3fn_table():
    """All 256 float8_e4m3fn values; the format has no infinity and one NaN pair."""
    values = []
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        if exponent == 15 and mantissa == 7:
            value = math.nan
        elif exponent == 0:
            value = mantissa / 8 * 2.0**-6
        else:
            value = (1 + mantissa / 8) * 2.0 ** (exponent - 7)
        values.append(-value if byte >> 7 else value)
    return values


def dequant_nvfp4(packed, block_scale, global_scale):
    """uint8 [n, k/2] codes, uint8 [n, k/16] e4m3fn scales, scalar -> float32 [n, k].

    The low nibble holds the even column, as ModelOpt NVFP4 checkpoints pack it.
    """
    import numpy as np

    rows, half = packed.shape
    if block_scale.shape != (rows, half * 2 // GROUP):
        raise ValueError("Block scales do not match the packed weight shape")
    codes = np.stack([packed & 0xF, packed >> 4], axis=-1).reshape(rows, half * 2)
    magnitude = np.asarray(E2M1, dtype=np.float32)[codes & 7]
    values = np.where(codes & 8, -magnitude, magnitude)
    scales = np.asarray(e4m3fn_table(), dtype=np.float32)[block_scale]
    return (
        values.reshape(rows, -1, GROUP) * scales[..., None] * np.float32(global_scale)
    ).reshape(rows, half * 2)


def tensor_error(original, restored):
    """Relative Frobenius error, largest absolute error and the worst output row."""
    import numpy as np

    if original.shape != restored.shape:
        raise ValueError("Shapes differ")
    original = original.astype(np.float64)
    delta = restored.astype(np.float64) - original
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite value in the comparison")
    norm = float(np.linalg.norm(original))
    row_norm = np.linalg.norm(original, axis=1)
    row_error = np.linalg.norm(delta, axis=1)
    row_relative = np.where(row_norm > 0, row_error / np.maximum(row_norm, 1e-300), 0.0)
    worst = int(row_relative.argmax())
    max_abs = float(np.abs(delta).max())
    amax = float(np.abs(original).max())
    return {
        "shape": list(original.shape),
        "relative_frobenius": float(np.linalg.norm(delta)) / norm if norm else 0.0,
        "max_abs": max_abs,
        "max_abs_over_amax": max_abs / amax if amax else 0.0,
        "worst_row": worst,
        "worst_row_relative": float(row_relative[worst]),
    }


def module_kind(name):
    """Group key: the projection name without its layer number."""
    parts = name.removesuffix(".weight").split(".")
    return ".".join(p for p in parts[3:] if not p.isdigit()) or parts[0]


def summarize(rows):
    kinds = {}
    for row in rows:
        kinds.setdefault(row["kind"], []).append(row["relative_frobenius"])
    return {
        kind: {
            "tensors": len(values),
            "max_relative_frobenius": max(values),
            "median_relative_frobenius": sorted(values)[len(values) // 2],
        }
        for kind, values in sorted(kinds.items())
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--write-status",
        action="store_true",
        help="replace the inherited fixture-status.json in --quantized with this result",
    )
    args = parser.parse_args(argv)
    import torch
    from safetensors import safe_open

    from .make_fixture import tensor_hash

    def weight_map(root):
        return json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

    before, after = weight_map(args.original), weight_map(args.quantized)
    readers = {}

    def tensor(root, mapping, key):
        path = root / mapping[key]
        if path not in readers:
            readers[path] = safe_open(path, framework="pt", device="cpu")
        return readers[path].get_tensor(key)

    rows, changed, added = [], [], []
    copied = 0
    for key in sorted(after):
        module = key.removesuffix(".weight")
        if key not in before:
            added.append(key)
        elif (
            key.endswith(".weight")
            and module + ".weight_scale_2" in after
            and module + ".weight_scale_2" not in before
        ):
            source = tensor(args.original, before, key)
            restored = dequant_nvfp4(
                tensor(args.quantized, after, key).numpy(),
                tensor(args.quantized, after, module + ".weight_scale")
                .view(torch.uint8)
                .numpy(),
                float(tensor(args.quantized, after, module + ".weight_scale_2")),
            )
            row = tensor_error(source.float().numpy(), restored)
            row.update(name=key, kind=module_kind(key), source_dtype=str(source.dtype))
            rows.append(row)
        elif tensor_hash(tensor(args.original, before, key).contiguous()) == (
            tensor_hash(tensor(args.quantized, after, key).contiguous())
        ):
            copied += 1
        else:
            changed.append(key)
    missing = sorted(set(before) - set(after))
    rows.sort(key=lambda row: row["relative_frobenius"], reverse=True)
    report = {
        "scope": "weight-space error only; says nothing about logits or candidates",
        "original": str(args.original),
        "quantized": str(args.quantized),
        "requantized_tensors": len(rows),
        "byte_identical_tensors": copied,
        "added_tensors": len(added),
        "changed_without_requant": changed,
        "missing_from_quantized": missing,
        "passed": bool(rows) and not changed and not missing,
        "by_kind": summarize(rows),
        "tensors": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "quant-error.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if args.write_status:
        inherited = json.loads((args.original / "fixture-status.json").read_text())
        inherited.pop("total_bytes", None)  # the source's size, not this directory's
        inherited.update(
            tensor_count=len(after),
            all_tensor_bytes_verified=report["passed"]
            and inherited.get("all_tensor_bytes_verified") is True,
            derived_from=str(args.original),
            requantized_tensors=len(rows),
            byte_identical_tensors=copied,
        )
        # requant.py hard-links small files; write_json replaces the link, not the source.
        write_json(args.quantized / "fixture-status.json", inherited)
    print(json.dumps({k: report[k] for k in report if k != "tensors"}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
