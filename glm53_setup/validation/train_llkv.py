"""Fit a small frozen-basis residual projector from teacher activation shards."""

import argparse
import json
import math
from pathlib import Path


def solve_projected_ridge(gram, cross, ridge):
    import torch

    scale = gram.diag().mean().clamp_min(torch.finfo(gram.dtype).eps)
    matrix = gram + ridge * scale * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    return torch.linalg.solve(matrix, cross)


def densify_affine_weights(weights):
    """Exact algebraic export for the original residual-only experimental runtime."""
    import torch

    first = next(iter(weights.values()))
    identity = torch.eye(
        first["mean"].numel(), device=first["mean"].device, dtype=first["mean"].dtype
    )
    return {
        layer: {
            "mean": w["mean"],
            "down": identity,
            "up": torch.diag(w["scale"] - 1) + w["down"] @ w["up"],
            "bias": w["bias"] + w["mean"] * (w["scale"] - 1),
        }
        for layer, w in weights.items()
    }


def cpu_weights(weights):
    """Keep shared basis tensors shared in the serialized artifact."""
    converted = {}
    result = {}
    for layer, tensors in weights.items():
        result[layer] = {}
        for key, value in tensors.items():
            if id(value) not in converted:
                converted[id(value)] = value.cpu()
            result[layer][key] = converted[id(value)]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cut", type=int, help="Use a later cut from a wider teacher capture"
    )
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dense-compat",
        action="store_true",
        help="Also export a larger artifact for the first runtime",
    )
    args = parser.parse_args(argv)
    if args.rank < 1 or not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("rank and ridge must be positive")
    import torch

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.captures / "result.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only a completed teacher collection can be fitted")
    cut = args.cut if args.cut is not None else manifest["cut"]
    if not manifest["cut"] <= cut < 44:
        raise ValueError("Requested cut is outside the captured suffix")
    cases = manifest["cases"]
    train = [c for c in cases if c["split"] == "train"]
    validation = [c for c in cases if c["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Train and validation documents must both be present")
    first_dir = args.captures / train[0]["id"] / "rank-0"
    layers = sorted(
        int(p.stem.split("-")[1])
        for p in first_dir.glob("layer-*.pt")
        if int(p.stem.split("-")[1]) >= cut
    )
    if layers != list(range(cut, 45)):
        raise ValueError("Expected full-model suffix captures through layer 44")

    def read(case, layer):
        path = args.captures / case["id"] / "rank-0" / f"layer-{layer}.pt"
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if (
            tensor.shape != (case["prompt_tokens"], 4096)
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Invalid teacher activation tensor")
        return tensor.to(device=args.device, dtype=torch.float32)

    # Basis sampling uses only training documents; no validation/test token is fitted.
    sample = []
    mean = torch.zeros(4096, device=args.device)
    source_square = torch.zeros_like(mean)
    target_sum = {layer: torch.zeros_like(mean) for layer in layers[1:]}
    paired_sum = {layer: torch.zeros_like(mean) for layer in layers[1:]}
    total = 0
    for case in train:
        source = read(case, cut)
        mean += source.sum(0)
        source_square += source.square().sum(0)
        for layer in layers[1:]:
            target = read(case, layer)
            target_sum[layer] += target.sum(0)
            paired_sum[layer] += (source * target).sum(0)
        total += source.shape[0]
        # Stratify across documents with deterministic, evenly spaced token positions.
        indices = torch.linspace(
            0, source.shape[0] - 1, steps=min(16, source.shape[0]), device=args.device
        ).long()
        sample.append(source[indices])
    mean /= total
    variance = (source_square / total - mean.square()).clamp_min(0)
    denominator = variance + args.ridge * variance.mean().clamp_min(
        torch.finfo(variance.dtype).eps
    )
    scales = {
        layer: (paired_sum[layer] / total - mean * (target_sum[layer] / total))
        / denominator
        for layer in layers[1:]
    }
    del source_square, paired_sum, target_sum
    sampled = torch.cat(sample, dim=0) - mean
    rank = min(args.rank, sampled.shape[0] - 1, 4096)
    _, _, down = torch.pca_lowrank(sampled, q=rank, center=False, niter=2)
    del sample, sampled
    gram = torch.zeros((rank, rank), device=args.device)
    cross = {
        layer: torch.zeros((rank, 4096), device=args.device) for layer in layers[1:]
    }
    bias = {layer: torch.zeros(4096, device=args.device) for layer in layers[1:]}
    for number, case in enumerate(train):
        source = read(case, cut)
        projected = (source - mean) @ down
        gram += projected.T @ projected
        for layer in layers[1:]:
            residual = read(case, layer) - source * scales[layer]
            cross[layer] += projected.T @ residual
            bias[layer] += residual.sum(0)
        if number % 32 == 0:
            print("fit", number, flush=True)
    weights = {}
    for layer in layers[1:]:
        weights[layer] = {
            "mean": mean,
            "down": down,
            "up": solve_projected_ridge(gram, cross[layer], args.ridge),
            "bias": bias[layer] / total,
            "scale": scales[layer],
        }
    del cross, gram
    metrics = {}
    # Test documents stay sealed until downstream candidate selection is complete.
    for split, selection in (("train", train), ("validation", validation)):
        sums = {
            layer: {
                "mse_sum": 0.0,
                "identity_mse_sum": 0.0,
                "target_energy": 0.0,
                "elements": 0,
            }
            for layer in layers[1:]
        }
        for case in selection:
            source = read(case, cut)
            for layer, w in weights.items():
                target = read(case, layer)
                prediction = (
                    source * w["scale"]
                    + ((source - w["mean"]) @ w["down"]) @ w["up"]
                    + w["bias"]
                )
                row = sums[layer]
                row["mse_sum"] += (prediction - target).square().sum().item()
                row["identity_mse_sum"] += (source - target).square().sum().item()
                row["target_energy"] += target.square().sum().item()
                row["elements"] += target.numel()
        metrics[split] = {
            layer: {
                "normalized_mse": row["mse_sum"] / row["target_energy"],
                "identity_normalized_mse": row["identity_mse_sum"]
                / row["target_energy"],
                "mse": row["mse_sum"] / row["elements"],
            }
            for layer, row in sums.items()
        }
    artifact = {
        "cut": cut,
        "layers": 45,
        "rank": rank,
        "ridge": args.ridge,
        "teacher_revision": "423acf37583782c51c142d145aef733d72943d93",
        "teacher_precision": "NVFP4-Marlin-W4A16",
        "seed": 42,
        "format_version": 2,
        "representation": "diagonal-low-rank",
        "weights": cpu_weights(weights),
    }
    torch.save(artifact, args.output / "projector.pt")
    if args.dense_compat:
        compatible = {
            **artifact,
            "format_version": 1,
            "representation": "dense-residual-compatibility",
            "rank": 4096,
            "fitted_rank": rank,
            "weights": cpu_weights(densify_affine_weights(weights)),
        }
        torch.save(compatible, args.output / "projector-compat.pt")
    summary = {
        "status": "complete",
        "cut": cut,
        "rank": rank,
        "ridge": args.ridge,
        "training_documents": len(train),
        "validation_documents": len(validation),
        "training_tokens": total,
        "test_used": False,
        "metrics": metrics,
    }
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
