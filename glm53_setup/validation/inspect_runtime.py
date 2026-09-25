"""Inspect a candidate in-container; this step validates the config, not inference."""

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    import vllm
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.quantization import modelopt

    # Nothing here runs inference: inference_validated is always False (the record's shape).
    result = {"vllm": importlib.metadata.version("vllm"), "inference_validated": False}
    engine_args = EngineArgs(
        model=str(args.snapshot),
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        nnodes=2,
        node_rank=0,
        max_model_len=32768,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        kv_cache_dtype="fp8",
        enforce_eager=True,
        enable_prefix_caching=False,
        language_model_only=True,
    )
    try:
        config = engine_args.create_engine_config()
        result.update(
            config_valid=True,
            architecture=config.model_config.hf_config.architectures,
            quantization=config.model_config.quantization,
            quant_config_type=type(config.quant_config).__name__,
        )
    except Exception as error:  # noqa: BLE001 -- persist diagnostics before failing the inspection
        result.update(
            config_valid=False, error_type=type(error).__name__, error=str(error)
        )
    source = Path(inspect.getfile(modelopt))
    (args.output / "modelopt.py").write_text(source.read_text(), encoding="utf-8")
    package = Path(vllm.__file__).parent
    filenames = [
        "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
        "models/glm5next/nvidia/attention.py",
        "models/glm5next/nvidia/model.py",
        "models/glm5next/nvidia/mtp.py",
        "models/glm5next/nvidia/ops/kpool_compress.py",
    ]
    result["source_files"] = []
    for relative in filenames:
        source = package / relative
        if source.exists():
            target = args.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            result["source_files"].append(relative)
    (args.output / "config-inspection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    if not result.get("config_valid"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
