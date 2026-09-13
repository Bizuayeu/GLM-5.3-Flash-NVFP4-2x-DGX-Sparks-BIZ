"""Patch only the pinned cache publication and worker dispatch boundaries."""

import argparse
import ast
import hashlib
import json
import sysconfig
from pathlib import Path

from .patch_nope_reference import replace_once

SOURCES = {
    "v1/core/kv_cache_manager.py": "ef312a280a1746ca4adc8516d87a05b3c8b331a18015e20758513a33efc7421f",
    "v1/core/block_pool.py": "709b67ebcd2ee393654c3a875f8ecb360b22bcd10f43503be078e692dd38fc20",
    "v1/worker/gpu_worker.py": "8d81dfb9e058f2bf86cca646209df0afe592e0e7a47add9fc6b73cdccfd1d79c",
    "v1/engine/input_processor.py": "91dde56a548c09613ee9aaef8c43631481d12dc1bd439fddc12c6569f5e2e799",
}


def rewrite(name, text):
    if name == "v1/core/kv_cache_manager.py":
        text = replace_once(
            text,
            "class KVCacheManager:",
            "from glm53_setup.runtime.apc_runtime import allocation_guard, publication_end\n\n\nclass KVCacheManager:",
        )
        text = replace_once(
            text,
            "    def allocate_slots(\n",
            "    @allocation_guard\n    def allocate_slots(\n",
        )
        for variable in ("num_tokens_to_cache", "num_computed_tokens"):
            text = replace_once(
                text,
                f"self.coordinator.cache_blocks(request, {variable})",
                f"self.coordinator.cache_blocks(request, publication_end(request, {variable}))",
            )
    elif name == "v1/core/block_pool.py":
        text = replace_once(
            text,
            "class BlockPool:",
            "from glm53_setup.runtime.apc_runtime import validate_publication, validate_partial_publication\n\n\nclass BlockPool:",
        )
        text = replace_once(
            text,
            "        assert block_mask is None or len(block_mask) == len(new_full_blocks)\n",
            "        assert block_mask is None or len(block_mask) == len(new_full_blocks)\n"
            "        validate_publication(request, new_full_blocks, num_cached_blocks, block_size, block_mask)\n",
        )
        text = replace_once(
            text,
            "        if block.is_null:\n            return None\n\n        assert block_size % self.hash_block_size == 0\n",
            "        if block.is_null:\n            return None\n\n"
            "        validate_partial_publication(request, num_tokens)\n"
            "        assert block_size % self.hash_block_size == 0\n",
        )
    elif name == "v1/worker/gpu_worker.py":
        text = replace_once(
            text,
            "class Worker(WorkerBase):",
            "from glm53_setup.runtime.apc_worker import before_forward, warmup_scope\n\n\nclass Worker(WorkerBase):",
        )
        text = replace_once(
            text,
            "            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)\n",
            "            with warmup_scope(self):\n"
            "                warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)\n",
        )
        text = replace_once(
            text,
            "        intermediate_tensors = None\n        forward_pass = scheduler_output.total_num_scheduled_tokens > 0\n",
            "        before_forward(self, scheduler_output)\n"
            "        intermediate_tensors = None\n        forward_pass = scheduler_output.total_num_scheduled_tokens > 0\n",
        )
    elif name == "v1/engine/input_processor.py":
        text = replace_once(
            text,
            "class InputProcessor:",
            "from glm53_setup.runtime.apc_runtime import validate_client_options\n\n\nclass InputProcessor:",
        )
        text = replace_once(
            text,
            "        if isinstance(params, SamplingParams):\n            supported_generation_tasks = [\n",
            "        if isinstance(params, SamplingParams):\n"
            "            try:\n"
            "                validate_client_options(params.extra_args)\n"
            "            except ValueError as error:\n"
            "                raise VLLMValidationError(str(error)) from error\n"
            "            supported_generation_tasks = [\n",
        )
    else:
        raise ValueError("Unknown APC/LPA patch target")
    ast.parse(text)
    return text


def prepare(package):
    patches = {}
    for name, digest in SOURCES.items():
        raw = (package / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("APC/LPA source hash mismatch: " + name)
        patches[name] = rewrite(name, raw.decode("utf-8"))
    return patches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-package",
        type=Path,
        default=Path(sysconfig.get_paths()["purelib"]) / "vllm",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    patches = prepare(args.vllm_package)
    if not args.dry_run:
        for name, text in patches.items():
            (args.vllm_package / name).write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    "source_sha256": SOURCES[name],
                    "patched_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                for name, text in patches.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
