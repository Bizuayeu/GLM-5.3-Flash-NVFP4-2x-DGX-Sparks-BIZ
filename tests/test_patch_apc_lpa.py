import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup.runtime import patch_apc_lpa

# The lines of each pinned vLLM 385dce36 file that the patch touches.
SOURCES = {
    "v1/core/kv_cache_manager.py": """class KVCacheManager:
    def allocate_slots(
        self, request, num_tokens_to_cache
    ):
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

    def cache_blocks(self, request, num_computed_tokens):
        self.coordinator.cache_blocks(request, num_computed_tokens)
""",
    "v1/core/block_pool.py": """class BlockPool:
    def cache_full_blocks(self, request, new_full_blocks, block_mask):
        assert block_mask is None or len(block_mask) == len(new_full_blocks)

    def cache_partial_block(self, request, block, block_size, num_tokens):
        if block.is_null:
            return None

        assert block_size % self.hash_block_size == 0
""",
    "v1/worker/gpu_worker.py": """class Worker(WorkerBase):
    def compile_or_warm_up_model(self):
        if self.warm:
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)

    def execute_model(self, scheduler_output):
        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
""",
    "v1/engine/input_processor.py": """class InputProcessor:
    def _validate_params(self, params):
        if isinstance(params, SamplingParams):
            supported_generation_tasks = [
                "generate",
            ]
""",
}

ANCHORS = {
    "v1/core/kv_cache_manager.py": (
        "class KVCacheManager:",
        "    def allocate_slots(\n",
        "self.coordinator.cache_blocks(request, num_tokens_to_cache)",
        "self.coordinator.cache_blocks(request, num_computed_tokens)",
    ),
    "v1/core/block_pool.py": (
        "class BlockPool:",
        "        assert block_mask is None or len(block_mask) == len(new_full_blocks)\n",
        "        if block.is_null:\n            return None\n\n"
        "        assert block_size % self.hash_block_size == 0\n",
    ),
    "v1/worker/gpu_worker.py": (
        "class Worker(WorkerBase):",
        "            warmup_kernels(self.model_runner, self.execute_model, "
        "self.sample_tokens)\n",
        "        intermediate_tensors = None\n"
        "        forward_pass = scheduler_output.total_num_scheduled_tokens > 0\n",
    ),
    "v1/engine/input_processor.py": (
        "class InputProcessor:",
        "        if isinstance(params, SamplingParams):\n"
        "            supported_generation_tasks = [\n",
    ),
}


class PatchApcLpaTests(unittest.TestCase):
    def test_every_pinned_target_has_a_fixture(self):
        self.assertEqual(set(SOURCES), set(patch_apc_lpa.SOURCES))
        self.assertEqual(set(ANCHORS), set(patch_apc_lpa.SOURCES))

    def test_each_target_is_rewritten_at_its_anchors(self):
        text = patch_apc_lpa.rewrite(
            "v1/core/kv_cache_manager.py", SOURCES["v1/core/kv_cache_manager.py"]
        )
        self.assertIn("    @allocation_guard\n    def allocate_slots(\n", text)
        self.assertEqual(
            text.count("cache_blocks(request, publication_end(request, "), 2
        )
        text = patch_apc_lpa.rewrite(
            "v1/core/block_pool.py", SOURCES["v1/core/block_pool.py"]
        )
        self.assertIn("validate_publication(request, new_full_blocks, ", text)
        self.assertLess(
            text.index("validate_partial_publication(request, num_tokens)"),
            text.index("assert block_size % self.hash_block_size == 0"),
        )
        text = patch_apc_lpa.rewrite(
            "v1/worker/gpu_worker.py", SOURCES["v1/worker/gpu_worker.py"]
        )
        self.assertLess(
            text.index("with warmup_scope(self):"), text.index("warmup_kernels(")
        )
        self.assertLess(
            text.index("before_forward(self, scheduler_output)"),
            text.index("intermediate_tensors = None"),
        )
        text = patch_apc_lpa.rewrite(
            "v1/engine/input_processor.py", SOURCES["v1/engine/input_processor.py"]
        )
        self.assertLess(
            text.index("validate_client_options(params.extra_args)"),
            text.index("supported_generation_tasks = ["),
        )
        self.assertIn("raise VLLMValidationError(str(error)) from error", text)

    def test_a_missing_or_repeated_anchor_is_refused(self):
        for name, anchors in ANCHORS.items():
            for anchor in anchors:
                self.assertEqual(SOURCES[name].count(anchor), 1, anchor)
                for drifted in (
                    SOURCES[name].replace(anchor, ""),
                    SOURCES[name] + anchor,
                ):
                    with (
                        self.subTest(anchor=anchor),
                        self.assertRaisesRegex(ValueError, "anchor"),
                    ):
                        patch_apc_lpa.rewrite(name, drifted)

    def test_an_unknown_target_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown APC/LPA patch target"):
            patch_apc_lpa.rewrite("v1/other.py", "class Other:\n    pass\n")

    def test_prepare_patches_only_sources_on_their_pinned_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "vllm"
            for name, text in SOURCES.items():
                (package / name).parent.mkdir(parents=True, exist_ok=True)
                (package / name).write_bytes(text.encode())
            with self.assertRaisesRegex(ValueError, "APC/LPA source hash mismatch"):
                patch_apc_lpa.prepare(package)
            pinned = {
                name: hashlib.sha256(text.encode()).hexdigest()
                for name, text in SOURCES.items()
            }
            with patch.dict(patch_apc_lpa.SOURCES, pinned):
                patches = patch_apc_lpa.prepare(package)
            self.assertEqual(
                patches,
                {
                    name: patch_apc_lpa.rewrite(name, text)
                    for name, text in SOURCES.items()
                },
            )
            # prepare returns the texts; only main writes them.
            for name, text in SOURCES.items():
                self.assertEqual((package / name).read_bytes(), text.encode())


if __name__ == "__main__":
    unittest.main()
