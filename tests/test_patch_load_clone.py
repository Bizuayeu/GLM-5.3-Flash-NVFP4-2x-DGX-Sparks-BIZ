import hashlib
import tempfile
import unittest
from pathlib import Path

from glm53_setup.runtime import patch_load_clone

# The lines of vLLM 385dce36 model_executor/model_loader/weight_utils.py around the anchor.
SOURCE = """def safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load):
    for st_file in hf_weights_files:
        if safetensors_load_strategy == "torchao":
            with safe_open(st_file, framework="pt") as f:
                state_dict = {}
                for name in f.keys():  # noqa: SIM118
                    state_dict[name] = f.get_tensor(name)
        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
"""


class PatchTests(unittest.TestCase):
    def test_the_lazy_branch_clones_each_tensor(self):
        patched = patch_load_clone.patch_text(SOURCE)
        self.assertIn("param = f.get_tensor(name).clone()\n", patched)
        self.assertNotIn("param = f.get_tensor(name)\n", patched)
        self.assertTrue(patched.startswith("# Modified by GLM setup"))

    def test_the_torchao_branch_is_left_alone(self):
        patched = patch_load_clone.patch_text(SOURCE)
        self.assertIn("state_dict[name] = f.get_tensor(name)\n", patched)

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_load_clone.patch_text(
                SOURCE.replace("yield name, param", "yield name, x")
            )
        with self.assertRaises(ValueError):
            patch_load_clone.patch_text(patch_load_clone.patch_text(SOURCE))


class PrepareTests(unittest.TestCase):
    def test_the_pinned_hash_gates_the_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "vllm"
            target = package / patch_load_clone.TARGET
            target.parent.mkdir(parents=True)
            # Bytes, not write_text: on Windows that writes CRLF and the anchor misses.
            target.write_bytes(SOURCE.encode("utf-8"))
            with self.assertRaisesRegex(
                ValueError, "weight loader source hash mismatch"
            ):
                patch_load_clone.prepare(package)
            original = patch_load_clone.SOURCE_SHA256
            try:
                patch_load_clone.SOURCE_SHA256 = hashlib.sha256(
                    target.read_bytes()
                ).hexdigest()
                patched = patch_load_clone.prepare(package)
            finally:
                patch_load_clone.SOURCE_SHA256 = original
            self.assertIn(b"f.get_tensor(name).clone()", patched)


if __name__ == "__main__":
    unittest.main()
