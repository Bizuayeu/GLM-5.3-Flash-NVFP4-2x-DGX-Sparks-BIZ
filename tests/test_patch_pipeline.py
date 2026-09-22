import ast
import unittest

from glm53_setup.runtime import patch_pipeline

# The lines of vLLM 385dce36 models/glm5next/nvidia/model.py that the patch touches.
SOURCE = """class IntermediateTensors:
    pass


class Glm5NextModel:
    def __init__(self, vllm_config):
        config = vllm_config.model_config.hf_config
        self.config = config

    def forward(self, hidden_states, residual, intermediate_tensors):
        if intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            # post/comb (deferred mHC hc_post state) are not propagated across
            # PP ranks; the receiving rank's first mHC layer uses standalone pre.
            post = None
            comb = None
        for layer in self.layers:
            hidden_states, residual, post, comb = layer(hidden_states, residual)
        if not self.last:
            # PP is gated off for GLM-5.3-Flash (no make_empty_intermediate_tensors),
            # so this branch is not exercised. post/comb are the deferred
            # hc_post state of this rank's last mHC layer; a future PP path
            # would need to propagate them, but for now they are dropped (the
            # receiving rank's first layer would fall back to standalone pre).
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        return hidden_states


class Glm5NextForCausalLM:
    def __init__(self, vllm_config):
        self.model = Glm5NextModel(vllm_config)
        # Glm5NextForCausalLM does not implement make_empty_intermediate_tensors,
        # so pipeline parallelism is gated off (consistent with the text-only
        # model) and we intentionally do not alias it here.


class Glm5NextForConditionalGeneration:
    def __init__(self, vllm_config):
        self.language_model = Glm5NextForCausalLM(vllm_config)
"""


def methods(text, class_name):
    node = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return [n.name for n in node.body if isinstance(n, ast.FunctionDef)]


class PatchTests(unittest.TestCase):
    def test_deferred_state_crosses_the_stage_boundary_and_pp_buffers_exist(self):
        patched = patch_pipeline.patch_text(SOURCE)
        self.assertTrue(patched.startswith("# Modified by GLM setup"))
        self.assertIn("validate_pipeline(vllm_config)", patched)
        self.assertIn('post = intermediate_tensors["post"]', patched)
        self.assertIn('"post": post, "comb": comb}', patched)
        self.assertNotIn("post = None", patched)
        for class_name in (
            "Glm5NextModel",
            "Glm5NextForCausalLM",
            "Glm5NextForConditionalGeneration",
        ):
            self.assertIn(
                "make_empty_intermediate_tensors", methods(patched, class_name)
            )
        # The wrappers delegate inward; the model allocates.
        self.assertIn("return self.model.make_empty_intermediate_tensors", patched)
        self.assertIn(
            "return self.language_model.make_empty_intermediate_tensors", patched
        )
        self.assertIn("allocate_intermediate(self.config", patched)

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_pipeline.patch_text(SOURCE.replace("post = None", "post = 0"))
        with self.assertRaises(ValueError):
            patch_pipeline.patch_text(patch_pipeline.patch_text(SOURCE))

    def test_append_method_refuses_a_class_that_already_has_the_method(self):
        text = (
            "class A:\n    def make_empty_intermediate_tensors(self):\n        pass\n"
        )
        with self.assertRaises(ValueError):
            patch_pipeline.append_method(
                text, "A", "    def other(self):\n        pass"
            )


if __name__ == "__main__":
    unittest.main()
