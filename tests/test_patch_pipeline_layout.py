import unittest

from glm53_setup.runtime import patch_pipeline_layout

# The lines of vLLM 385dce36 v1/attention/backends/utils.py that the patch touches.
SOURCE = """def resolve_layout(vllm_config, supported_layouts):
    assert all(names == supported_layouts[0] for names in supported_layouts[1:]), (
        f"Workers disagree on supported KV cache layouts: {supported_layouts}."
    )
    return supported_layouts[0]
"""


class PatchTests(unittest.TestCase):
    def test_two_pp_stages_agree_on_a_common_layout_before_the_pinned_assert(self):
        patched = patch_pipeline_layout.patch_text(SOURCE)
        self.assertTrue(patched.startswith("# Modified by GLM setup"))
        self.assertIn(
            "if vllm_config.parallel_config.pipeline_parallel_size == 2:", patched
        )
        self.assertIn("common = common_layout_names(supported_layouts)", patched)
        # The pinned assertion stays and runs after the layouts were reconciled.
        self.assertLess(
            patched.index("common_layout_names(supported_layouts)"),
            patched.index("assert all(names == supported_layouts[0]"),
        )

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_pipeline_layout.patch_text(
                SOURCE.replace("Workers disagree", "Workers differ")
            )
        with self.assertRaises(ValueError):
            patch_pipeline_layout.patch_text(patch_pipeline_layout.patch_text(SOURCE))


if __name__ == "__main__":
    unittest.main()
