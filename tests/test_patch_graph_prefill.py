import unittest

from glm53_setup.runtime import patch_graph_prefill

# The lines of vLLM 385dce36 v1/worker/gpu/model_runner.py that the patch touches.
SOURCE = """import torch


def dispatch(batch_req_state, is_profile, skip_compiled):
    return run(
        need_eager=is_profile or skip_compiled,
    )
"""


class PatchTests(unittest.TestCase):
    def test_lpa_prefill_is_routed_eager_through_the_policy(self):
        patched = patch_graph_prefill.patch_text(SOURCE)
        self.assertTrue(patched.startswith("# Modified by GLM setup"))
        self.assertIn(
            "from glm53_setup.runtime.graph_policy import force_eager_prefill", patched
        )
        self.assertIn(
            "need_eager=is_profile or skip_compiled"
            " or _glm53_force_eager_prefill(batch_req_state),",
            patched,
        )
        # The import lands where the pinned file imports torch, before its use.
        self.assertLess(
            patched.index("import torch\nfrom glm53_setup"),
            patched.index("_glm53_force_eager_prefill(batch_req_state)"),
        )

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            patch_graph_prefill.patch_text(
                SOURCE.replace("skip_compiled,", "skip_compiled or other,")
            )
        with self.assertRaises(ValueError):
            patch_graph_prefill.patch_text(patch_graph_prefill.patch_text(SOURCE))


if __name__ == "__main__":
    unittest.main()
