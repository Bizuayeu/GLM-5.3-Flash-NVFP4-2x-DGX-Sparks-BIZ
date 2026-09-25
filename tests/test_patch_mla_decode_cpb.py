import unittest

from glm53_setup.runtime import patch_mla_decode_cpb

# The lines of the image's v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py (vLLM 385dce36
# after ``patch-reference``) that the patch touches.
SOURCE = """# Modified by GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ contributors.
# Changes: GLM NoPE zero-padding, canonical logical candidates and reference attention.
# Original vLLM notices below remain applicable; see distribution NOTICE.
# SPDX-License-Identifier: Apache-2.0
class FlashInferMLASparseSM120Impl:
    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )
"""


class PatchMlaDecodeCpbTest(unittest.TestCase):
    def test_asks_before_the_flashinfer_call_and_keeps_it_otherwise(self):
        patched = patch_mla_decode_cpb.patch_text(SOURCE)
        self.assertTrue(patched.startswith(patch_mla_decode_cpb.HEADER))
        ask = patched.index("glm53_decode_chunks_per_block(")
        call = patched.index("from vllm.utils.flashinfer import (")
        workspace = patched.index(
            "self._workspace_buffer = _get_workspace_buffer(q.device)"
        )
        self.assertLess(workspace, ask)
        self.assertLess(ask, call)
        self.assertIn("attn_metadata.num_reqs", patched)
        self.assertIn("attn_metadata.max_query_len", patched)
        self.assertIn("attn_metadata.num_prefills", patched)
        self.assertIn("return output, None", patched)
        # The original call stays for every step the rule leaves alone.
        self.assertIn("flashinfer_trtllm_batch_decode_with_kv_cache_mla,", patched)

    def test_refuses_a_patched_or_drifted_source(self):
        patched = patch_mla_decode_cpb.patch_text(SOURCE)
        with self.assertRaises(ValueError):
            patch_mla_decode_cpb.patch_text(patched)
        with self.assertRaises(ValueError):
            patch_mla_decode_cpb.patch_text(
                SOURCE.replace("_get_workspace_buffer(q.device)", "x")
            )


if __name__ == "__main__":
    unittest.main()
