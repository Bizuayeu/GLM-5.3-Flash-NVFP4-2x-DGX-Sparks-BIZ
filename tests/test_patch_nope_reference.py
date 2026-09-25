import hashlib
import importlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup.runtime import patch_mla_decode_cpb, patch_nope_reference

ROOT = Path(__file__).resolve().parents[1]
MLA = "model_executor/layers/mla.py"
BACKEND = "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"

# The lines of vLLM 385dce36 model_executor/layers/mla.py that the patch touches.
MLA_SOURCE = """class MultiHeadLatentAttentionWrapper:
    def __init__(self, qk_nope_head_dim, qk_rope_head_dim):
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.mla_attn = MLAAttention(
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )

    def forward(self, q, k_pe):
        attn_out = self.mla_attn(
            q,
            k_pe,
        )
        return attn_out
"""

# The lines of the unpatched SM120 backend that patch-reference and, after it,
# patch_mla_decode_cpb touch, in the pinned file's order: the flag in __init__,
# the top-k read, then output = q.new_empty( above the workspace buffer.
BACKEND_SOURCE = """class FlashInferMLASparseSM120Impl:
    def __init__(self, vllm_config, model_type, indexer):
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        self.topk_indices_buffer = indexer.topk_indices_buffer
        self.supports_quant_query_input = False
        self._workspace_buffer = None

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        num_actual_toks = q.shape[0]
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        topk_indices_physical = convert(topk_indices, attn_metadata)

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        return flashinfer_trtllm_batch_decode_with_kv_cache_mla(q), None
"""

ANCHORS = (
    (MLA, "        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim\n"),
    (
        MLA,
        "            qk_nope_head_dim=self.qk_nope_head_dim,\n"
        "            qk_rope_head_dim=self.qk_rope_head_dim,",
    ),
    (MLA, "        attn_out = self.mla_attn(\n"),
    (
        BACKEND,
        "        self.kv_scale_format = _kv_scale_format_for_model(model_type)\n",
    ),
    (BACKEND, "        output = q.new_empty(\n"),
    # add_candidate_order
    (BACKEND, "        self.supports_quant_query_input = False\n"),
    (BACKEND, "        topk_indices = self.topk_indices_buffer[:num_actual_toks]\n"),
)


def prepare(sources=None, pin=True):
    """patch_nope_reference.prepare on the synthetic sources, pinned to their hashes."""
    sources = sources or {MLA: MLA_SOURCE, BACKEND: BACKEND_SOURCE}
    with tempfile.TemporaryDirectory() as directory:
        package = Path(directory) / "vllm"
        for name, text in sources.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode())
        hashes = {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in sources.items()
        }
        with patch.dict(patch_nope_reference.HASHES, hashes if pin else {}):
            outputs = patch_nope_reference.prepare(package)
    return {name: data.decode() for name, data in outputs.items()}


def between(text, start, end):
    begin = text.index(start)
    return text[begin : text.index(end, begin)]


class PatchNopeReferenceTests(unittest.TestCase):
    def test_every_anchor_is_replaced_once_under_the_modification_notice(self):
        outputs = prepare()
        self.assertEqual(list(outputs), [MLA, BACKEND])
        for text in outputs.values():
            self.assertTrue(text.startswith("# Modified by "))
            self.assertIn("# Original vLLM notices below remain applicable", text)
        mla, backend = outputs[MLA], outputs[BACKEND]
        self.assertIn(
            "qk_rope_head_dim=self.qk_rope_head_dim + self._glm53_reference_pad,", mla
        )
        self.assertLess(
            mla.index("q = torch.nn.functional.pad(q, (0, self._glm53_reference_pad))"),
            mla.index("attn_out = self.mla_attn("),
        )
        self.assertEqual(backend.count("return sparse_nope_reference("), 1)
        self.assertLess(
            backend.index("return sparse_nope_reference("),
            backend.index("output = q.new_empty("),
        )
        self.assertEqual(
            backend.count("topk_indices = canonical_logical_candidates(topk_indices)"),
            1,
        )

    def test_a_missing_or_repeated_anchor_is_refused(self):
        sources = {MLA: MLA_SOURCE, BACKEND: BACKEND_SOURCE}
        for name, anchor in ANCHORS:
            self.assertEqual(sources[name].count(anchor), 1, anchor)
            for drifted in (sources[name].replace(anchor, ""), sources[name] + anchor):
                with (
                    self.subTest(anchor=anchor),
                    self.assertRaisesRegex(ValueError, "anchor"),
                ):
                    prepare({**sources, name: drifted})

    def test_the_reference_path_needs_glm_nope_and_the_switch(self):
        outputs = prepare()
        pad = between(outputs[MLA], "self._glm53_reference_pad = 64 if (", ") else 0")
        flag = between(
            outputs[BACKEND], "self._glm53_reference_nope = (", "\n        )"
        )
        for condition, text in (
            ("'glm5_next_text'", pad),
            ("qk_rope_head_dim == 0", pad),
            ("os.environ.get('GLM53_REFERENCE_ATTENTION') == '1'", pad),
            ("model_type == 'glm5_next_text'", flag),
            ("qk_rope_head_dim == 0", flag),
            ("os.environ.get('GLM53_REFERENCE_ATTENTION') == '1'", flag),
        ):
            self.assertIn(condition, text)

    def test_a_source_off_the_pinned_hash_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
            prepare(pin=False)


class BackendPatchOrderTests(unittest.TestCase):
    """What the image build leaves in forward_mqa of the SM120 backend."""

    def test_the_backend_is_patched_by_patch_reference_then_by_mla_decode_cpb(self):
        dockerfile = (ROOT / "docker/Dockerfile.reference").read_text(encoding="utf-8")
        runs = re.findall(r"^RUN python3 -m glm53_setup[ .](\S+)$", dockerfile, re.M)
        self.assertLess(
            runs.index("patch-reference"),
            runs.index("runtime.patch_mla_decode_cpb"),
        )
        # No build patch in between edits the same file.
        editors = [
            name
            for name in runs
            if name.startswith("runtime.patch_")
            and getattr(importlib.import_module("glm53_setup." + name), "TARGET", None)
            == BACKEND
        ]
        self.assertEqual(editors, ["runtime.patch_mla_decode_cpb"])
        self.assertIn(BACKEND, patch_nope_reference.HASHES)

    def test_cpb_insertion_sits_below_reference_return(self):
        # A fact about the patch outputs, not a verdict on it: with the reference
        # flag set, forward_mqa returns before the chunks_per_block question is
        # asked. See the investigation of 1.14.0's reachability.
        patched = patch_mla_decode_cpb.patch_text(prepare()[BACKEND])
        self.assertTrue(patched.startswith(patch_mla_decode_cpb.HEADER))
        forward = patched.index("def forward_mqa(")
        reference = patched.index("return sparse_nope_reference(")
        cpb = patched.index("glm53_cpb = glm53_decode_chunks_per_block(")
        self.assertLess(forward, reference)
        self.assertLess(reference, cpb)


if __name__ == "__main__":
    unittest.main()
