"""Source-pinned patch: the sparse-MLA decode takes chunks_per_block from the tokens of one sequence.

The SM120 backend of the image (vLLM 385dce36 after ``patch-reference``) hands every decode step to
FlashInfer without ``chunks_per_block``, whose heuristic then follows the token count of the whole
step, so a partner sequence changes how a request's candidates are summed. This patch asks
``glm53_setup.runtime.mla_decode_cpb`` first, from the host-side step shape already in the metadata;
for the steps it names it runs the same FlashInfer decode entry with the value pinned. Off
(``GLM53_MLA_DECODE_CPB`` unset or 0) and for every other step, the backend calls FlashInfer as pinned.
"""

from . import pinned_patch
from .pinned_patch import replace_once

TARGET = "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
# The file as ``patch-reference`` leaves it (image f53b563b, 2026-09-25).
SOURCE_SHA256 = "72edb55560280be667f52637e62dae3f64409efc6c64e4186101a31816d36be0"
MISMATCH = "sparse MLA SM120 backend source hash mismatch"
RECORD = "glm53-mla-decode-cpb-patch.json"
ANCHOR = (
    "            self._workspace_buffer = _get_workspace_buffer(q.device)\n"
    "\n"
    "        from vllm.utils.flashinfer import (\n"
)
ASK = (
    "            self._workspace_buffer = _get_workspace_buffer(q.device)\n"
    "\n"
    "        from glm53_setup.runtime.mla_decode_cpb import (\n"
    "            decode_chunks_per_block as glm53_decode_chunks_per_block,\n"
    "            pinned_decode as glm53_pinned_decode,\n"
    "        )\n"
    "\n"
    "        glm53_cpb = glm53_decode_chunks_per_block(\n"
    "            num_actual_toks,\n"
    "            attn_metadata.num_reqs,\n"
    "            attn_metadata.max_query_len,\n"
    "            attn_metadata.num_prefills,\n"
    "            self.num_heads,\n"
    "            attn_metadata.topk_tokens,\n"
    "        )\n"
    "        if glm53_cpb is not None and glm53_pinned_decode(\n"
    "            q,\n"
    "            kv_c_and_k_pe_cache,\n"
    "            topk_indices_physical,\n"
    "            output,\n"
    "            self._workspace_buffer,\n"
    "            self.scale,\n"
    "            self.kv_scale_format,\n"
    "            glm53_cpb,\n"
    "        ):\n"
    "            return output, None\n"
    "\n"
    "        from vllm.utils.flashinfer import (\n"
)
HEADER = (
    "# Modified by GLM setup: the decode takes chunks_per_block from the tokens of one sequence.\n"
    "# Original vLLM Apache-2.0 notices below remain applicable.\n"
)


def patch_text(text):
    if "glm53_decode_chunks_per_block" in text:
        raise ValueError("MLA decode chunks_per_block patch already applied")
    patched = HEADER + replace_once(text, ANCHOR, ASK)
    compile(patched, TARGET, "exec")
    return patched


def prepare(package):
    return pinned_patch.prepare(package, TARGET, SOURCE_SHA256, MISMATCH, patch_text)


def main(argv=None):
    pinned_patch.main(
        argv,
        doc=__doc__,
        target=TARGET,
        sha256=SOURCE_SHA256,
        prepare=prepare,
        record=RECORD,
    )


if __name__ == "__main__":
    main()
