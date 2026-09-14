"""Install a source-pinned, candidate-preserving eager NoPE fallback.

Zero-padding adaptation follows the MIT kingjones30 recipe (notice in LICENSES).
Modified vLLM source retains its Apache-2.0 notices. No top-k entries are removed.
"""

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

HASHES = {
    "model_executor/layers/mla.py": "936b06c4671d52fce52ae85bb24b986db17855f32740bf55b226053fc385c4b4",
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py": "a0023f72125cb0d5599b5bf940c86be1f0c9985bd62b0919f243a8fda76f4449",
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Patch anchor is not unique; refusing source drift")
    return text.replace(old, new, 1)


def add_candidate_order(backend):
    backend = replace_once(
        backend,
        "        self.supports_quant_query_input = False\n",
        "        from glm53_setup.runtime.candidate_order import candidate_order_enabled\n"
        "        self._glm53_canonical_candidates = (\n"
        "            model_type == 'glm5_next_text' and candidate_order_enabled()\n"
        "        )\n"
        "        self.supports_quant_query_input = False\n",
    )
    return replace_once(
        backend,
        "        topk_indices = self.topk_indices_buffer[:num_actual_toks]\n",
        "        topk_indices = self.topk_indices_buffer[:num_actual_toks]\n"
        "        if self._glm53_canonical_candidates:\n"
        "            from glm53_setup.runtime.candidate_order import canonical_logical_candidates\n"
        "            topk_indices = canonical_logical_candidates(topk_indices)\n",
    )


def prepare(package):
    originals = {name: (package / name).read_bytes() for name in HASHES}
    for name, data in originals.items():
        if hashlib.sha256(data).hexdigest() != HASHES[name]:
            raise ValueError(f"Source hash mismatch: {name}")
    mla, backend = (data.decode() for data in originals.values())
    mla = replace_once(
        mla,
        "        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim\n",
        "        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim\n"
        "        import os\n"
        "        from vllm.config import get_current_vllm_config\n"
        "        _model = get_current_vllm_config().model_config\n"
        "        self._glm53_reference_pad = 64 if (\n"
        "            qk_rope_head_dim == 0 and _model is not None\n"
        "            and getattr(_model.hf_text_config, 'model_type', '') == 'glm5_next_text'\n"
        "            and os.environ.get('GLM53_REFERENCE_ATTENTION') == '1'\n"
        "        ) else 0\n",
    )
    mla = replace_once(
        mla,
        "            qk_nope_head_dim=self.qk_nope_head_dim,\n"
        "            qk_rope_head_dim=self.qk_rope_head_dim,",
        "            qk_nope_head_dim=self.qk_nope_head_dim,\n"
        "            qk_rope_head_dim=self.qk_rope_head_dim + self._glm53_reference_pad,",
    )
    mla = replace_once(
        mla,
        "        attn_out = self.mla_attn(\n",
        "        if self._glm53_reference_pad:\n"
        "            q = torch.nn.functional.pad(q, (0, self._glm53_reference_pad))\n"
        "            k_pe = q.new_zeros((k_pe.shape[0], 1, self._glm53_reference_pad))\n"
        "        attn_out = self.mla_attn(\n",
    )
    backend = replace_once(
        backend,
        "        self.kv_scale_format = _kv_scale_format_for_model(model_type)\n",
        "        self.kv_scale_format = _kv_scale_format_for_model(model_type)\n"
        "        import os\n"
        "        self._glm53_reference_nope = (\n"
        "            model_type == 'glm5_next_text'\n"
        "            and vllm_config.model_config.hf_text_config.qk_rope_head_dim == 0\n"
        "            and os.environ.get('GLM53_REFERENCE_ATTENTION') == '1'\n"
        "        )\n",
    )
    backend = replace_once(
        backend,
        "        output = q.new_empty(\n",
        "        if self._glm53_reference_nope:\n"
        "            from glm53_reference import sparse_nope_reference\n"
        "            return sparse_nope_reference(\n"
        "                q[..., :512], kv_c_and_k_pe_cache.view(torch.uint8),\n"
        "                topk_indices_physical, self.scale,\n"
        "            ), None\n"
        "        output = q.new_empty(\n",
    )
    backend = add_candidate_order(backend)
    changed = (
        "# Modified by GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ contributors.\n"
        "# Changes: GLM NoPE zero-padding, canonical logical candidates and reference attention.\n"
        "# Original vLLM notices below remain applicable; see distribution NOTICE.\n"
    )
    result = dict(
        zip(
            HASHES,
            [(changed + mla).encode(), (changed + backend).encode()],
            strict=True,
        )
    )
    for name, data in result.items():
        compile(data, name, "exec")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    package = args.package or Path(sysconfig.get_paths()["purelib"]) / "vllm"
    outputs = prepare(package)
    manifest = {
        name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()
    }
    if not args.check:
        for name, data in outputs.items():
            (package / name).write_bytes(data)
        (package.parent / "glm53_reference.py").write_bytes(
            Path(__file__).with_name("reference_attention.py").read_bytes()
        )
        (package.parent / "glm53-reference-patch.json").write_text(
            json.dumps(manifest, indent=2)
        )
    print(
        json.dumps(
            {
                "source_hashes": HASHES,
                "patched_hashes": manifest,
                "check_only": args.check,
            }
        )
    )


if __name__ == "__main__":
    main()
