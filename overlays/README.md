# vLLM source overlays

The pinned vLLM's two GLM-5.3-Flash model sources (revision `385dce36`, package
directory `glm5next/nvidia`), carrying the quantization overlay and the split KDA
input projection: the fused `in_proj_qkvbfg_a` is replaced by `q_proj`, `k_proj`,
`v_proj` and `in_proj_bfg_a`, because the W4A16 Marlin cost is the GEMM width
(P23 in [the optimization catalog](../docs/optimization-catalog.md)). A derived
checkpoint whose attention projections are repacked cannot be loaded by the image's
own sources; these mount over them.

| File | Replaces in the image | SHA-256 | Base SHA-256 | Marker |
| --- | --- | --- | --- | --- |
| `kda-quant-split.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py` | `144a835faa57c56f5257c05a5cf6742102a21e38a92d39c2a1e26cd4ada680fd` | `99cfc24678c0a99f2c813c20e3dcc30c8bfac4f55ec4299737689d5d51f472cd` | `kda-quant-overlay` |
| `mla-quant-split.py` | `/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py` | `a9b1ea5ce05983973bc79954147809a17e0e5aa0f96eae47c6011306f51b533d` | `48f60a65afab1afd737b282d109c6c11f97d881d3ec774442e174fab09c2f525` | `mla-quant-overlay` |

The base SHA-256 is that of the file each overlay was made against, in the pinned image.

The launcher mounts them read-only over their targets, one
`{ target, source, sha256, base_sha256, marker }` entry each under
`runtime.derived_checkpoint.overlays`. `server preflight` refuses the launch unless
every overlay file has the declared SHA-256 and contains its marker, and unless the
image's own target has the declared base SHA-256, so an overlay built for another
image cannot be mounted ([runtime.derived_checkpoint](../docs/server-configuration.md)).

They pair with the Hugging Face checkpoint
[Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16)
at the revision whose `quantization_config.producer.in_proj_layout` is
`"split-qkv-bfg"`. The earlier fused overlays belong to the revision without that key;
either half of a mismatched pair fails at load.

Place them: copy both files to a directory under `state/` on each host, keeping the
names. Give each `source` the absolute path of its copy on that host in the profile
TOML.

License: Apache-2.0, as the vLLM sources they are derived from.
