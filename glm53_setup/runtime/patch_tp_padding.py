"""Source-pinned patch: zero-padding for tensor-parallel sizes that do not divide the heads.

vLLM 385dce36 splits GLM-5.3-Flash's 64 heads, its 2,048-wide MoE intermediate and its
154,880-token vocabulary evenly across ranks; TP=3 divides none of them and stops at
config time. ``runtime.tp_padding`` holds the padded geometry; this patch installs it at
five sites, all read at load time, so the checkpoint stays as NVIDIA publishes it:

- the text config reports the padded heads and MoE width (``Glm5NextTextConfig``), which
  every consumer reads: the target model, the MTP draft that ``SpeculativeConfig`` builds
  from ``config.json`` without ``--hf-overrides``, and the KDA state shape;
- ``parameter.py``'s column, merged-column and row loaders (the BF16 KDA and MLA
  projections, KDA's fused ``in_proj_qkvbfg_a`` segments and conv weights, the shared
  experts) and ``sharded_weight_loader`` (``dt_bias``, ``A_log``) zero-extend a
  checkpoint tensor before they take a rank's shard;
- the FusedMoE loaders give each rank its contiguous share of the padded width, NVFP4
  packed weights (two values per byte along K) and FP8 group scales alike, and the BF16
  experts of the MTP draft;
- ``VocabParallelEmbedding`` pads to ``lcm(64, multiple)``.

``models/glm5next/nvidia/model.py`` and ``kda.py`` are not touched: the published option
replaces them whole and its preflight pins their image hashes. ``weight_utils.py`` is the
file ``patch_load_clone`` already rewrote, so its pinned hash is that patch's output and
this patch applies after it. With ``GLM53_TP_PAD_MULTIPLE`` unset every site runs the
pinned code: the config sets nothing, the loaders call the pinned ``narrow`` and the
MoE loaders take the pinned branch.
"""

# cc-defer: carries zero-padded heads (66/64), MoE width (2112/2048) and
# vocabulary (192-row alignment) on the pinned loaders, paid in compute and
# memory on every rank. Drop it (and the Dockerfile RUN) when a vLLM pin shards
# uneven heads, or when serving leaves tensor-parallel sizes that need it.

from .pinned_patch import main_files, prepare_files, replace_exactly, replace_once

CONFIG = "transformers_utils/configs/glm5_next.py"
PARAMETER = "model_executor/parameter.py"
VOCAB = "model_executor/layers/vocab_parallel_embedding.py"
MOE = "model_executor/layers/fused_moe/routed_experts.py"
LOADER = "model_executor/model_loader/weight_utils.py"
# LOADER as patch_load_clone leaves it; the others as pinned.
SOURCES = {
    CONFIG: "4ba87fc3f76c7cdf98949638fac140a8a27f370903d516c6ed706f9f1ce56b3c",
    PARAMETER: "ff6054fbd19ec932c548d562c6f4cc4506383b71cbe411fcfd554c8b1d87b510",
    VOCAB: "ccb91c6b42aff7db1ee3dab67e993dc9ae5ecb04816fc5acf83a413d30424dec",
    MOE: "5bf64e9680b3198ababc2d55d8f4e9c3dc4fea137da7f37c8aab5c225f222222",
    LOADER: "facca7b7b26b3e9a18b6a9dc4122ba5c63c989c79e2e2100bff2d3286e2d0d49",
}
RECORD = "glm53-tp-padding-patch.json"
MARK = "glm53_setup.runtime.tp_padding"
HEADER = (
    "# Modified by GLM setup: zero-padding for tensor-parallel sizes that do not divide\n"
    "# the heads (GLM53_TP_PAD_MULTIPLE; unchanged when unset). Original vLLM Apache-2.0\n"
    "# notices below remain applicable.\n"
)

CONFIG_TAIL = (
    "        self.swiglu_limit = swiglu_limit\n"
    "        self.logit_scale = logit_scale\n"
    "\n"
    "        super().__init__(\n"
)
PADDED_CONFIG_TAIL = (
    "        self.swiglu_limit = swiglu_limit\n"
    "        self.logit_scale = logit_scale\n"
    "\n"
    "        # GLM setup: the zero-padded view for TP sizes that do not divide the heads.\n"
    "        from glm53_setup.runtime.tp_padding import pad_text_config\n"
    "\n"
    "        pad_text_config(self, kwargs)\n"
    "\n"
    "        super().__init__(\n"
)

COLUMN = (
    "        loaded_weight = loaded_weight.narrow(\n"
    "            self.output_dim, self.tp_rank * shard_size, shard_size\n"
    "        )\n"
)
PADDED_COLUMN = (
    "        from glm53_setup.runtime.tp_padding import pad_then_narrow\n"
    "\n"
    "        loaded_weight = pad_then_narrow(\n"
    "            loaded_weight, self.output_dim, self.tp_rank * shard_size, shard_size\n"
    "        )\n"
)
ROW = (
    "        loaded_weight = loaded_weight.narrow(\n"
    "            self.input_dim, self.tp_rank * shard_size, shard_size\n"
    "        )\n"
)
PADDED_ROW = (
    "        from glm53_setup.runtime.tp_padding import pad_then_narrow\n"
    "\n"
    "        loaded_weight = pad_then_narrow(\n"
    "            loaded_weight, self.input_dim, self.tp_rank * shard_size, shard_size\n"
    "        )\n"
)

PADDING = "        self.padding_size = padding_size\n"
PADDED_PADDING = (
    "        from glm53_setup.runtime.tp_padding import vocab_padding_size\n"
    "\n"
    "        self.padding_size = vocab_padding_size(padding_size)\n"
)

W13 = "        if not load_full and loaded_weight.ndim > 0:\n"
PADDED_W13 = (
    "        from glm53_setup.runtime.tp_padding import pad_multiple, pad_then_narrow\n"
    "\n"
    "        if pad_multiple() > 1 and not load_full and loaded_weight.ndim > 0:\n"
    "            # GLM setup: each rank takes its contiguous share of the zero-padded\n"
    "            # width and fills the whole shard.\n"
    "            loaded_weight = pad_then_narrow(\n"
    "                loaded_weight, shard_dim, shard_size * tp_rank, shard_size\n"
    "            )\n"
    "        elif not load_full and loaded_weight.ndim > 0:\n"
)
W2 = (
    "        if loaded_weight.ndim > 0:\n"
    "            # Same padding fix as _load_w13: use unpadded per-rank size.\n"
)
PADDED_W2 = (
    "        from glm53_setup.runtime.tp_padding import pad_multiple, pad_then_narrow\n"
    "\n"
    "        if pad_multiple() > 1 and loaded_weight.ndim > 0:\n"
    "            # GLM setup: as in _load_w13.\n"
    "            loaded_weight = pad_then_narrow(\n"
    "                loaded_weight,\n"
    "                shard_dim,\n"
    "                expert_data.shape[shard_dim] * tp_rank,\n"
    "                expert_data.shape[shard_dim],\n"
    "            )\n"
    "        elif loaded_weight.ndim > 0:\n"
    "            # Same padding fix as _load_w13: use unpadded per-rank size.\n"
)

SHARDED = (
    "        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)\n"
)
PADDED_SHARDED = (
    "        from glm53_setup.runtime.tp_padding import pad_then_narrow\n"
    "\n"
    "        loaded_weight = pad_then_narrow(loaded_weight, shard_axis, start_idx, shard_size)\n"
)
# patch_load_clone's rewrite; its absence means the pristine pinned loader file.
CLONED = "param = f.get_tensor(name).clone()\n"


def refuse_applied(text):
    if MARK in text:
        raise ValueError("tp padding patch already applied")


def finish(name, text):
    patched = HEADER + text
    compile(patched, name, "exec")
    return patched


def patch_config(text):
    refuse_applied(text)
    return finish(CONFIG, replace_once(text, CONFIG_TAIL, PADDED_CONFIG_TAIL))


def patch_parameter(text):
    refuse_applied(text)
    patched = replace_exactly(text, COLUMN, PADDED_COLUMN, 2)
    return finish(PARAMETER, replace_once(patched, ROW, PADDED_ROW))


def patch_vocab(text):
    refuse_applied(text)
    return finish(VOCAB, replace_once(text, PADDING, PADDED_PADDING))


def patch_moe(text):
    refuse_applied(text)
    patched = replace_once(text, W13, PADDED_W13)
    return finish(MOE, replace_once(patched, W2, PADDED_W2))


def patch_loader(text):
    refuse_applied(text)
    if CLONED not in text:
        raise ValueError("tp padding patch needs the load clone patch first")
    return finish(LOADER, replace_once(text, SHARDED, PADDED_SHARDED))


PATCHES = {
    CONFIG: patch_config,
    PARAMETER: patch_parameter,
    VOCAB: patch_vocab,
    MOE: patch_moe,
    LOADER: patch_loader,
}


def prepare(package):
    """Check every file against its pinned hash before patching any."""
    return prepare_files(package, SOURCES, PATCHES, "tp padding source hash mismatch: ")


def main(argv=None):
    main_files(argv, doc=__doc__, sources=SOURCES, prepare=prepare, record=RECORD)


if __name__ == "__main__":
    main()
