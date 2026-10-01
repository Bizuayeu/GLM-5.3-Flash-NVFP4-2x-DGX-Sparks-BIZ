import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from glm53_setup.runtime import patch_load_clone, patch_tp_padding, tp_padding

P = patch_tp_padding

# vLLM 385dce36 transformers_utils/configs/glm5_next.py, cut to the fields the patch
# reads; the lines around the anchor are verbatim, and the vision config's tail
# shows the anchor is the text config's.
CONFIG = """class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"

    def __init__(
        self,
        hidden_size: int = 4096,
        head_dim: int | None = None,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        moe_intermediate_size: int = 2048,
        linear_num_heads: int = 64,
        tie_word_embeddings: bool = False,
        swiglu_limit: float | None = None,
        logit_scale: float = 1.0,
        **kwargs,
    ):
        linear_cfg = kwargs.get("linear_attn_config") or {}
        if linear_cfg:
            linear_num_heads = linear_cfg.get("num_heads", linear_num_heads)

        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.moe_intermediate_size = moe_intermediate_size
        self.linear_num_heads = linear_num_heads

        self.swiglu_limit = swiglu_limit
        self.logit_scale = logit_scale

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Glm5NextVisionConfig(PretrainedConfig):
    def __init__(self, swiglu_limit: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.swiglu_limit = swiglu_limit
"""

# vLLM 385dce36 model_executor/parameter.py: the three narrows of the column and
# row loaders, and the qkv loader the patch leaves alone.
PARAMETER = """class _ColumnvLLMParameter(BasevLLMParameter):
    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.output_dim]
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        loaded_weight = loaded_weight.narrow(
            self.output_dim, shard_id_int * shard_size, shard_size
        )


class RowvLLMParameter(BasevLLMParameter):
    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.input_dim]
        loaded_weight = loaded_weight.narrow(
            self.input_dim, self.tp_rank * shard_size, shard_size
        )
"""

# vLLM 385dce36 model_executor/layers/vocab_parallel_embedding.py around the anchor.
VOCAB = """class VocabParallelEmbedding(PluggableLayer):
    def __init__(self, num_embeddings, org_num_embeddings=None, padding_size=64):
        tp_rank = 0
        self.tp_rank = tp_rank
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
"""

# vLLM 385dce36 model_executor/layers/fused_moe/routed_experts.py: _load_w13's and
# _load_w2's narrows, verbatim.
MOE = """class RoutedExperts(PluggableLayer):
    def _load_w13(self, expert_data, shard_dim, shard_id, loaded_weight, tp_rank, load_full=False):
        if self.moe_config.is_act_and_mul:
            shard_size = expert_data.shape[shard_dim] // 2
        else:
            shard_size = expert_data.shape[shard_dim]
        # Only narrow if the loaded_weight is not a scalar (0-dim tensor)
        # and we're not loading the full weight
        if not load_full and loaded_weight.ndim > 0:
            # When the parameter has been padded (e.g. MXFP4 rounding up
            # intermediate_size_per_partition), shard_size is the padded
            # size.  Compute the offset into the checkpoint weight using
            # the *unpadded* per-rank size so that every TP rank lands at
            # the correct slice.
            tp_size = self.moe_config.moe_parallel_config.tp_size
            loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
            start_offset = loaded_per_rank * tp_rank
            available = loaded_weight.shape[shard_dim] - start_offset
            if available <= 0:
                # If there is no available weight to load for this TP rank
                # (can happen on last TP rank with padding), we can skip
                # loading and return early
                return
            narrow_size = min(loaded_per_rank, available)
            loaded_weight = loaded_weight.narrow(shard_dim, start_offset, narrow_size)

    def _load_w2(self, expert_data, shard_dim, loaded_weight, tp_rank):
        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Only narrow if the loaded_weight is not a scalar (0-dim tensor).
        if loaded_weight.ndim > 0:
            # Same padding fix as _load_w13: use unpadded per-rank size.
            tp_size = self.moe_config.moe_parallel_config.tp_size
            loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
            start_offset = loaded_per_rank * tp_rank
            available = loaded_weight.shape[shard_dim] - start_offset
            if available <= 0:
                # If there is no available weight to load for this TP rank
                # (can happen on last TP rank with padding), we can skip
                # loading and return early
                return
            narrow_size = min(loaded_per_rank, available)
            loaded_weight = loaded_weight.narrow(shard_dim, start_offset, narrow_size)
"""

# vLLM 385dce36 model_executor/model_loader/weight_utils.py: the lazy safetensors
# read patch_load_clone rewrites, and sharded_weight_loader verbatim.
PINNED_LOADER = (
    """def safetensors_weights_iterator(hf_weights_files):
    for st_file in hf_weights_files:
        if safetensors_load_strategy != "torchao":
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
"""
    + patch_load_clone.LAZY
    + """

def sharded_weight_loader(shard_axis: int) -> LoaderFunction:
    \"\"\"Create a weight loader that shards the weights along the given axis\"\"\"

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)

        return default_weight_loader(param, loaded_weight)

    return loader
"""
)
LOADER = patch_load_clone.patch_text(PINNED_LOADER)

SOURCES = {
    P.CONFIG: CONFIG,
    P.PARAMETER: PARAMETER,
    P.VOCAB: VOCAB,
    P.MOE: MOE,
    P.LOADER: LOADER,
}


class Stub:
    """PretrainedConfig's part the config reads: keyword arguments become attributes."""

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


def config_class(multiple):
    namespace = {"PretrainedConfig": Stub}
    exec(P.PATCHES[P.CONFIG](CONFIG), namespace)
    env = {} if multiple is None else {tp_padding.ENV: str(multiple)}
    return namespace["Glm5NextTextConfig"], env


class ConfigPatchTests(unittest.TestCase):
    CHECKPOINT = dict(
        head_dim=0,
        linear_attn_config={"head_dim": 128, "num_heads": 64},
    )

    def read(self, multiple, **fields):
        cls, env = config_class(multiple)
        with mock.patch.dict(os.environ, env, clear=False):
            if multiple is None:
                os.environ.pop(tp_padding.ENV, None)
            return cls(**dict(self.CHECKPOINT, **fields))

    def test_unset_leaves_the_checkpoint_values(self):
        config = self.read(None)
        self.assertEqual(
            (config.num_attention_heads, config.linear_num_heads),
            (64, 64),
        )
        self.assertEqual(config.moe_intermediate_size, 2048)
        self.assertEqual(config.linear_attn_config["num_heads"], 64)

    def test_three_pads_the_heads_the_moe_width_and_the_linear_dict(self):
        config = self.read(3)
        self.assertEqual(
            (
                config.num_attention_heads,
                config.num_key_value_heads,
                config.linear_num_heads,
                config.moe_intermediate_size,
                config.linear_attn_config["num_heads"],
            ),
            (66, 66, 66, 2112, 66),
        )
        self.assertEqual(config.head_dim, 0)

    def test_a_derived_head_dim_comes_from_the_unpadded_heads(self):
        self.assertEqual(self.read(3, head_dim=None).head_dim, 4096 // 64)

    def test_a_round_trip_through_the_attributes_stays_padded(self):
        config = self.read(3)
        again = self.read(3, **vars(config))
        self.assertEqual(vars(again), vars(config))

    def test_the_vision_config_is_not_touched(self):
        patched = P.PATCHES[P.CONFIG](CONFIG)
        vision = patched[patched.index("class Glm5NextVisionConfig") :]
        self.assertEqual(vision, CONFIG[CONFIG.index("class Glm5NextVisionConfig") :])


class LoaderPatchTests(unittest.TestCase):
    def test_every_pinned_narrow_goes_through_pad_then_narrow(self):
        parameter = P.PATCHES[P.PARAMETER](PARAMETER)
        self.assertEqual(parameter.count("pad_then_narrow(\n"), 3)
        self.assertNotIn(
            "loaded_weight.narrow(\n            self.output_dim, self.tp", parameter
        )
        self.assertNotIn("loaded_weight.narrow(\n            self.input_dim", parameter)
        # The qkv loader is not on GLM's path and keeps the pinned narrow.
        self.assertIn(
            "self.output_dim, shard_id_int * shard_size, shard_size", parameter
        )
        loader = P.PATCHES[P.LOADER](LOADER)
        self.assertIn(
            "loaded_weight = pad_then_narrow(loaded_weight, shard_axis, start_idx, shard_size)",
            loader,
        )

    def test_the_moe_loaders_keep_the_pinned_branch_when_unset(self):
        patched = P.PATCHES[P.MOE](MOE)
        self.assertEqual(patched.count("if pad_multiple() > 1 and "), 2)
        # The pinned narrowing is kept whole behind the new branch.
        self.assertIn(
            "        elif not load_full and loaded_weight.ndim > 0:\n"
            "            # When the parameter has been padded (e.g. MXFP4 rounding up\n",
            patched,
        )
        self.assertIn(
            "        elif loaded_weight.ndim > 0:\n"
            "            # Same padding fix as _load_w13: use unpadded per-rank size.\n",
            patched,
        )
        self.assertEqual(patched.count("loaded_per_rank * tp_rank"), 2)
        # Each rank takes its contiguous share of the padded width, the whole shard.
        self.assertIn("shard_size * tp_rank, shard_size", patched)
        self.assertIn(
            "expert_data.shape[shard_dim] * tp_rank,\n"
            "                expert_data.shape[shard_dim],",
            patched,
        )

    def test_the_vocabulary_padding_goes_through_the_multiple(self):
        patched = P.PATCHES[P.VOCAB](VOCAB)
        self.assertIn(
            "        self.padding_size = vocab_padding_size(padding_size)\n", patched
        )


class PatchTextTests(unittest.TestCase):
    def test_every_file_is_marked_compiles_and_refuses_a_second_application(self):
        for name, source in SOURCES.items():
            with self.subTest(file=name):
                patched = P.PATCHES[name](source)
                self.assertIn("# Modified by GLM setup: zero-padding", patched)
                compile(patched, name, "exec")
                with self.assertRaises(ValueError):
                    P.PATCHES[name](patched)

    def test_every_file_refuses_a_drifted_anchor(self):
        drift = {
            P.CONFIG: ("self.logit_scale = logit_scale", "self.logit_scale = 1.0"),
            P.PARAMETER: ("self.input_dim, self.tp_rank", "self.input_dim, rank"),
            P.VOCAB: (
                "self.padding_size = padding_size",
                "self.padding = padding_size",
            ),
            P.MOE: ("if loaded_weight.ndim > 0:", "if loaded_weight.dim() > 0:"),
            P.LOADER: ("start_idx, shard_size)", "start_idx, size)"),
        }
        for name, (old, new) in drift.items():
            with self.subTest(file=name), self.assertRaises(ValueError):
                P.PATCHES[name](SOURCES[name].replace(old, new))

    def test_the_loader_needs_the_load_clone_patch_first(self):
        with self.assertRaisesRegex(ValueError, "load clone"):
            P.PATCHES[P.LOADER](PINNED_LOADER)


class PrepareTests(unittest.TestCase):
    def test_every_pinned_hash_gates_the_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "vllm"
            for name, source in SOURCES.items():
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                # Bytes, not write_text: on Windows that writes CRLF and the anchor misses.
                path.write_bytes(source.encode("utf-8"))
            with self.assertRaisesRegex(ValueError, "tp padding source hash mismatch"):
                P.prepare(package)
            hashes = {
                name: hashlib.sha256(source.encode("utf-8")).hexdigest()
                for name, source in SOURCES.items()
            }
            with mock.patch.dict(P.SOURCES, hashes):
                patched = P.prepare(package)
            self.assertEqual(set(patched), set(SOURCES))
            self.assertIn(b"vocab_padding_size(padding_size)", patched[P.VOCAB])

    def test_the_loader_pin_is_the_load_clone_output(self):
        self.assertEqual(
            P.SOURCES[P.LOADER],
            "facca7b7b26b3e9a18b6a9dc4122ba5c63c989c79e2e2100bff2d3286e2d0d49",
        )
        self.assertNotEqual(P.SOURCES[P.LOADER], patch_load_clone.SOURCE_SHA256)


if __name__ == "__main__":
    unittest.main()
