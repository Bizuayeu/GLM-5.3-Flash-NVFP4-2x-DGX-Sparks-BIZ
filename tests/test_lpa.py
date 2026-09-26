"""Contracts that keep experimental prefill changes out of decode/cache reuse."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup.runtime.lpa import (
    AttentionInputExperiment,
    ExperimentSpec,
    LPAWorkerExtension,
)


class ExperimentSpecTests(unittest.TestCase):
    def test_speculative_cursor_correction_cannot_cross_into_the_prompt(self):
        spec = ExperimentSpec("predict", 32, 4012, tail=512)
        positions = [4013, 4014, 4015, 4016]
        self.assertTrue(spec.validate_scheduled_start(positions, 4016, True))
        self.assertEqual(spec.approximate_count(positions), 0)
        self.assertFalse(spec.validate_scheduled_start([0, 1], 0, False))
        for actual, expected, speculative in (
            (4011, 4016, True),
            (4017, 4016, True),
            (4013, 4016, False),
            (16, 20, True),
        ):
            with (
                self.subTest(actual=actual, expected=expected),
                self.assertRaises(ValueError),
            ):
                spec.validate_scheduled_start([actual], expected, speculative)

    def test_resumed_exact_prefix_and_tail_surround_the_approximate_span(self):
        spec = ExperimentSpec("predict", 0, 12, tail=2, approximate_start=4)
        self.assertEqual(spec.approximate_span([0, 1, 2, 3]), (0, 0))
        self.assertEqual(spec.approximate_span([2, 3, 4, 5]), (2, 4))
        self.assertEqual(spec.approximate_span([8, 9, 10, 11]), (0, 2))
        self.assertEqual(spec.approximate_count([12, 13]), 0)

    def test_ordinary_path_does_not_copy_gpu_positions_without_diagnostics(self):
        experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
        experiment.spec = ExperimentSpec("off", 0, 12)
        experiment.verify_state = False
        self.assertIsNone(
            experiment._attention_hook(0)(
                None,
                (),
                {
                    "hidden_states": object(),
                    "positions": object(),
                },
            )
        )

    def test_mlp_preserves_both_exact_sides_after_prefix_eviction(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch environment required")
        experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
        experiment.torch = torch
        experiment.spec = ExperimentSpec("predict", 0, 12, tail=2, approximate_start=4)
        experiment.skip_mlp = True
        experiment.counts = {"mlp_skipped_tokens": {}}
        experiment.current_positions = list(range(2, 12))
        x = torch.arange(10).float().reshape(10, 1)
        result = experiment._mlp_forward(0, lambda v: v * 2)(x)
        expected = x * 2
        expected[2:8] = 0
        self.assertTrue(torch.equal(result, expected))

    def test_prefill_state_diagnostic_ignores_unused_mtp_conv_slots(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch environment required")
        for dim_first in (False, True):
            experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
            experiment.torch = torch
            experiment.spec = ExperimentSpec("capture", 0, 5)
            experiment.verify_state = True
            experiment.current_positions = [0, 1, 2, 3]
            experiment.state_reference = {}
            experiment.state_errors = []
            conv = torch.zeros((1, 8, 6) if dim_first else (1, 6, 8))
            module = SimpleNamespace(
                prefix="layer",
                kv_cache=(conv, torch.zeros(1, 2, 2)),
                _conv_state_dim_first=dim_first,
                conv_size=4,
            )
            metadata = SimpleNamespace(non_spec_state_indices_tensor=torch.tensor([0]))
            context = SimpleNamespace(attn_metadata={"layer": metadata})
            fake = SimpleNamespace(get_forward_context=lambda: context)
            with patch.dict(sys.modules, {"vllm.forward_context": fake}):
                hook = experiment._state_hook(0)
                hook(module, (), None)
                conv.narrow(2 if dim_first else 1, 3, 3).fill_(123)
                experiment.spec = ExperimentSpec("oracle", 0, 5)
                hook(module, (), None)
                self.assertTrue(
                    all(row["max_abs"] == 0 for row in experiment.state_errors)
                )
                conv.narrow(2 if dim_first else 1, 0, 3).fill_(1)
                hook(module, (), None)
                self.assertEqual(experiment.state_errors[-2]["max_abs"], 1)

    def test_mtp_verification_and_rollback_positions_are_never_approximated(self):
        spec = ExperimentSpec("predict", 32, 1024, tail=512)
        for positions in ([1023], [1024, 1025, 1026, 1027], [1025, 1026, 1027, 1028]):
            self.assertEqual(spec.approximate_count(positions), 0)

    def test_mtp_requires_explicit_opt_in_and_supported_method(self):
        worker = LPAWorkerExtension()
        worker.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(enforce_eager=True),
            speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=3),
        )
        worker.get_model = lambda: "target-only"
        with self.assertRaises(ValueError):
            worker.lpa_configure()
        with patch("glm53_setup.runtime.lpa.AttentionInputExperiment") as experiment:
            worker.lpa_configure(allow_mtp=True, mode="off")
            experiment.assert_called_once_with("target-only")
            experiment.return_value.configure.assert_called_once_with(mode="off")
        worker.vllm_config.speculative_config.method = "eagle"
        with self.assertRaises(ValueError):
            worker.lpa_configure(allow_mtp=True)

    def test_mtp_depths_one_to_three_are_accepted(self):
        for depth in (1, 2, 3, 4, 5):
            with self.subTest(depth=depth):
                worker = LPAWorkerExtension()
                worker.vllm_config = SimpleNamespace(
                    scheduler_config=SimpleNamespace(max_num_seqs=1),
                    cache_config=SimpleNamespace(enable_prefix_caching=False),
                    model_config=SimpleNamespace(enforce_eager=True),
                    speculative_config=SimpleNamespace(
                        method="mtp", num_speculative_tokens=depth
                    ),
                )
                worker.get_model = lambda: "target-only"
                with patch("glm53_setup.runtime.lpa.AttentionInputExperiment"):
                    if depth <= 3:
                        worker.lpa_configure(allow_mtp=True, mode="off")
                        continue
                    with self.assertRaises(ValueError):
                        worker.lpa_configure(allow_mtp=True, mode="off")

    def test_fully_protected_prompt_bypasses_loading_and_prediction(self):
        experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
        experiment.layers = [None, None]
        experiment.predictor_identity = None
        experiment.reference_mask = None
        result = experiment.configure(
            "predict", 0, 20, tail=20, predictor_path="not-present-projector.pt"
        )
        self.assertEqual(result["mode"], "off")
        self.assertEqual(result["requested_mode"], "predict")
        self.assertEqual(experiment.spec.mode, "off")

    def test_new_prompt_reuses_loaded_artifact_but_file_change_reloads_it(self):
        experiment = AttentionInputExperiment.__new__(AttentionInputExperiment)
        experiment.layers = [None, None]
        experiment.predictor_identity = None
        experiment.reference_mask = None
        stat = SimpleNamespace(st_mtime_ns=1, st_size=100)
        with (
            patch("pathlib.Path.stat", return_value=stat),
            patch.object(
                experiment, "_load_predictor", return_value={1: "cached"}
            ) as load,
        ):
            experiment.configure("predict", 0, 10, predictor_path="projector.pt")
            experiment.configure("predict", 0, 20, predictor_path="projector.pt")
            self.assertEqual(load.call_count, 1)
            stat.st_mtime_ns = 2
            experiment.configure("predict", 0, 30, predictor_path="projector.pt")
            self.assertEqual(load.call_count, 2)

    def test_last_prompt_token_and_decode_are_protected(self):
        spec = ExperimentSpec("oracle", 2, 9, tail=1)
        self.assertEqual(spec.approximate_count([0, 1, 2, 3]), 4)
        self.assertEqual(spec.approximate_count([7, 8]), 1)
        self.assertEqual(spec.approximate_count([9]), 0)

    def test_single_token_prompt_never_approximates(self):
        self.assertEqual(ExperimentSpec("oracle", 0, 1).approximate_count([0]), 0)

    def test_reordered_or_batched_positions_are_rejected(self):
        spec = ExperimentSpec("oracle", 2, 9)
        for positions in ([0, 1, 0], [2, 4], [1, 1], [-1]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                spec.approximate_count(positions)

    def test_invalid_spec_cannot_silently_enable_an_experiment(self):
        for kwargs in (
            {"mode": "unknown"},
            {"cut": -1},
            {"prompt_length": 0},
            {"tail": 0},
            {"tail": 10},
            {"cut": 2.5},
            {"prompt_length": True},
            {"mode": []},
        ):
            values = dict(mode="oracle", cut=2, prompt_length=9)
            values.update(kwargs)
            with self.assertRaises(ValueError):
                ExperimentSpec(**values)


def split_model(torch):
    """Four fake layers, KDA/KDA/KDA/MLA, with the pinned projection call sites."""
    nn = torch.nn
    functional = torch.nn.functional
    width, heads, head_dim, q_rank, kv_rank, index_dim = 8, 2, 3, 5, 4, 3

    class Linear(nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    class KDA(nn.Module):
        def __init__(self):
            super().__init__()
            self.local_num_heads, self.head_dim = heads, head_dim
            self.q_proj, self.k_proj, self.v_proj = (
                Linear(width, 6, bias=False) for _ in range(3)
            )
            self.in_proj_bfg_a = Linear(width, heads + 2 * head_dim, bias=False)
            self.o_proj = Linear(18 + heads + 2 * head_dim, width, bias=False)

        def forward(self, hidden_states, positions):
            q, k, v = (
                m(hidden_states)[0] for m in (self.q_proj, self.k_proj, self.v_proj)
            )
            b, f, g = self.in_proj_bfg_a(hidden_states)[0].split(
                [heads, head_dim, head_dim], dim=-1
            )
            self.seen = dict(x=hidden_states, q=q, k=k, v=v, b=b, f=f, g=g)
            return self.o_proj(torch.cat([q, k, v, b, f, g], dim=-1).tanh())[0]

    class IndexerOp(nn.Module):
        def forward(self, hidden_states, q, k, weights, gate_score=None):
            self.seen = dict(k=k, weights=weights, gate=gate_score)

    class Indexer(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = index_dim
            self.wq_b = Linear(q_rank, index_dim, bias=False)
            self.wk_weights_proj = Linear(width, index_dim + 2, bias=False)
            self.index_kpool_compress_gate = nn.Parameter(torch.randn(index_dim, width))
            self.indexer_op = IndexerOp()

        def forward(self, hidden_states, qr, positions, rotary_emb):
            k = self.wk_weights_proj(hidden_states)[0][:, : self.head_dim]
            weights = hidden_states.float() @ (
                self.wk_weights_proj.weight[self.head_dim :].t().float()
            )
            gate = functional.linear(hidden_states, self.index_kpool_compress_gate)
            self.indexer_op(
                hidden_states, self.wq_b(qr)[0], k, weights, gate_score=gate
            )

    class Wrapper(nn.Module):
        def __init__(self, fused, indexer):
            super().__init__()
            self.fused_qkv_a_proj, self.indexer = fused, indexer
            self.q_b_proj = Linear(q_rank, 6, bias=False)
            self.o_proj = Linear(6 + kv_rank, width, bias=False)

        def forward(self, positions, hidden_states):
            q_c, kv = self.fused_qkv_a_proj(hidden_states)[0].split(
                [q_rank, kv_rank], dim=-1
            )
            self.indexer(hidden_states, q_c, positions, None)
            self.seen = dict(x=hidden_states, q_c=q_c, kv=kv)
            q = self.q_b_proj(q_c)[0]
            return self.o_proj(torch.cat([q, kv], dim=-1).tanh())[0]

    class MLA(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lora_rank = q_rank
            self.fused_qkv_a_proj = Linear(width, q_rank + kv_rank, bias=False)
            self.indexer = Indexer()
            self.mla_attn = Wrapper(self.fused_qkv_a_proj, self.indexer)

        def forward(self, hidden_states, positions):
            return self.mla_attn(positions, hidden_states)

    class Glm5NextDecoderLayer(nn.Module):
        def __init__(self, index, kind):
            super().__init__()
            self.layer_idx, self.layer_kind, self.hidden_size = index, kind, width
            self.is_mtp_layer = self.is_sequence_parallel = False
            self.self_attn = KDA() if kind == "kda" else MLA()
            self.mlp = Linear(width, width, bias=False)

        def forward(self, x, positions):
            x = x + self.self_attn(hidden_states=x.tanh(), positions=positions)
            return x + self.mlp(x)[0].tanh()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                Glm5NextDecoderLayer(i, k)
                for i, k in enumerate(("kda", "kda", "kda", "mla"))
            )

        def forward(self, x, positions):
            for layer in self.layers:
                x = layer(x, positions)
            return x

    return Model()


class SplitModeTests(unittest.TestCase):
    """Late layers write state from p and read with their own input x."""

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch environment required")
        self.torch = torch
        torch.manual_seed(0)
        self.model = split_model(torch)
        self.experiment = AttentionInputExperiment(self.model)
        self.x = torch.randn(5, 8)
        self.positions = torch.arange(5)
        width = 8
        # p = scale * source + bias; a constant row keeps its origin visible.
        self.weights = {
            i: {
                "mean": torch.zeros(width),
                "down": torch.zeros(width, 1),
                "up": torch.zeros(1, width),
                "scale": torch.zeros(width),
                "bias": torch.linspace(-0.5, 0.5, width),
            }
            for i in (2, 3)
        }

    def run_mode(self, mode, **kwargs):
        with (
            patch(
                "pathlib.Path.stat",
                return_value=SimpleNamespace(st_mtime_ns=1, st_size=1),
            ),
            patch.object(self.experiment, "_load_predictor", return_value=self.weights),
        ):
            result = self.experiment.configure(
                mode, 1, 5, predictor_path="projector.pt", **kwargs
            )
        with self.torch.no_grad():
            output = self.model(self.x, self.positions)
        seen = [
            dict(
                getattr(layer.self_attn, "seen", {}),
                **getattr(getattr(layer.self_attn, "mla_attn", None), "seen", {}),
            )
            for layer in self.model.layers
        ]
        index = self.model.layers[3].self_attn.indexer.indexer_op.seen
        return result, output, seen, index

    def test_split_self_is_the_native_forward(self):
        _, off, off_seen, off_index = self.run_mode("off")
        result, split, split_seen, split_index = self.run_mode("split-self")
        self.assertEqual(result["mode"], "split-self")
        self.assertTrue(self.torch.equal(off, split))
        for before, after in zip(off_seen, split_seen):
            for key in before:
                self.assertTrue(self.torch.equal(before[key], after[key]), key)
        for key in off_index:
            self.assertTrue(self.torch.equal(off_index[key], split_index[key]), key)

    def test_split_writes_from_the_projection_and_reads_from_the_layer_input(self):
        torch = self.torch
        _, off, off_seen, _ = self.run_mode("off")
        _, split, seen, index = self.run_mode("split")
        self.assertFalse(torch.equal(off, split))
        # The cut layer and every layer before it are unchanged.
        for i in (0, 1):
            for key in off_seen[i]:
                self.assertTrue(torch.equal(off_seen[i][key], seen[i][key]), key)
        self.assertEqual(self.experiment.counts["split_tokens"], {2: 5, 3: 5})
        # Outside a split forward the wrapped projections are the native ones.
        self.experiment.configure("off", 1, 5)
        p = self.weights[2]["bias"].expand(5, 8)
        attn = self.model.layers[2].self_attn
        x = seen[2]["x"]
        self.assertTrue(torch.equal(seen[2]["q"], attn.q_proj(x)[0]))
        self.assertTrue(torch.equal(seen[2]["k"], attn.k_proj(p)[0]))
        self.assertTrue(torch.equal(seen[2]["v"], attn.v_proj(p)[0]))
        from_p = attn.in_proj_bfg_a(p)[0]
        self.assertTrue(torch.equal(seen[2]["b"], from_p[:, :2]))
        self.assertTrue(torch.equal(seen[2]["f"], from_p[:, 2:5]))
        self.assertTrue(torch.equal(seen[2]["g"], attn.in_proj_bfg_a(x)[0][:, 5:]))
        attn = self.model.layers[3].self_attn
        p = self.weights[3]["bias"].expand(5, 8)
        x = seen[3]["x"]
        self.assertTrue(torch.equal(seen[3]["q_c"], attn.fused_qkv_a_proj(x)[0][:, :5]))
        self.assertTrue(torch.equal(seen[3]["kv"], attn.fused_qkv_a_proj(p)[0][:, 5:]))
        indexer = attn.indexer
        wk = indexer.wk_weights_proj
        self.assertTrue(torch.equal(index["k"], wk(p)[0][:, :3]))
        self.assertTrue(
            torch.equal(index["weights"], x.float() @ wk.weight[3:].t().float())
        )
        self.assertTrue(
            torch.equal(
                index["gate"],
                torch.nn.functional.linear(p, indexer.index_kpool_compress_gate),
            )
        )

    def test_projection_inputs_are_released_after_each_attention(self):
        self.run_mode("split")
        self.assertEqual(self.experiment.split_inputs, {})
        # Off again: every wrapped projection is the native call.
        _, off, _, _ = self.run_mode("off")
        _, again, _, _ = self.run_mode("off")
        self.assertTrue(self.torch.equal(off, again))

    def test_split_refuses_what_it_does_not_cover(self):
        for kwargs in ({"skip_mla_queries": True}, {"tail": 2}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.run_mode("split", **kwargs)
        with self.assertRaises(ValueError):
            ExperimentSpec("split-self", 1, 5, approximate_start=1)
        with self.assertRaises(ValueError):
            self.experiment.configure("split", 1, 5)  # no projector

    def test_split_refuses_mtp_and_prefix_caching(self):
        for speculative, caching in (
            (SimpleNamespace(method="mtp", num_speculative_tokens=1), False),
            (None, True),
        ):
            worker = LPAWorkerExtension()
            worker.vllm_config = SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_seqs=1),
                cache_config=SimpleNamespace(enable_prefix_caching=caching),
                model_config=SimpleNamespace(enforce_eager=True),
                speculative_config=speculative,
            )
            worker.get_model = lambda: "target-only"
            for mode in ("split", "split-self"):
                with (
                    self.subTest(mode=mode, caching=caching),
                    patch("glm53_setup.runtime.lpa.AttentionInputExperiment"),
                    self.assertRaises(ValueError),
                ):
                    worker.lpa_configure(allow_mtp=True, mode=mode, cut=1)


if __name__ == "__main__":
    unittest.main()
