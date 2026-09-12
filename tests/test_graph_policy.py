import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup.runtime import graph_policy, patch_graph_prefill


class GraphPolicyTests(unittest.TestCase):
    def test_prefill_routing_is_explicit_and_includes_single_token_chunks(self):
        with patch.dict(os.environ, {"GLM53_LPA_GRAPH_PREFILL": "0"}):
            self.assertFalse(
                graph_policy.force_eager_prefill(SimpleNamespace(has_prefill=True))
            )
        with patch.dict(os.environ, {"GLM53_LPA_GRAPH_PREFILL": "1"}):
            self.assertTrue(
                graph_policy.force_eager_prefill(
                    SimpleNamespace(has_prefill=True, num_tokens=1)
                )
            )
            self.assertFalse(
                graph_policy.force_eager_prefill(SimpleNamespace(has_prefill=False))
            )
            self.assertFalse(graph_policy.force_eager_prefill(None))

    def test_lpa_graph_contract_rejects_compiled_or_graphed_prefill(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            compilation_config=SimpleNamespace(
                mode=SimpleNamespace(name="NONE"),
                cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
            ),
        )
        with patch.dict(
            os.environ,
            {"GLM53_LPA_GRAPH_PREFILL": "1", "GLM53_ASYNC_INDEX_CHECKS": "1"},
        ):
            self.assertTrue(graph_policy.lpa_execution_supported(config))
            config.compilation_config.mode.name = "VLLM_COMPILE"
            self.assertFalse(graph_policy.lpa_execution_supported(config))
            config.compilation_config.mode.name = "NONE"
            config.compilation_config.cudagraph_mode.name = "FULL_AND_PIECEWISE"
            self.assertFalse(graph_policy.lpa_execution_supported(config))

    def test_runner_patch_is_pinned_and_does_not_mutate_during_prepare(self):
        source = b"import torch\ndef f():\n    dispatch(need_eager=is_profile or skip_compiled,)\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / patch_graph_prefill.RUNNER
            path.parent.mkdir(parents=True)
            path.write_bytes(source)
            with patch.object(
                patch_graph_prefill, "SOURCE_SHA256", hashlib.sha256(source).hexdigest()
            ):
                result = patch_graph_prefill.prepare(root)
                self.assertIn(b"_glm53_force_eager_prefill(batch_req_state)", result)
                self.assertEqual(path.read_bytes(), source)
                path.write_bytes(source + b"# drift\n")
                with self.assertRaises(ValueError):
                    patch_graph_prefill.prepare(root)
