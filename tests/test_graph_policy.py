import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup.runtime import graph_policy


class GraphPolicyTests(unittest.TestCase):
    def test_lpa_runs_only_under_eager_execution(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True),
            compilation_config=SimpleNamespace(
                mode=SimpleNamespace(name="NONE"),
                cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
            ),
        )
        self.assertTrue(graph_policy.lpa_execution_supported(config))
        config.model_config.enforce_eager = False
        # The former eager-prefill switches no longer admit decode Graphs.
        with patch.dict(
            os.environ,
            {"GLM53_LPA_GRAPH_PREFILL": "1", "GLM53_ASYNC_INDEX_CHECKS": "1"},
        ):
            self.assertFalse(graph_policy.lpa_execution_supported(config))
