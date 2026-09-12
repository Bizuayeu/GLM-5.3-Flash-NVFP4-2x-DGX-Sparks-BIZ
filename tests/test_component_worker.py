import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm53_setup.runtime.component_worker import ComponentWorker


class ComponentWorkerTests(unittest.TestCase):
    def test_async_checks_are_reversible_and_keep_unqualified_modes_blocked(self):
        worker = ComponentWorker()
        worker.rank = 0
        worker.vllm_config = SimpleNamespace(
            speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(enforce_eager=True),
        )
        with patch.dict(os.environ, {"GLM53_ASYNC_INDEX_CHECKS": "0"}):
            self.assertFalse(worker.index_checks_configure(True)["previous"])
            self.assertEqual(os.environ["GLM53_ASYNC_INDEX_CHECKS"], "1")
            self.assertTrue(worker.index_checks_configure(False)["previous"])
            for invalid in (1, "true"):
                with self.assertRaises(ValueError):
                    worker.index_checks_configure(invalid)
            worker.indexer_capture = object()
            with self.assertRaises(ValueError):
                worker.index_checks_configure(True)
            del worker.indexer_capture
            worker.vllm_config.speculative_config = object()
            with self.assertRaises(ValueError):
                worker.index_checks_configure(True)
            self.assertEqual(os.environ["GLM53_ASYNC_INDEX_CHECKS"], "0")

    def test_mode_is_reversible_and_rejects_switch_during_capture(self):
        worker = ComponentWorker()
        worker.rank = 0
        worker.vllm_config = SimpleNamespace(
            speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(enforce_eager=True),
        )
        with patch.dict(os.environ, {"GLM53_FUSED_UNPACK": "0"}):
            self.assertFalse(worker.unpack_configure(True)["previous"])
            self.assertTrue(worker.unpack_configure(False)["previous"])
            for invalid in ("true", 1):
                with self.assertRaises(ValueError):
                    worker.unpack_configure(invalid)
            worker.indexer_capture = object()
            with self.assertRaises(ValueError):
                worker.unpack_configure(True)
            self.assertEqual(os.environ["GLM53_FUSED_UNPACK"], "0")
