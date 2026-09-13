"""Real pinned vLLM allocation/hash-table contracts; no model weights needed."""

import hashlib
import importlib.util
import json
import os
import unittest
from unittest.mock import patch


def digest(value):
    return hashlib.sha256(repr(value).encode()).digest()


@unittest.skipUnless(importlib.util.find_spec("vllm"), "Pinned vLLM image required")
class RealCacheManagerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
        )

        self.assertTrue(hasattr(KVCacheManager.allocate_slots, "__wrapped__"))
        init_none_hash(digest)
        self.hasher = get_request_block_hasher(4, digest)
        spec = FullAttentionSpec(
            block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        config = KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)
            ],
        )
        self.manager = KVCacheManager(
            config,
            max_model_len=32,
            scheduler_block_size=4,
            hash_block_size=4,
            max_in_flight_tokens=32,
        )
        runtime = json.dumps(
            {
                "cut": 0,
                "tail": 2,
                "break_even": 4,
                "projector_path": "/unused.pt",
                "projector_sha256": "0" * 64,
                "skip_mla_queries": True,
            }
        )
        env = patch.dict(os.environ, {"GLM53_APC_LPA_CONFIG": runtime})
        env.start()
        self.addCleanup(env.stop)

    def request(self, name, length, mode="auto"):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        return Request(
            name,
            list(range(length)),
            SamplingParams(max_tokens=3, extra_args={"glm53_lpa_mode": mode}),
            None,
            block_hasher=self.hasher,
        )

    def prime(self):
        req = self.request("prime", 9, "off")
        self.assertIsNotNone(self.manager.allocate_slots(req, 9))
        self.manager.free(req)

    def test_private_suffix_tail_and_decode_never_enter_shared_hash_table(self):
        self.prime()
        req = self.request("approximate", 21)
        blocks, hit, _ = self.manager.get_computed_blocks(req)
        self.assertEqual(hit, 8)
        self.assertIsNotNone(
            self.manager.allocate_slots(
                req, 13, num_new_computed_tokens=hit, new_computed_blocks=blocks
            )
        )
        allocated = self.manager.get_blocks(req.request_id).blocks[0]
        self.assertEqual(len(allocated), 6)
        self.assertTrue(all(b.block_hash is None for b in allocated[2:]))
        req.num_computed_tokens = 21
        req.append_output_token_ids([100, 101, 102])
        self.assertIsNotNone(self.manager.allocate_slots(req, 3))
        self.manager.cache_blocks(req, 24)
        self.assertTrue(all(b.block_hash is None for b in allocated[2:]))
        self.manager.free(req)

        normal = self.request("normal", 21, "off")
        blocks, hit, _ = self.manager.get_computed_blocks(normal)
        self.assertEqual(hit, 8)
        self.assertIsNotNone(
            self.manager.allocate_slots(
                normal, 13, num_new_computed_tokens=hit, new_computed_blocks=blocks
            )
        )
        self.manager.free(normal)
        later = self.request("later", 21, "off")
        self.assertEqual(self.manager.get_computed_blocks(later)[1], 20)

    def test_native_publication_is_a_negative_control_for_the_leak_check(self):
        with patch.dict(os.environ, {"GLM53_APC_LPA_CONFIG": ""}):
            self.prime()
            req = self.request("unguarded", 21)
            blocks, hit, _ = self.manager.get_computed_blocks(req)
            self.assertEqual(hit, 8)
            self.manager.allocate_slots(
                req, 13, num_new_computed_tokens=hit, new_computed_blocks=blocks
            )
            self.manager.free(req)
            self.assertEqual(
                self.manager.get_computed_blocks(self.request("later", 21))[1], 20
            )


if __name__ == "__main__":
    unittest.main()
