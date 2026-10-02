import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from glm53_setup.runtime.apc_runtime import POLICY_KEY, allocation_guard, settings
from glm53_setup.runtime.apc_worker import (
    _read_policy,
    _verify_projector,
    before_forward,
    report,
    validate_worker,
    warmup_scope,
)
from glm53_setup.runtime.lpa import LPAWorkerExtension

CONFIG = json.dumps(
    {
        "cut": 0,
        "tail": 4,
        "break_even": 8,
        "projector_path": "/lpa/projector.pt",
        "projector_sha256": "0" * 64,
        "skip_mla_queries": True,
    }
)


def data(name, hit, mode="auto"):
    request = NS(
        request_id=name,
        num_prompt_tokens=64,
        num_computed_tokens=0,
        sampling_params=NS(extra_args={"glm53_lpa_mode": mode}),
        prompt_embeds=None,
        mm_features=[],
        lora_request=None,
    )

    @allocation_guard
    def allocate(manager, request, new, **kwargs):
        return []

    allocate(NS(enable_caching=True), request, 64 - hit, num_new_computed_tokens=hit)
    # Exercise a primitive-only wire copy rather than sharing the state object.
    extra = json.loads(json.dumps(request.sampling_params.extra_args))
    return NS(
        req_id=name,
        prompt_token_ids=list(range(64)),
        num_computed_tokens=hit,
        sampling_params=NS(extra_args=extra),
    )


def step(new=(), active=None, cached=(), finished=()):
    return NS(
        scheduled_new_reqs=list(new),
        num_scheduled_tokens=active or {},
        scheduled_cached_reqs=NS(
            req_ids=[x[0] for x in cached],
            num_computed_tokens=[x[1] for x in cached],
            num_output_tokens=[max(0, x[1] - 64 + 1) for x in cached],
        ),
        finished_req_ids=set(finished),
    )


class APCWorkerTests(unittest.TestCase):
    def test_trusted_startup_warmup_does_not_weaken_real_request_checks(self):
        with self.assertRaises(RuntimeError):
            with warmup_scope(self.worker):
                before_forward(self.worker, object())
                raise RuntimeError("warmup failure")
        with self.assertRaises(ValueError):
            before_forward(
                self.worker, step(active={"unmanaged": 1}, cached=[("unmanaged", 0)])
            )
        before_forward(self.worker, step([data("a", 63)], {"a": 1}))
        with self.assertRaises(ValueError):
            with warmup_scope(self.worker):
                pass

    def setUp(self):
        environment = patch.dict(os.environ, {"GLM53_APC_LPA_CONFIG": CONFIG})
        environment.start()
        self.addCleanup(environment.stop)
        self.worker = LPAWorkerExtension()
        self.worker.rank = 0
        self.worker.get_model = Mock(return_value="target")
        self.worker.vllm_config = NS(
            model_config=NS(enforce_eager=True),
            scheduler_config=NS(max_num_seqs=1),
            parallel_config=NS(
                pipeline_parallel_size=1,
                tensor_parallel_size=2,
                data_parallel_size=1,
                enable_expert_parallel=False,
            ),
            cache_config=NS(
                enable_prefix_caching=True,
                cache_dtype="fp8_ds_mla",
                mamba_cache_mode="align",
                enable_mamba_fine_grained_prefix_cache=False,
                prefix_match_unit=None,
            ),
            kv_transfer_config=None,
            ec_transfer_config=None,
            speculative_config=None,
        )

    def test_exact_request_never_loads_or_constructs_a_projector(self):
        with patch("glm53_setup.runtime.apc_worker._verify_projector") as verify:
            before_forward(self.worker, step([data("a", 63)], {"a": 1}))
        verify.assert_not_called()
        self.worker.get_model.assert_not_called()
        self.assertIsNone(report(self.worker)["lpa"])

    def test_unresolved_or_unsupported_cache_format_is_rejected(self):
        for dtype in ("fp8", "auto", "bfloat16"):
            with self.subTest(dtype=dtype):
                self.worker.vllm_config.cache_config.cache_dtype = dtype
                with self.assertRaises(ValueError):
                    before_forward(self.worker, step([data("a", 63)], {"a": 1}))

    def test_policy_is_applied_once_and_cleared_for_the_next_exact_request(self):
        experiment = Mock()
        with (
            patch("glm53_setup.runtime.apc_worker._verify_projector"),
            patch(
                "glm53_setup.runtime.lpa.AttentionInputExperiment",
                return_value=experiment,
            ),
        ):
            before_forward(self.worker, step([data("a", 16)], {"a": 8}))
            self.assertEqual(
                experiment.configure.call_args.kwargs["approximate_start"], 16
            )
            self.assertEqual(experiment.configure.call_args.kwargs["mode"], "predict")
            before_forward(self.worker, step(active={"a": 8}, cached=[("a", 24)]))
            self.assertEqual(experiment.configure.call_count, 1)
            self.assertEqual(experiment.expected_position, 24)
            before_forward(
                self.worker, step([data("b", 0, "off")], {"b": 8}, finished=["a"])
            )
            self.assertEqual(experiment.configure.call_args.kwargs["mode"], "off")
            self.assertNotIn("a", self.worker._glm53_apc_requests)
        with self.assertRaises(ValueError):
            self.worker.lpa_configure(mode="predict")

    def test_boundary_and_policy_corruption_are_rejected(self):
        value = data("a", 16)
        value.num_computed_tokens = 20
        with self.assertRaises(ValueError):
            before_forward(self.worker, step([value], {"a": 8}))
        value = data("b", 16)
        value.sampling_params.extra_args[POLICY_KEY]["policy"]["tail"] = 5
        with self.assertRaises(ValueError):
            before_forward(self.worker, step([value], {"b": 8}))

    def test_multiple_scheduled_requests_are_rejected_before_forward(self):
        with self.assertRaises(ValueError):
            before_forward(
                self.worker, step([data("a", 16), data("b", 16)], {"a": 8, "b": 8})
            )
        self.worker.get_model.assert_not_called()

    def test_mtp_depths_one_to_three_are_the_candidates(self):
        for depth in (1, 2, 3, 4, 5):
            with self.subTest(depth=depth):
                self.worker.vllm_config.speculative_config = NS(
                    method="mtp", num_speculative_tokens=depth
                )
                if depth <= 3:
                    validate_worker(self.worker.vllm_config)
                    continue
                with self.assertRaises(ValueError):
                    validate_worker(self.worker.vllm_config)

    def test_only_mtp_decode_uses_the_optimistic_cursor_contract(self):
        self.worker.vllm_config.speculative_config = NS(
            method="mtp", num_speculative_tokens=3
        )
        experiment = Mock()
        with (
            patch("glm53_setup.runtime.apc_worker._verify_projector"),
            patch(
                "glm53_setup.runtime.lpa.AttentionInputExperiment",
                return_value=experiment,
            ),
        ):
            before_forward(self.worker, step([data("a", 16)], {"a": 8}))
            self.assertFalse(experiment.speculative_decode)
            before_forward(self.worker, step(active={"a": 4}, cached=[("a", 68)]))
            self.assertTrue(experiment.speculative_decode)
            self.assertEqual(experiment.expected_position, 68)
            before_forward(self.worker, step(active={"a": 4}, cached=[("a", 60)]))
            self.assertFalse(experiment.speculative_decode)

    def test_wire_shape_and_identity_mismatches_are_rejected(self):
        config = settings()
        self.assertEqual(_read_policy(data("a", 16), config).request_id, "a")
        shape = "did not receive the cache manager's LPA policy"
        identity = "^Worker policy identity mismatch$"
        cases = (
            (shape, lambda wire: wire.pop("override")),
            (shape, lambda wire: wire.update(extra=1)),
            (identity, lambda wire: wire.update(version=True)),
            (identity, lambda wire: wire.update(version="1")),
            (identity, lambda wire: wire.update(version=2)),
            (identity, lambda wire: wire.update(request_id="b")),
            (identity, lambda wire: wire.update(signature="0" * 64)),
            (identity, lambda wire: wire.update(override="predict")),
        )
        for message, tamper in cases:
            value = data("a", 16)
            tamper(value.sampling_params.extra_args[POLICY_KEY])
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                _read_policy(value, config)
        for extra in ({POLICY_KEY: "forged"}, {}, None):
            value = data("a", 16)
            value.sampling_params.extra_args = extra
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, shape):
                _read_policy(value, config)
        value = data("a", 16)
        value.sampling_params = None
        with self.assertRaisesRegex(ValueError, shape):
            _read_policy(value, config)


class VerifyProjectorTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "projector.pt"
        self.path.write_bytes(b"projector-v1")
        self.config = NS(
            projector_path=str(self.path),
            projector_sha256=hashlib.sha256(b"projector-v1").hexdigest(),
            signature="sig",
        )
        self.worker = NS()

    def test_digest_mismatch_raises_and_is_not_cached(self):
        self.config.projector_sha256 = "0" * 64
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "projector digest mismatch"):
                _verify_projector(self.worker, self.config)
        self.assertFalse(hasattr(self.worker, "_glm53_apc_projector"))

    def test_verified_identity_is_cached_by_mtime_and_size(self):
        _verify_projector(self.worker, self.config)
        before = self.path.stat()
        # Same size and restored mtime: a re-hash would see the wrong bytes.
        self.path.write_bytes(b"projector-v2")
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = self.path.stat()
        self.assertEqual(
            (after.st_size, after.st_mtime_ns), (before.st_size, before.st_mtime_ns)
        )
        _verify_projector(self.worker, self.config)
        self.path.write_bytes(b"projector-v2+")
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "projector digest mismatch"):
            _verify_projector(self.worker, self.config)

    def test_signature_change_forces_a_rehash(self):
        _verify_projector(self.worker, self.config)
        before = self.path.stat()
        self.path.write_bytes(b"projector-v2")
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.config.signature = "other"
        with self.assertRaisesRegex(ValueError, "projector digest mismatch"):
            _verify_projector(self.worker, self.config)


if __name__ == "__main__":
    unittest.main()
