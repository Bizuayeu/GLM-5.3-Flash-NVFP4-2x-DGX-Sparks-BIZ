"""Admission precedes publication; failed admission must not freeze a stale H."""

import json
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from glm53_setup.runtime.apc_runtime import (
    POLICY_KEY,
    allocation_guard,
    publication_end,
    validate_client_options,
    validate_publication,
)

CONFIG = json.dumps(
    {
        "cut": 32,
        "tail": 4,
        "break_even": 8,
        "projector_path": "/lpa/projector.pt",
        "projector_sha256": "0" * 64,
        "skip_mla_queries": True,
    }
)


def request(name="a", **extra):
    return NS(
        request_id=name,
        num_prompt_tokens=64,
        num_computed_tokens=0,
        sampling_params=NS(extra_args=extra),
        prompt_embeds=None,
        mm_features=[],
        lora_request=None,
    )


class APCRuntimeTests(unittest.TestCase):
    def test_client_options_reject_bad_modes_and_reserved_policy_before_admission(self):
        for extra in (None, {}, {"glm53_lpa_mode": "auto"}, {"glm53_lpa_mode": "off"}):
            validate_client_options(extra)
        for extra in (
            {POLICY_KEY: "forged"},
            {"glm53_lpa_mode": "predict"},
            {"glm53_lpa_mode": 1},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_client_options(extra)

    def setUp(self):
        self.environment = patch.dict(os.environ, {"GLM53_APC_LPA_CONFIG": CONFIG})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_private_allocation_is_not_shortened_and_publication_is_capped(self):
        observed = []

        @allocation_guard
        def allocate(manager, req, new, **kwargs):
            observed.append((new, publication_end(req, 64)))
            return ["private blocks"]

        req = request()
        result = allocate(NS(enable_caching=True), req, 48, num_new_computed_tokens=16)
        self.assertEqual(result, ["private blocks"])
        self.assertEqual(observed, [(48, 16)])
        self.assertEqual(
            req.sampling_params.extra_args[POLICY_KEY]["policy"]["cached_tokens"], 16
        )
        req.num_computed_tokens = 64
        self.assertEqual(publication_end(req, 80), 16)

    def test_failed_admission_can_retry_with_a_different_cache_hit(self):
        outcomes = iter((None, ["blocks"]))

        @allocation_guard
        def allocate(manager, req, new, **kwargs):
            return next(outcomes)

        req = request()
        self.assertIsNone(allocate(NS(enable_caching=True), req, 64))
        self.assertNotIn(POLICY_KEY, req.sampling_params.extra_args)
        allocate(NS(enable_caching=True), req, 1, num_new_computed_tokens=63)
        self.assertIsNone(
            req.sampling_params.extra_args[POLICY_KEY]["policy"]["shared_cache_limit"]
        )

    def test_running_progress_does_not_reselect_the_threshold(self):
        @allocation_guard
        def allocate(manager, req, new, **kwargs):
            return []

        req = request()
        allocate(NS(enable_caching=True), req, 8, num_new_computed_tokens=16)
        first = req.sampling_params.extra_args[POLICY_KEY]
        req.num_computed_tokens = 56
        allocate(NS(enable_caching=True), req, 4)
        self.assertEqual(req.sampling_params.extra_args[POLICY_KEY], first)
        self.assertEqual(publication_end(req, 60), 16)

    def test_exact_override_and_forged_internal_policy(self):
        @allocation_guard
        def allocate(manager, req, new, **kwargs):
            return []

        prime = request(glm53_lpa_mode="off")
        allocate(NS(enable_caching=True), prime, 64)
        self.assertEqual(publication_end(prime, 68), 68)
        with self.assertRaises(ValueError):
            allocate(NS(enable_caching=True), request(**{POLICY_KEY: "forged"}), 64)
        with self.assertRaises(ValueError):
            allocate(NS(enable_caching=True), request(glm53_lpa_mode="force"), 64)

    def test_low_level_guard_detects_attempted_tainted_registration(self):
        @allocation_guard
        def allocate(manager, req, new, **kwargs):
            return []

        req = request()
        allocate(NS(enable_caching=True), req, 48, num_new_computed_tokens=16)
        blocks = [NS(is_null=False), NS(is_null=False)]
        validate_publication(req, blocks, 0, 8, None)
        with self.assertRaises(ValueError):
            validate_publication(req, blocks, 1, 16, None)
        validate_publication(req, blocks, 1, 16, [False, False])

    def test_disabled_feature_is_an_unmodified_allocation(self):
        observed = []

        @allocation_guard
        def allocate(manager, req, new, **kwargs):
            observed.append((new, kwargs))
            return "unchanged"

        with patch.dict(os.environ, {"GLM53_APC_LPA_CONFIG": ""}):
            self.assertEqual(
                allocate(None, None, 9, num_lookahead_tokens=3), "unchanged"
            )
        self.assertEqual(observed, [(9, {"num_lookahead_tokens": 3})])


if __name__ == "__main__":
    unittest.main()
