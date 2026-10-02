"""Admission precedes publication; failed admission must not freeze a stale H."""

import json
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from glm53_setup.runtime.apc_runtime import (
    POLICY_KEY,
    STATE_ATTR,
    RuntimeSettings,
    allocation_guard,
    publication_end,
    settings,
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


VALID = json.loads(CONFIG)


class RuntimeSettingsTests(unittest.TestCase):
    def assert_refused(self, message, **changes):
        with self.assertRaisesRegex(ValueError, message):
            RuntimeSettings(**{**VALID, **changes})

    def test_token_boundaries_must_be_nonnegative_integers_with_a_tail(self):
        for name in ("cut", "tail", "break_even"):
            for value in (1.0, "32", True, None):
                with self.subTest(name=name, value=value):
                    self.assert_refused("integer token boundaries", **{name: value})
        for changes in ({"cut": -1}, {"tail": 0}, {"tail": -1}, {"break_even": -1}):
            with self.subTest(changes=changes):
                self.assert_refused("^Invalid LPA policy settings$", **changes)
        RuntimeSettings(**{**VALID, "cut": 0, "tail": 1, "break_even": 0})

    def test_projector_path_digest_and_query_flag_are_strict(self):
        for path in ("lpa/projector.pt", "C:/lpa/projector.pt", "", None):
            with self.subTest(path=path):
                self.assert_refused("absolute container path", projector_path=path)
        for digest in ("0" * 63, "0" * 65, "A" * 64, "g" * 64, None):
            with self.subTest(digest=digest):
                self.assert_refused(
                    "^Invalid projector digest$", projector_sha256=digest
                )
        for flag in (1, 0, "true", None):
            with self.subTest(flag=flag):
                self.assert_refused("must be boolean", skip_mla_queries=flag)

    def test_environment_json_errors_are_wrapped_but_field_errors_are_not(self):
        for raw, expected in (("", None), (CONFIG, RuntimeSettings(**VALID))):
            with (
                self.subTest(raw=raw),
                patch.dict(os.environ, {"GLM53_APC_LPA_CONFIG": raw}),
            ):
                self.assertEqual(settings(), expected)
        malformed = (
            "{bad",
            " ",
            "[]",
            "null",
            "{}",
            json.dumps({**VALID, "unknown": 1}),
        )
        for raw in malformed:
            with (
                self.subTest(raw=raw),
                patch.dict(os.environ, {"GLM53_APC_LPA_CONFIG": raw}),
                self.assertRaisesRegex(
                    ValueError, "^Invalid APC/LPA environment configuration$"
                ),
            ):
                settings()
        with (
            patch.dict(
                os.environ, {"GLM53_APC_LPA_CONFIG": json.dumps({**VALID, "tail": 0})}
            ),
            self.assertRaisesRegex(ValueError, "^Invalid LPA policy settings$"),
        ):
            settings()


class AllocationAdmissionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"GLM53_APC_LPA_CONFIG": CONFIG})
        environment.start()
        self.addCleanup(environment.stop)
        self.calls = []
        self.outcome = []
        self.manager = NS(enable_caching=True)

        @allocation_guard
        def allocate(manager, req, new, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.outcome

        self.allocate = allocate

    def admitted(self):
        req = request()
        self.allocate(self.manager, req, 48, num_new_computed_tokens=16)
        self.calls.clear()
        return req

    def assert_refused(self, error, message, req, *args, manager=None, **kwargs):
        with self.assertRaisesRegex(error, message):
            self.allocate(manager or self.manager, req, 8, *args, **kwargs)
        self.assertEqual(self.calls, [])

    def test_pinned_signature_shape(self):
        self.allocate(self.manager, request(), 8, *([0] * 9))
        self.assertEqual(len(self.calls), 1)
        self.calls.clear()
        self.assert_refused(
            TypeError, "Unexpected pinned allocation signature", request(), *([0] * 10)
        )
        self.assert_refused(
            TypeError,
            "Duplicate allocation arguments",
            request(),
            16,
            num_new_computed_tokens=16,
        )

    def test_computed_token_boundaries_must_be_nonnegative_integers(self):
        for restored in (-1, 1.0, True):
            with self.subTest(restored=restored):
                self.assert_refused(
                    ValueError,
                    "Invalid computed-token boundary",
                    request(),
                    num_new_computed_tokens=restored,
                )
        for computed in (-1, None):
            with self.subTest(computed=computed):
                req = request()
                req.num_computed_tokens = computed
                self.assert_refused(ValueError, "Invalid computed-token boundary", req)

    def test_only_locally_restored_text_cache_is_admitted(self):
        message = "requires locally restored text cache"
        self.assert_refused(
            ValueError, message, request(), manager=NS(enable_caching=False)
        )
        for option in (
            {"num_external_computed_tokens": 8},
            {"delay_cache_blocks": True},
            {"num_encoder_tokens": 4},
        ):
            with self.subTest(option=option):
                self.assert_refused(ValueError, message, request(), **option)

    def test_lora_multimodal_embeds_and_pooling_requests_are_refused(self):
        for field, value in (
            ("lora_request", object()),
            ("mm_features", [object()]),
            ("prompt_embeds", object()),
            ("sampling_params", None),
        ):
            with self.subTest(field=field):
                req = request()
                setattr(req, field, value)
                self.assert_refused(ValueError, "plain text without LoRA", req)

    def test_client_cannot_supply_the_internal_policy(self):
        self.assert_refused(
            ValueError,
            "cannot be supplied by the client",
            request(**{POLICY_KEY: {"version": 1}}),
        )

    def test_live_request_cannot_change_mode_or_configuration(self):
        message = "Cannot change a live request's computation policy"
        req = self.admitted()
        req.sampling_params.extra_args["glm53_lpa_mode"] = "off"
        self.assert_refused(ValueError, message, req)
        req = self.admitted()
        changed = json.dumps({**VALID, "tail": 5})
        with patch.dict(os.environ, {"GLM53_APC_LPA_CONFIG": changed}):
            self.assert_refused(ValueError, message, req)

    def test_streaming_prompt_growth_is_refused(self):
        req = self.admitted()
        req.num_prompt_tokens = 72
        self.assert_refused(ValueError, "Streaming input changes are unsupported", req)

    def test_modified_internal_policy_is_refused(self):
        req = self.admitted()
        req.sampling_params.extra_args[POLICY_KEY]["policy"]["cached_tokens"] = 8
        self.assert_refused(ValueError, "Internal request policy was modified", req)

    def test_resumed_request_without_policy_is_refused(self):
        req = request()
        req.num_computed_tokens = 16
        self.assert_refused(ValueError, "lacks its admission policy", req)

    def test_restored_prefix_cannot_exceed_a_new_prompt(self):
        self.assert_refused(
            ValueError,
            "Restored prefix exceeds a new request's prompt",
            request(),
            num_new_computed_tokens=65,
        )
        self.allocate(self.manager, request(), 1, num_new_computed_tokens=64)
        self.assertEqual(len(self.calls), 1)

    def test_failed_readmission_restores_the_previous_state(self):
        req = self.admitted()
        previous = getattr(req, STATE_ATTR)
        wire = req.sampling_params.extra_args
        self.outcome = None
        self.assertIsNone(
            self.allocate(self.manager, req, 64, num_new_computed_tokens=0)
        )
        self.assertEqual(len(self.calls), 1)
        self.assertIs(getattr(req, STATE_ATTR), previous)
        self.assertIs(req.sampling_params.extra_args, wire)
        self.assertEqual(wire[POLICY_KEY], previous.wire())

    def test_failed_first_admission_leaves_no_state(self):
        req = request()
        self.outcome = None
        self.assertIsNone(self.allocate(self.manager, req, 64))
        self.assertFalse(hasattr(req, STATE_ATTR))
        self.assertEqual(req.sampling_params.extra_args, {})


if __name__ == "__main__":
    unittest.main()
