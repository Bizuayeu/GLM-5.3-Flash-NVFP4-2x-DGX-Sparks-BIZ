import os
import unittest
from types import SimpleNamespace
from unittest import mock

from glm53_setup.runtime import adaptive_depth
from glm53_setup.runtime.adaptive_depth import DepthPolicy, worthwhile_depth

# S_1..S_5 measured with k=5 on the reference pair (2026-09-21, Track B stage 0).
PROSE = [0.703, 0.392, 0.153, 0.054, 0.000]
CODE = [0.882, 0.732, 0.630, 0.457, 0.331]
COUNT = [0.958, 0.905, 0.895, 0.832, 0.821]


def tokens_per_ms(shares, depth, base_ms=47.0, step_ms=13.5):
    return (1.0 + sum(shares[:depth])) / (base_ms + step_ms * depth)


class ThresholdTests(unittest.TestCase):
    def test_measured_shares_give_the_depths_the_fixed_depth_runs_favoured(self):
        self.assertEqual(worthwhile_depth(PROSE, 47.0, 13.5), 2)
        self.assertEqual(worthwhile_depth(CODE, 47.0, 13.5), 3)
        self.assertEqual(worthwhile_depth(COUNT, 47.0, 13.5), 5)

    def test_each_accepted_depth_raises_tokens_per_second_and_the_next_does_not(self):
        for shares in (PROSE, CODE):
            depth = worthwhile_depth(shares, 47.0, 13.5)
            for shallower in range(1, depth):
                self.assertLess(
                    tokens_per_ms(shares, shallower),
                    tokens_per_ms(shares, shallower + 1),
                )
            self.assertGreaterEqual(
                tokens_per_ms(shares, depth), tokens_per_ms(shares, depth + 1)
            )

    def test_depth_one_is_kept_even_when_no_draft_is_accepted(self):
        self.assertEqual(worthwhile_depth([0.0] * 5, 47.0, 13.5), 1)

    def test_the_climb_stops_at_the_first_depth_that_does_not_pay(self):
        self.assertEqual(worthwhile_depth([0.9, 0.1, 0.9, 0.9], 47.0, 13.5), 1)

    def test_an_untouched_deep_estimate_is_read_through_the_shallower_ones(self):
        # Fixture run 2026-09-21: pinned at 2 with a prohibitive draft cost, S_1 + S_2 fell
        # below 1 while S_3..S_5 still held their initial 1.0, and the climb went to 5.
        shares = [0.3, 0.1, 1.0, 1.0, 1.0]
        self.assertEqual(worthwhile_depth(shares, 47.0, 1e9, floor=2), 2)
        self.assertEqual(worthwhile_depth(shares, 47.0, 13.5), 1)
        self.assertEqual(shares, [0.3, 0.1, 1.0, 1.0, 1.0])

    def test_a_pinned_floor_holds_whatever_is_observed(self):
        policy = DepthPolicy(5, step_ms=1e9, probe_every=0, floor=2)
        for accepted in [0, 0, 2, 0, 1, 0, 0, 0, 2, 0] * 40:
            self.assertEqual(policy.next_depth(), 2)
            policy.observe(2, accepted)

    def test_free_drafts_go_to_the_ceiling_and_costly_ones_stay_at_the_floor(self):
        self.assertEqual(worthwhile_depth(PROSE[:4] + [0.01], 47.0, 0.0), 5)
        self.assertEqual(worthwhile_depth(COUNT, 47.0, 1e9), 1)
        self.assertEqual(worthwhile_depth(COUNT, 47.0, 1e9, floor=2), 2)
        self.assertEqual(worthwhile_depth(COUNT, 47.0, 1e9, floor=9), 5)


class PolicyTests(unittest.TestCase):
    def test_a_new_request_starts_at_the_ceiling(self):
        self.assertEqual(DepthPolicy(5).next_depth(), 5)

    def test_the_average_moves_by_alpha_and_only_for_drafted_positions(self):
        policy = DepthPolicy(5, alpha=0.25)
        policy.observe(drafted=3, accepted=1)
        self.assertEqual(policy.shares, [1.0, 0.75, 0.75, 1.0, 1.0])
        policy.observe(drafted=2, accepted=2)
        self.assertEqual(policy.shares, [1.0, 0.8125, 0.75, 1.0, 1.0])

    def test_a_longer_draft_list_than_the_ceiling_is_ignored_past_it(self):
        policy = DepthPolicy(2, alpha=0.5)
        policy.observe(drafted=5, accepted=0)
        self.assertEqual(policy.shares, [0.5, 0.5])

    def test_the_full_depth_is_probed_every_nth_step(self):
        policy = DepthPolicy(5, probe_every=4)
        policy.shares = list(PROSE)
        self.assertEqual([policy.next_depth() for _ in range(8)], [2, 2, 2, 5] * 2)

    def test_probe_zero_never_probes(self):
        policy = DepthPolicy(5, probe_every=0)
        policy.shares = list(PROSE)
        self.assertEqual({policy.next_depth() for _ in range(64)}, {2})

    def test_probes_let_a_lowered_estimate_recover(self):
        policy = DepthPolicy(5, alpha=0.125, probe_every=16)
        for _ in range(64):
            policy.observe(policy.next_depth(), 0)
        self.assertEqual(worthwhile_depth(policy.shares, 47.0, 13.5), 1)
        for _ in range(16 * 16):
            depth = policy.next_depth()
            policy.observe(depth, depth)
        self.assertEqual(worthwhile_depth(policy.shares, 47.0, 13.5), 5)

    def test_a_ceiling_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            DepthPolicy(0)


class EnvironmentTests(unittest.TestCase):
    def test_the_switch_is_off_by_default_and_strict(self):
        with mock.patch.dict(os.environ, clear=True):
            self.assertFalse(adaptive_depth.enabled())
        with mock.patch.dict(os.environ, {"GLM53_ADAPTIVE_DEPTH": "1"}):
            self.assertTrue(adaptive_depth.enabled())
        with mock.patch.dict(os.environ, {"GLM53_ADAPTIVE_DEPTH": "true"}):
            with self.assertRaises(ValueError):
                adaptive_depth.enabled()

    def test_defaults_are_the_measured_costs(self):
        with mock.patch.dict(os.environ, clear=True):
            self.assertEqual(
                adaptive_depth.settings(),
                {
                    "base_ms": 47.0,
                    "step_ms": 13.5,
                    "alpha": 0.125,
                    "probe_every": 16,
                    "floor": 1,
                },
            )

    def test_a_pinned_depth_for_fixture_runs(self):
        pinned = {
            "GLM53_ADAPTIVE_DEPTH_STEP_MS": "1e9",
            "GLM53_ADAPTIVE_DEPTH_PROBE_EVERY": "0",
            "GLM53_ADAPTIVE_DEPTH_MIN": "2",
        }
        with mock.patch.dict(os.environ, pinned, clear=True):
            request = SimpleNamespace()
            for _ in range(40):
                self.assertEqual(adaptive_depth.batch_depth([request], 5), 2)
                adaptive_depth.observe(request, 5, 2, 2)

    def test_bad_numbers_are_refused(self):
        for name, value in (
            ("GLM53_ADAPTIVE_DEPTH_BASE_MS", "-1"),
            ("GLM53_ADAPTIVE_DEPTH_STEP_MS", "nan"),
            ("GLM53_ADAPTIVE_DEPTH_ALPHA", "0"),
            ("GLM53_ADAPTIVE_DEPTH_ALPHA", "1.5"),
            ("GLM53_ADAPTIVE_DEPTH_PROBE_EVERY", "2.5"),
            ("GLM53_ADAPTIVE_DEPTH_MIN", "0"),
        ):
            with mock.patch.dict(os.environ, {name: value}, clear=True):
                with self.assertRaises(ValueError):
                    adaptive_depth.settings()

    def test_state_lives_on_the_request_and_the_step_takes_the_deepest(self):
        with mock.patch.dict(os.environ, clear=True):
            low, fresh = SimpleNamespace(), SimpleNamespace()
            for _ in range(32):
                adaptive_depth.observe(low, 5, 5, 0)
            self.assertEqual(adaptive_depth.batch_depth([low], 5), 1)
            self.assertEqual(adaptive_depth.batch_depth([low, fresh], 5), 5)
            self.assertEqual(adaptive_depth.batch_depth([], 5), 5)


if __name__ == "__main__":
    unittest.main()
