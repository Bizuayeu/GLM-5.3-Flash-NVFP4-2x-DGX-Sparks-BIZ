import json
import random
import unittest

from glm53_setup.runtime.adaptive_depth import DepthPolicy
from glm53_setup.validation import depth_replay as replay

COSTS = replay.cost_table(5)


def record(tokens, depth=5):
    """What a fixed-depth run traces on a text whose token t has a holding draft iff tokens[t]."""
    steps, position, last = [], 1, len(tokens) - 1
    while position < last:
        drafted = max(1, min(depth, last - position - 1))
        accepted = 0
        while accepted < drafted and tokens[position + accepted + 1]:
            accepted += 1
        steps.append((position, drafted, accepted))
        position += accepted + 1
    return steps


def independent(count, rate, seed):
    rng = random.Random(seed)
    return [rng.random() < rate for _ in range(count)]


def regimes(count, easy, hard, length, seed):
    """Stretches of easy and hard text: hits and misses come in runs, as in real completions."""
    rng, good, tokens = random.Random(seed), True, []
    for _ in range(count):
        if rng.random() < 1 / length:
            good = not good
        tokens.append(rng.random() < (easy if good else hard))
    return tokens


class TraceTests(unittest.TestCase):
    def test_lines_group_by_request_in_order(self):
        lines = [
            json.dumps(["a", 0, 1, 5, 2, 5, 10]),
            json.dumps(["b", 0, 1, 5, 5, 5, 11]),
            "",
            json.dumps(["a", 1, 4, 5, 0, 5, 12]),
        ]
        self.assertEqual(
            replay.read_trace(lines), {"a": [(1, 5, 2), (4, 5, 0)], "b": [(1, 5, 5)]}
        )

    def test_a_step_says_which_tokens_had_a_holding_draft(self):
        steps = [(1, 5, 2), (4, 5, 0), (5, 3, 3)]
        self.assertEqual(
            replay.token_hits(steps, unknown="miss"),
            {2: True, 3: True, 4: False, 5: False, 6: True, 7: True, 8: True, 9: False},
        )
        # Token 9 followed a fully accepted step: no draft for it was ever checked.
        self.assertIs(replay.token_hits(steps, unknown="hit")[9], True)
        self.assertEqual(replay.span(steps), (1, 9))
        with self.assertRaises(ValueError):
            replay.token_hits(steps, unknown="guess")


class ReplayTests(unittest.TestCase):
    def test_the_recorded_depth_replays_to_the_recorded_steps(self):
        steps = record(regimes(1500, 0.95, 0.7, 30, seed=3))
        for unknown in ("miss", "impute", "hit"):
            again = replay.replay(steps, replay.Fixed(5), COSTS, unknown)
            self.assertEqual(again["steps"], len(steps))
            self.assertEqual(again["tokens"], replay.span(steps)[1] - 1)

    def test_a_shallower_depth_is_exact_where_no_step_was_fully_accepted(self):
        # Runs of at most four holding drafts, so a depth-5 record has no unknown token
        # (the last, shortened step aside).
        tokens = [bool(int(c)) for c in "1110100111101101" * 60] + [False] * 3
        steps = record(tokens)
        self.assertTrue(all(accepted < drafted for _, drafted, accepted in steps))
        for depth in (1, 2, 3, 4):
            again = replay.replay(steps, replay.Fixed(depth), COSTS, unknown="miss")
            self.assertEqual(again["steps"], len(record(tokens, depth)))

    def test_imputing_the_unknown_tokens_recovers_a_shallower_run(self):
        tokens = independent(6000, 0.9, seed=1)
        steps = record(tokens)
        truth = len(record(tokens, depth=3))
        by_mode = {
            mode: replay.replay(steps, replay.Fixed(3), COSTS, mode)["steps"]
            for mode in ("miss", "impute", "hit")
        }
        # All-accepted steps are common here; calling their next token a miss adds many steps.
        self.assertGreater(by_mode["miss"], 1.08 * truth)
        self.assertLess(abs(by_mode["impute"] - truth), 0.02 * truth)
        self.assertLessEqual(by_mode["hit"], by_mode["impute"])

    def test_cost_overrides_replace_single_depths(self):
        costs = replay.cost_table(3, overrides={1: 66.0})
        self.assertEqual(costs, {1: 66.0, 2: 74.0, 3: 87.5})

    def test_a_policy_never_drafts_past_the_end_of_the_request(self):
        result = replay.replay([(1, 5, 5), (7, 1, 1)], replay.Fixed(5), COSTS)
        self.assertEqual(result["tokens"], 8)


class CensoringTests(unittest.TestCase):
    """The reference pair, 2026-09-21: code sat at depth 2 where fixed 3 was best."""

    def setUp(self):
        self.steps = record(regimes(4000, 0.95, 0.7, 30, seed=0))
        self.rows = replay.compare(self.steps, 5, COSTS)

    def test_the_policy_that_ran_on_the_pair_stays_below_the_best_fixed_depth(self):
        fixed = {d: self.rows[f"fixed-{d}"]["tokens_per_second"] for d in range(1, 6)}
        self.assertEqual(max(fixed, key=fixed.get), 3)
        current = self.rows["current"]
        self.assertLess(current["tokens_per_second"], 0.97 * fixed[3])
        deep = sum(n for depth, n in current["depths"].items() if depth >= 3)
        self.assertLess(deep, 0.5 * current["steps"])

    def test_the_estimate_of_a_depth_it_left_is_fed_by_probes_only(self):
        policy = DepthPolicy(5)
        policy.shares = [0.9, 0.7, 0.3, 0.1, 0.05]
        before = list(policy.shares)
        for _ in range(15):
            depth = policy.next_depth()
            self.assertEqual(depth, 2)
            policy.observe(depth, 2)
        # Fifteen steps of two accepted drafts say nothing about depth 3.
        self.assertEqual(policy.shares[2:], before[2:])
        self.assertEqual(policy.next_depth(), 5)

    def test_every_candidate_is_replayed_on_the_same_text(self):
        self.assertEqual(
            set(self.rows), {f"fixed-{d}" for d in range(1, 6)} | set(replay.CANDIDATES)
        )
        for row in self.rows.values():
            self.assertEqual(row["tokens"], self.rows["fixed-5"]["tokens"])


class CandidateTests(unittest.TestCase):
    def test_defaults_are_the_policy_as_it_ran(self):
        plain, spelled = DepthPolicy(5), DepthPolicy(5, band=0.0, probe="ceiling")
        rng = random.Random(2)
        for _ in range(300):
            depth = plain.next_depth()
            self.assertEqual(depth, spelled.next_depth())
            accepted = min(depth, int(rng.random() * 4))
            plain.observe(depth, accepted)
            spelled.observe(depth, accepted)

    def test_a_band_keeps_a_depth_in_use_and_asks_more_of_a_new_one(self):
        # The threshold for depth 3 here is 13.5 * 2.5 / 74 = 0.456.
        def depth_after(current, third):
            policy = DepthPolicy(5, band=0.2, probe_every=0)
            policy.shares, policy.depth = [0.9, 0.6, third, 0.0, 0.0], current
            return policy.next_depth()

        self.assertEqual(depth_after(current=3, third=0.40), 3)
        self.assertEqual(depth_after(current=2, third=0.40), 2)
        self.assertEqual(depth_after(current=2, third=0.50), 2)
        self.assertEqual(depth_after(current=2, third=0.56), 3)
        self.assertEqual(depth_after(current=3, third=0.35), 2)

    def test_the_next_probe_tries_one_draft_more_than_the_depth_in_use(self):
        policy = DepthPolicy(5, probe="next", probe_every=4)
        policy.shares = [0.7, 0.39, 0.15, 0.05, 0.0]
        self.assertEqual([policy.next_depth() for _ in range(8)], [2, 2, 2, 3] * 2)
        policy.shares = [0.95, 0.9, 0.9, 0.85, 0.8]
        self.assertEqual([policy.next_depth() for _ in range(4)], [5, 5, 5, 5])

    def test_bad_candidate_settings_are_refused(self):
        with self.assertRaises(ValueError):
            DepthPolicy(5, probe="sometimes")
        with self.assertRaises(ValueError):
            DepthPolicy(5, band=1.0)


if __name__ == "__main__":
    unittest.main()
