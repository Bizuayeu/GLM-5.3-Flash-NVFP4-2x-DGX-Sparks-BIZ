import json
import random
import unittest

from glm53_setup.validation import draft_observe_stats as stats


def step(accepted, probabilities, margins=None, wanted=None):
    """A record row of depth len(probabilities); draft ids are 100 + depth."""
    depth = len(probabilities)
    drafts = [100 + i for i in range(depth)]
    target = list(drafts) + [7]
    if accepted < depth:
        target[accepted] = wanted if wanted is not None else 999
    return {
        "r": "r",
        "p": 1,
        "a": accepted,
        "d": drafts,
        "dp": probabilities,
        "dm": [1.0] * depth,
        "d5": [[draft, 201, 202, 203, 204] for draft in drafts],
        "t": target,
        "tl": [-1.0] * depth,
        "tm": margins or [2.0] * (depth + 1),
    }


def informative(count, seed):
    """Drafts hold with exactly the probability the draft reports; a step is easy or hard."""
    rng, rows = random.Random(seed), []
    for _ in range(count):
        levels = (0.98, 0.9) if rng.random() < 0.5 else (0.6, 0.3)
        probabilities = [rng.choice(levels) for _ in range(5)]
        accepted = 0
        while accepted < 5 and rng.random() < probabilities[accepted]:
            accepted += 1
        rows.append(step(accepted, probabilities))
    return rows


class SampleTests(unittest.TestCase):
    def test_a_depth_counts_only_where_the_shallower_drafts_held(self):
        rows = [step(0, [0.9, 0.8]), step(1, [0.7, 0.2]), step(2, [0.6, 0.5])]
        self.assertEqual(
            stats.samples(rows, 1), [(0.9, False), (0.7, True), (0.6, True)]
        )
        self.assertEqual(stats.samples(rows, 2), [(0.2, False), (0.5, True)])

    def test_lines_round_trip(self):
        rows = [step(1, [0.5, 0.5])]
        self.assertEqual(stats.read([json.dumps(rows[0]), "", "\n"]), rows)


class AucTests(unittest.TestCase):
    def test_perfect_inverse_tied_and_one_sided(self):
        self.assertEqual(stats.auc([(0.9, True), (0.8, True), (0.2, False)]), 1.0)
        self.assertEqual(stats.auc([(0.1, True), (0.8, False)]), 0.0)
        self.assertEqual(stats.auc([(0.5, True), (0.5, False)]), 0.5)
        self.assertIsNone(stats.auc([(0.5, True), (0.7, True)]))

    def test_it_is_the_share_of_correctly_ordered_pairs(self):
        pairs = [(0.9, True), (0.4, True), (0.6, False), (0.4, False)]
        # (0.9 > 0.6), (0.9 > 0.4), (0.4 < 0.6), (0.4 = 0.4 -> half): 2.5 of 4.
        self.assertEqual(stats.auc(pairs), 0.625)

    def test_an_informative_draft_scores_well_and_is_calibrated(self):
        pairs = stats.samples(informative(4000, seed=0), 1)
        self.assertGreater(stats.auc(pairs), 0.7)
        for row in stats.calibration(pairs):
            self.assertLess(abs(row["held"] - row["mean_probability"]), 0.05)


class RejectionTests(unittest.TestCase):
    def test_rejections_are_classed_by_the_targets_margin_and_the_drafts_top5(self):
        rows = [
            step(1, [0.9, 0.5], margins=[3.0, 0.0, 1.0]),
            step(1, [0.9, 0.5], margins=[3.0, 0.0625, 1.0], wanted=202),
            step(1, [0.9, 0.5], margins=[3.0, 0.5, 1.0], wanted=204),
            step(1, [0.9, 0.5], margins=[3.0, 6.0, 1.0]),
            step(2, [0.9, 0.5]),
            step(0, [0.9, 0.5]),
        ]
        found = stats.rejections(rows, 2)
        self.assertEqual(
            found["target_margin"],
            {"tie": 1, "near-tie": 1, "uncertain": 1, "confident": 1},
        )
        self.assertEqual(found["target_rank_in_draft"], {">5": 2, "3": 1, "5": 1})
        self.assertEqual(stats.rejections(rows, 1)["target_margin"], {"confident": 1})

    def test_a_draft_side_row_that_is_not_about_this_draft_is_dropped(self):
        other = step(0, [0.9, 0.9])
        other["d5"][0][0] = 555
        rows = [other, step(1, [0.8, 0.7]), step(2, [0.8, 0.7])]
        self.assertFalse(stats.paired(other, 1))
        self.assertTrue(stats.paired(other, 2))
        self.assertEqual(stats.rejections(rows, 1)["unpaired"], 1)
        self.assertEqual(stats.samples(rows, 1), [(0.8, True), (0.8, True)])
        result = stats.summary(rows, stats.cost_table(2))
        self.assertEqual(result["depths"][1]["unpaired"], 1)
        # The speed projections use only steps whose every depth is paired.
        self.assertAlmostEqual(
            result["speed"]["fixed-2"]["tokens_per_second"], 1000 * 5 / (2 * 74.0)
        )

    def test_on_an_exact_tie_argmax_and_topk_may_name_different_ids(self):
        # Fixture smoke 2026-09-21: 21 of 2035 draft rows, all with margin 0, the draft
        # second in the top 5.
        row = step(1, [0.4, 0.4])
        row["d5"][0] = [16, 100, 202, 203, 204]
        self.assertFalse(stats.paired(row, 1))
        row["dm"][0] = 0.0
        self.assertTrue(stats.paired(row, 1))
        self.assertEqual(
            stats.summary([row], stats.cost_table(2))["depths"][1]["draft_ties"], 1
        )
        row["d5"][0] = [16, 17, 202, 203, 204]
        self.assertFalse(stats.paired(row, 1))


class SpeedTests(unittest.TestCase):
    def test_default_costs_carry_the_measured_depth_one(self):
        self.assertEqual(stats.cost_table(3), {1: 66.0, 2: 74.0, 3: 87.5})
        self.assertEqual(stats.cost_table(2, overrides={}), {1: 60.5, 2: 74.0})

    def test_the_gate_keeps_the_draft_whose_confidence_stopped_it(self):
        row = step(3, [0.95, 0.9, 0.4, 0.99, 0.99])
        self.assertEqual(stats.gated_depth(row, 0.5), 3)
        self.assertEqual(stats.gated_depth(row, 0.97), 1)
        self.assertEqual(stats.gated_depth(row, 0.1), 5)

    def test_projection_arithmetic(self):
        rows = [step(3, [0.9] * 5), step(0, [0.9] * 5)]
        costs = stats.cost_table(5)
        fixed2 = stats.project(rows, costs, lambda row: 2)
        self.assertAlmostEqual(fixed2["tokens_per_second"], 1000 * (3 + 1) / (2 * 74.0))
        oracle = stats.speed(rows, costs)["oracle"]
        self.assertAlmostEqual(
            oracle["tokens_per_second"], 1000 * (4 + 1) / (costs[4] + costs[1])
        )

    def test_a_gate_beats_every_fixed_depth_only_when_confidence_informs(self):
        costs = stats.cost_table(5)
        speed = stats.speed(informative(4000, seed=1), costs)
        fixed = max(
            v["tokens_per_second"] for k, v in speed.items() if k[:5] == "fixed"
        )
        gate = max(v["tokens_per_second"] for k, v in speed.items() if k[:4] == "gate")
        self.assertGreater(gate, 1.03 * fixed)
        self.assertGreater(speed["oracle"]["tokens_per_second"], gate)
        # The same acceptance with a confidence that says nothing: no gate helps.
        rng = random.Random(2)
        blind = informative(4000, seed=1)
        for row in blind:
            row["dp"] = [rng.random() for _ in row["dp"]]
        speed = stats.speed(blind, costs)
        fixed = max(
            v["tokens_per_second"] for k, v in speed.items() if k[:5] == "fixed"
        )
        gate = max(v["tokens_per_second"] for k, v in speed.items() if k[:4] == "gate")
        self.assertLess(gate, 1.01 * fixed)

    def test_summary_has_every_reached_depth(self):
        result = stats.summary(informative(500, seed=3), stats.cost_table(5))
        self.assertEqual(sorted(result["depths"]), [1, 2, 3, 4, 5])
        self.assertEqual(result["steps"], 500)


if __name__ == "__main__":
    unittest.main()
