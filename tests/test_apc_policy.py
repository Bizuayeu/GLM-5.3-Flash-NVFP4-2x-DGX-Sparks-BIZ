"""APC-first admission and exact-only sharing contracts."""

import unittest

from glm53_setup.runtime.apc_policy import PrefillPolicy, plan_prefill


class APCPolicyTests(unittest.TestCase):
    def test_decision_uses_restored_prefix_not_total_context(self):
        cold = plan_prefill(16384, 0, 512, 1024)
        warm = plan_prefill(16384, 15360, 512, 1024)
        self.assertEqual(cold.eligible_tokens, 15872)
        self.assertTrue(cold.approximates)
        self.assertEqual(cold.shared_cache_limit, 0)
        self.assertEqual(warm.eligible_tokens, 512)
        self.assertFalse(warm.approximates)
        self.assertIsNone(warm.shared_cache_limit)

    def test_strict_break_even_and_exact_tail(self):
        at = plan_prefill(2048, 512, 512, 1024)
        above = plan_prefill(2049, 512, 512, 1024)
        self.assertFalse(at.approximates)
        self.assertTrue(above.approximates)
        self.assertEqual((above.approximate_start, above.approximate_end), (512, 1537))

    def test_exact_priming_and_short_prompts_do_not_approximate(self):
        prime = plan_prefill(16000, 0, 512, 0, exact=True)
        short = plan_prefill(32, 0, 512, 0)
        self.assertIsNone(prime.shared_cache_limit)
        self.assertFalse(prime.approximates)
        self.assertEqual(short.tail, 32)
        self.assertEqual(short.eligible_tokens, 0)
        self.assertFalse(short.approximates)

    def test_shared_limit_also_excludes_exact_tail_and_decode(self):
        policy = plan_prefill(16000, 8704, 512, 1024)
        for attempted_end in (9000, 15488, 16000, 16128):
            self.assertEqual(policy.cacheable_end(attempted_end), 8704)
        self.assertEqual(policy.cacheable_end(4352), 4352)

    def test_preemption_preserves_original_window_and_taint(self):
        first = plan_prefill(16000, 8704, 512, 1024)
        lost = plan_prefill(16000, 0, 512, 1024, previous=first)
        self.assertEqual(lost.approximate_start, 8704)
        self.assertEqual(lost.shared_cache_limit, 8704)
        covered = plan_prefill(16000, 15872, 512, 1024, previous=lost)
        self.assertFalse(covered.approximates)
        self.assertEqual(covered.shared_cache_limit, 8704)

    def test_preempted_exact_request_keeps_its_exact_policy(self):
        first = plan_prefill(16000, 15360, 512, 1024)
        resumed = plan_prefill(16000, 0, 512, 1024, previous=first)
        self.assertFalse(resumed.approximates)
        self.assertIsNone(resumed.shared_cache_limit)

    def test_wire_round_trip_rejects_unrecognized_or_inconsistent_values(self):
        policy = plan_prefill(16000, 8704, 512, 1024)
        self.assertEqual(PrefillPolicy.from_dict(policy.to_dict()), policy)
        data = policy.to_dict()
        data["shared_cache_limit"] = 16000
        with self.assertRaises(ValueError):
            PrefillPolicy.from_dict(data)
        data = policy.to_dict()
        data["unknown"] = 1
        with self.assertRaises(ValueError):
            PrefillPolicy.from_dict(data)

    def test_invalid_lengths_and_cross_request_resume_are_rejected(self):
        for values in (
            (0, 0, 1, 0),
            (10, 11, 1, 0),
            (10, -1, 1, 0),
            (10, 0, 0, 0),
            (10, 0, 1, -1),
            (True, 0, 1, 0),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                plan_prefill(*values)
        with self.assertRaises(ValueError):
            plan_prefill(100, 0, 5, 2, previous=plan_prefill(101, 0, 5, 2))


if __name__ == "__main__":
    unittest.main()
