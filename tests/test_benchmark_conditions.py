"""The paired condition an APC/LPA break-even measurement is only valid under.

Every arm of that benchmark is supposed to restore an identical prefix, so a
timing difference means the approximation and not a different amount of work.
``check`` is what enforces it: both ranks must report the prompt length, the
cached prefix and the eligible span the calibration asked for, and must have
skipped exactly the queries the arm's mode implies.

It captured nothing, yet lived inside a main() that needs a running two-rank
pair, so nothing tested the one function that decides whether a recorded
number means anything.
"""

import unittest

from glm53_setup.validation.benchmark_apc_lpa import check

APPROXIMATED_LAYERS = (35, 39, 43)


def worker(*, prompt, cached, eligible, skipped=None):
    return {
        "policy": {"policy": {"prompt_tokens": prompt, "cached_tokens": cached}},
        "eligible_tokens": eligible,
        "lpa": {"mla_queries_skipped": skipped} if skipped is not None else None,
    }


def row(*workers):
    return {"workers": list(workers)}


class PairedConditionTests(unittest.TestCase):
    def test_an_exact_arm_that_skipped_nothing_is_accepted(self):
        pair = row(*[worker(prompt=8192, cached=1024, eligible=0) for _ in range(2)])
        check(pair, 8192, 1024, 0, approximate=False)

    def test_an_approximate_arm_skipped_the_eligible_span_on_each_cut_layer(self):
        skipped = {str(layer): 6912 for layer in APPROXIMATED_LAYERS}
        pair = row(
            *[
                worker(prompt=8192, cached=1024, eligible=6912, skipped=skipped)
                for _ in range(2)
            ]
        )
        check(pair, 8192, 1024, 6912, approximate=True)

    def test_an_approximate_arm_with_nothing_eligible_skips_nothing(self):
        pair = row(
            *[worker(prompt=512, cached=0, eligible=0, skipped={}) for _ in range(2)]
        )
        check(pair, 512, 0, 0, approximate=True)

    def test_a_different_prompt_length_breaks_the_pairing(self):
        pair = row(*[worker(prompt=8000, cached=1024, eligible=0) for _ in range(2)])
        with self.assertRaises(ValueError):
            check(pair, 8192, 1024, 0, approximate=False)

    def test_a_different_restored_prefix_breaks_the_pairing(self):
        # The whole point of the arm: the same prefix must come back from cache,
        # or the timing compares two different amounts of prefill.
        pair = row(*[worker(prompt=8192, cached=0, eligible=0) for _ in range(2)])
        with self.assertRaises(ValueError):
            check(pair, 8192, 1024, 0, approximate=False)

    def test_a_different_eligible_span_breaks_the_pairing(self):
        pair = row(*[worker(prompt=8192, cached=1024, eligible=1) for _ in range(2)])
        with self.assertRaises(ValueError):
            check(pair, 8192, 1024, 0, approximate=False)

    def test_an_exact_arm_that_approximated_anyway_is_refused(self):
        pair = row(
            *[
                worker(prompt=8192, cached=1024, eligible=0, skipped={"35": 10})
                for _ in range(2)
            ]
        )
        with self.assertRaises(ValueError):
            check(pair, 8192, 1024, 0, approximate=False)

    def test_an_approximate_arm_that_missed_a_cut_layer_is_refused(self):
        partial = {str(layer): 6912 for layer in APPROXIMATED_LAYERS[:-1]}
        pair = row(
            *[
                worker(prompt=8192, cached=1024, eligible=6912, skipped=partial)
                for _ in range(2)
            ]
        )
        with self.assertRaises(ValueError):
            check(pair, 8192, 1024, 6912, approximate=True)

    def test_one_rank_off_the_condition_is_enough_to_refuse(self):
        pair = row(
            worker(prompt=8192, cached=1024, eligible=0),
            worker(prompt=8192, cached=512, eligible=0),
        )
        with self.assertRaises(ValueError):
            check(pair, 8192, 1024, 0, approximate=False)


if __name__ == "__main__":
    unittest.main()
