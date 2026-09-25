import os
import unittest
from unittest import mock

from glm53_setup.runtime import mla_decode_cpb


def cpb(tokens, reqs, query_len, prefills=0, heads=32, topk=2048):
    return mla_decode_cpb.decode_chunks_per_block(
        tokens, reqs, query_len, prefills, heads, topk
    )


class MlaDecodeCpbTest(unittest.TestCase):
    def test_off_unless_the_environment_turns_it_on(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(cpb(1, 1, 1))
        with mock.patch.dict(os.environ, {"GLM53_MLA_DECODE_CPB": "0"}):
            self.assertIsNone(cpb(1, 1, 1))
        with mock.patch.dict(os.environ, {"GLM53_MLA_DECODE_CPB": "yes"}):
            with self.assertRaises(ValueError):
                cpb(1, 1, 1)

    def test_one_value_per_tokens_of_one_sequence_whatever_the_partner(self):
        with mock.patch.dict(os.environ, {"GLM53_MLA_DECODE_CPB": "1"}):
            # Draft steps: one token per sequence, alone and with a partner.
            self.assertEqual(cpb(1, 1, 1), 3)
            self.assertEqual(cpb(2, 2, 1), 3)
            # Verification steps at MTP depth 3: four tokens per sequence.
            self.assertEqual(cpb(4, 1, 4), 6)
            self.assertEqual(cpb(8, 2, 4), 6)

    def test_everything_outside_the_measured_steps_keeps_the_kernel_choice(self):
        with mock.patch.dict(os.environ, {"GLM53_MLA_DECODE_CPB": "1"}):
            self.assertIsNone(cpb(5, 2, 4))  # sequences of unequal length
            self.assertIsNone(cpb(8, 2, 4, prefills=1))  # a prefill in the batch
            self.assertIsNone(cpb(6, 2, 3))  # a per-sequence length never measured
            self.assertIsNone(cpb(12, 3, 4))  # more sequences than measured
            self.assertIsNone(cpb(4, 1, 4, heads=64))  # another head count per rank
            self.assertIsNone(cpb(4, 1, 4, topk=1024))  # another top-k


if __name__ == "__main__":
    unittest.main()
