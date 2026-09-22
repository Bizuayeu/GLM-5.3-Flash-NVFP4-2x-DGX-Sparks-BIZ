import os
import unittest
from unittest import mock

from glm53_setup.runtime import prefix_dedup


class FakeMap:
    def __init__(self, cached):
        self.cached = cached

    def get_one_block(self, key):
        return self.cached.get(key)


class DedupTests(unittest.TestCase):
    def test_off_by_default_and_registers_as_pinned(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(prefix_dedup.dedup_enabled())
            self.assertFalse(
                prefix_dedup.skip_duplicate_page(FakeMap({"h": object()}), "h")
            )

    def test_on_skips_only_a_hash_that_already_has_a_block(self):
        with mock.patch.dict(os.environ, {"GLM53_PREFIX_PAGE_DEDUP": "1"}):
            cached = FakeMap({"h": object()})
            self.assertTrue(prefix_dedup.skip_duplicate_page(cached, "h"))
            self.assertFalse(prefix_dedup.skip_duplicate_page(cached, "other"))

    def test_rejects_values_other_than_0_and_1(self):
        with mock.patch.dict(os.environ, {"GLM53_PREFIX_PAGE_DEDUP": "yes"}):
            with self.assertRaises(ValueError):
                prefix_dedup.dedup_enabled()


if __name__ == "__main__":
    unittest.main()
