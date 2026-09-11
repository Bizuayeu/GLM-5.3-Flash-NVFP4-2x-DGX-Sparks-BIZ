import unittest

from glm53_setup.validation.llkv_corpus import (
    accepted_code_license,
    document_key,
    split_for,
)


class CorpusSplitTests(unittest.TestCase):
    def test_missing_or_mixed_restrictive_license_is_not_selected(self):
        self.assertTrue(accepted_code_license({"max_stars_repo_licenses": ["MIT"]}))
        self.assertFalse(accepted_code_license({}))
        self.assertFalse(
            accepted_code_license({"max_stars_repo_licenses": ["MIT", "GPL-3.0"]})
        )

    def test_duplicate_text_stays_in_one_split_despite_metadata_or_whitespace(self):
        a = document_key({"text": "A document.\nSecond sentence.", "meta": {"id": "1"}})
        b = document_key({"text": "A document. Second sentence.", "meta": {"id": "2"}})
        self.assertEqual(a, b)
        self.assertEqual(split_for(a), split_for(b))

    def test_named_partitions_are_stable_and_disjoint(self):
        self.assertEqual(split_for("00000008"), "validation")
        self.assertEqual(split_for("00000009"), "test")
        self.assertEqual(split_for("00000000"), "train")


if __name__ == "__main__":
    unittest.main()
