import unittest

from tools.release_notes import section

CHANGELOG = """# Changelog

## Unreleased

- pending

## 1.10.0 — 2026-09-18

### Changed

- ten

## 1.1.0 — 2026-09-15

- one
"""


class ReleaseNotesTests(unittest.TestCase):
    def test_returns_only_the_named_version(self):
        self.assertEqual(section(CHANGELOG, "1.10.0"), "### Changed\n\n- ten\n")
        self.assertEqual(section(CHANGELOG, "1.1.0"), "- one\n")

    def test_refuses_a_version_without_notes(self):
        for version in ("1.1", "1.2.0", "Unreleased", "1.1.0 "):
            with self.assertRaises(ValueError):
                section(CHANGELOG, version)
        with self.assertRaises(ValueError):
            section("## 2.0.0 — 2026-10-01\n\n## 1.0.0 — 2026-09-15\n\n- x\n", "2.0.0")


if __name__ == "__main__":
    unittest.main()
