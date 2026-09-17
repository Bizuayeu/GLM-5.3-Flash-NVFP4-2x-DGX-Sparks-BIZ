import tempfile
import unittest
from pathlib import Path

from tools.check_publication import audit, headline_problems, recipe_problems


class PublicationTests(unittest.TestCase):
    def test_harness_settings_are_not_public_even_if_explicitly_tracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = ".claude-local-test/settings.json"
            target = root / name
            target.parent.mkdir()
            target.write_text("{}", encoding="utf-8")
            self.assertIn(f"private/generated path: {name}", audit(root, {name}))

    def test_locally_existing_private_link_is_not_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "guide.md").write_text("[result](records/run.md)", encoding="utf-8")
            (root / "records").mkdir()
            (root / "records/run.md").write_text("private", encoding="utf-8")
            issues = audit(root, {"guide.md"})
            self.assertIn(
                "non-public Markdown target: guide.md -> records/run.md", issues
            )

    def test_sensitive_candidate_is_reported_without_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = "hf" + "_" + "a" * 30
            (root / "sample.txt").write_text(candidate, encoding="utf-8")
            issues = audit(root, {"sample.txt"})
            self.assertIn("sensitive-text candidate: sample.txt", issues)
            self.assertNotIn(candidate, "\n".join(issues))

    def test_tracked_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_text("{}", encoding="utf-8")
            self.assertIn(
                "private/generated path: model.safetensors",
                audit(root, {"model.safetensors"}),
            )


class HeadlineTests(unittest.TestCase):
    BENCHMARKS = "## Measurements on 1.4.0\n\n## Measurements on 1.10.0\n"

    def test_readme_headline_must_name_the_latest_measured_version(self):
        self.assertEqual(
            headline_problems(
                "README.md", "### Headline measurements (1.10.0)\n", self.BENCHMARKS
            ),
            [],
        )
        self.assertEqual(
            headline_problems(
                "README.ja.md", "### 主要な測定値（1.4.0）\n", "## 1.10.0での測定\n"
            ),
            ["stale headline measurements: README.ja.md has 1.4.0, latest is 1.10.0"],
        )

    def test_missing_headline_or_measurements_are_reported(self):
        self.assertEqual(
            headline_problems("README.md", "# Title\n", self.BENCHMARKS),
            ["missing headline measurements: README.md"],
        )
        self.assertEqual(
            headline_problems("README.md", "### Headline measurements (1.0.0)\n", ""),
            ["no versioned measurements beside README.md"],
        )


class RecipeCitationTests(unittest.TestCase):
    README = (
        "### Other recipes\n\n| Recipe | License |\n|---|---|\n"
        "| [Mia](https://github.com/MiaAI-Lab/GLM-Sparks) | AGPL-3.0 |\n"
        "| [sfxnz](https://github.com/sfxnz/GLM-vLLM) | MIT |\n"
    )

    def test_license_beside_a_citation_is_a_second_copy(self):
        issues = recipe_problems(
            self.README,
            {"docs/a.md": "Informed by Mia PR #1 (AGPL-3.0, no code adopted).\n"},
        )
        self.assertEqual(issues, ["recipe license restated outside README: docs/a.md"])
        self.assertEqual(
            recipe_problems(
                self.README,
                {"docs/a.ja.md": "sfxnz PR #1（MIT、コードは採用しない）\n"},
            ),
            ["recipe license restated outside README: docs/a.ja.md"],
        )

    def test_recipe_links_live_in_the_readme_table_only(self):
        issues = recipe_problems(
            self.README, {"docs/a.md": "See https://github.com/sfxnz/GLM-vLLM/pull/1\n"}
        )
        self.assertEqual(issues, ["recipe link outside README: docs/a.md"])

    def test_plain_citations_and_unrelated_licenses_pass(self):
        docs = {
            "docs/a.md": "Informed by Mia PR #1 (no code adopted); no new AGPL "
            "dependency. vLLM is Apache-2.0 (pinned, unchanged).\n"
        }
        self.assertEqual(recipe_problems(self.README, docs), [])
