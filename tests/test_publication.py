import tempfile
import unittest
from pathlib import Path

from tools.check_publication import audit


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
