import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup.io import write_json


class StateWriteTests(unittest.TestCase):
    def test_failed_publication_preserves_previous_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            write_json(path, {"state": "old"})
            original = path.read_bytes()
            with patch("pathlib.Path.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    write_json(path, {"state": "new"})
            self.assertEqual(path.read_bytes(), original)

    def test_temporary_names_are_unique_for_independent_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            with patch("pathlib.Path.replace") as replace:
                write_json(path, {"value": 1})
                write_json(path, {"value": 2})
            self.assertEqual(replace.call_count, 2)
            self.assertEqual(len(list(Path(directory).glob("*.tmp"))), 2)

    def test_a_hard_linked_file_is_replaced_and_its_other_name_keeps_the_old_bytes(
        self,
    ):
        # requant.py hard-links small files from the source fixture; rewriting the
        # derived fixture's status must not write through into the source's.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            write_json(source, {"status": "source"})
            derived = Path(directory) / "derived.json"
            derived.hardlink_to(source)
            write_json(derived, {"status": "derived"})
            self.assertIn(b'"source"', source.read_bytes())
            self.assertIn(b'"derived"', derived.read_bytes())
