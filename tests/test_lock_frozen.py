import hashlib
import unittest

from glm53_setup.config import LOCK_PATH

# config/runtime.lock.json is hashed into every launch fingerprint (server_config.fingerprint). Editing
# it, even a documentary field, changes the fingerprint of every profile and makes a checkout unable
# to address a pair launched before the edit (1.8.0 -> 1.8.1). Change this constant only together with
# a CHANGELOG entry that says the fingerprints move.
LOCK_SHA256 = "db06aa96ac0ed8185f4dd334d4fbe5e13001e2247e31b5f39a328df6e320eaa5"


class LockTests(unittest.TestCase):
    def test_the_lock_is_frozen_because_it_is_inside_every_fingerprint(self):
        text = LOCK_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), LOCK_SHA256)


if __name__ == "__main__":
    unittest.main()
