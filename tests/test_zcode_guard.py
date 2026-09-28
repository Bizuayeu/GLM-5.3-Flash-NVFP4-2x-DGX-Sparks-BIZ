import json
import shutil
import subprocess
import unittest
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / "examples/zcode-hooks/exists-guard.cjs"
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "node is not installed")
class GuardTests(unittest.TestCase):
    def decide(self, tool, **tool_input):
        payload = {"tool_name": tool, "tool_input": tool_input, "cwd": str(HOOK.parent)}
        out = subprocess.run(
            [NODE, str(HOOK)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return json.loads(out).get("hookSpecificOutput", {}).get("permissionDecision")

    def test_bash_that_names_the_config_directory_asks(self):
        # 2026-09-28 (records/20260928-harness-h-256k, H-04b): python -c wrote a
        # file under ~/.zcode without a prompt; only Write/Edit checked the path.
        for command in (
            "python -c \"open('D:/profile/.zcode/marker.txt','w').write('m')\"",
            'node -e "require(`fs`).writeFileSync(`D:\\\\profile\\\\.zcode\\\\x`,``)"',
            "cat notes >> ~/.zcode/AGENTS.md",
            "python -c \"import os;open(os.path.expanduser('~/.ZCODE/x'),'w')\"",
            'powershell -Command "Set-Location $env:USERPROFILE\\.zcode"',
        ):
            with self.subTest(command=command):
                self.assertEqual(self.decide("Bash", command=command), "ask")

    def test_ordinary_bash_is_still_allowed(self):
        for command in (
            "python -m unittest discover -s tests -v",
            "git log -1 --stat",
            "python -c \"print(open('data/part-01.log').read().count('ERROR'))\"",
            "ls my.zcoder backup-zcode",
        ):
            with self.subTest(command=command):
                self.assertEqual(self.decide("Bash", command=command), "allow")


if __name__ == "__main__":
    unittest.main()
