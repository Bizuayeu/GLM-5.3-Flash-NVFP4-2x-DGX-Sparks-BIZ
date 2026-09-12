# Repository instructions

Read README.md before changing this model deployment. Keep pinned model revisions and runtime settings explicit. Put CPU contract tests in tests/ and reproducible validation summaries in docs/.

For deployment, read SETUP.md and its linked operations/validation documents first. This is a beta: full-model TP=2 is unqualified, and no qualification-receipt producer exists yet. Do not fabricate a receipt or bypass the gate. Respect explicitly paused downloads and preserve other workloads. Record each action and its evidence privately under records/.

Use `python -m glm53_setup` from the checkout. Run `python -m unittest discover -s tests -t . -v`, Ruff checks, and `python tools/check_publication.py` after relevant changes. Keep English and Japanese README/setup guides consistent. Original code uses Apache-2.0; preserve all adapted-code notices.

Credentials (.env), generated state, local records, upstream checkouts and model weights are not tracked. Do not print keys or include raw private logs in commits. Keep hardware checks distinct from real-model inference results.

This repository can be used independently. Resolve source/data paths relative to this repository or an explicit argument; do not depend on a parent workspace. Changes must preserve existing generation settings unless the task asks to change them.
