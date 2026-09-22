# Repository instructions

Read README.md before changing this model deployment; docs/README.md lists every document's role and which file owns each fact. Keep pinned model revisions and runtime settings explicit. Put CPU contract tests in tests/ and reproducible validation summaries in docs/.

For deployment, read SETUP.md and its linked operations/validation documents first. Acceptance for routine use is a record, not a command: SETUP.md step 6 states what the serving profile is accepted for and where each item's evidence is, and nothing outside that declared scope is qualified. Do not record a PASS without evidence, extend the acceptance to another scope by inference, or bypass `server preflight`. Respect explicitly paused downloads and preserve other workloads. Record each action and its evidence privately under records/.

Use `python -m glm53_setup` from the checkout. Run `python -m unittest discover -s tests -t . -v`, Ruff checks, and `python tools/check_publication.py` after relevant changes. Keep every English/Japanese document pair consistent; docs/README.md lists the pairs and the pages that are English-only by design. Original code uses Apache-2.0; preserve all adapted-code notices.

Credentials (.env), generated state, local records, upstream checkouts and model weights are not tracked. Do not print keys or include raw private logs in commits. Keep hardware checks distinct from real-model inference results.

Implementation plans live in docs/plans/, untracked and excluded from publication; docs/plans/README.md indexes them and each plan's own leading status line owns its state. Add a line to that index when creating a plan, and update the status line rather than the body when closing one.

This repository can be used independently. Resolve source/data paths relative to this repository or an explicit argument; do not depend on a parent workspace. Changes must preserve existing generation settings unless the task asks to change them.
