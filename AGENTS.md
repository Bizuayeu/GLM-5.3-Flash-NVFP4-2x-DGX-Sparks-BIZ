# Repository instructions

Read README.md before changing this model deployment. Keep pinned model revisions and runtime settings explicit. Put CPU contract tests in tests/ and reproducible validation summaries in docs/.

Credentials (.env), generated state, local records, upstream checkouts and model weights are not tracked. Do not print keys or include raw private logs in commits. Keep hardware checks distinct from real-model inference results.

This repository can be used independently. Resolve source/data paths relative to this repository or an explicit argument; do not depend on a parent workspace. Changes must preserve existing generation settings unless the task asks to change them.
