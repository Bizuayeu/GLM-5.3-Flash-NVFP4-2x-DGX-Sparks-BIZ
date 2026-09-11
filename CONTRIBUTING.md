# Contributing

This is a beta engineering project. Keep claims limited to the exact hardware, image, weights, precision and workload tested.

Use Python 3.11 or newer and run from a checkout:

```sh
python -m unittest discover -s tests -v
ruff check glm53_setup tests tools
ruff format --check glm53_setup tests tools
python tools/check_publication.py
```

GPU checks are separate from CPU tests. Use the pinned image and record effective arguments, output completeness, numerical differences and failures. The validation procedure is in [docs/validation.md](docs/validation.md).

- Do not commit credentials, local site configuration, model weights, raw logs or private experiment records.
- Keep model/cache artifacts read-only during inference and preserve failed runs.
- Do not relax a runtime guard or a numerical criterion just to obtain a passing result.
- Keep original notices for copied/adapted code. New project contributions are submitted under Apache-2.0; third-party portions keep their applicable notices.
- Mark modified upstream files prominently and review the [licensing guide](docs/licensing.md) for the actual distribution scope.
- Keep ZCode and Claude Code results separate in the [harness acceptance matrix](docs/harnesses.md); never mark unexecuted cases passed.
- Update the English and Japanese READMEs together when their user-visible instructions change.
