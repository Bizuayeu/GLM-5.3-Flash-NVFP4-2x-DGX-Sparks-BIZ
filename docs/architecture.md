# Architecture

The project is a checkout-local operator toolkit. It does not contain model weights or a remotely managed service.

| Location | Responsibility |
|---|---|
| `glm53_setup/__main__.py` | Fixed command dispatch; no dynamic user-supplied module loading |
| `glm53_setup/config.py` | Checkout paths and validated pinned configuration |
| `glm53_setup/download.py`, `images.py`, `build_reference.py`, `service.py` | Asset preparation and guarded local operations |
| `glm53_setup/validation/` | Explicit CPU/GPU inspection, fixture creation and assessment |
| `glm53_setup/runtime/` | Source-pinned NoPE adaptation and candidate-preserving reference calculation |
| `config/` | Model/image pins; no credentials or measured site configuration |
| `examples/` | Site template containing illustrative values only |
| `docker/` | Image construction; base digest supplied from the lock by the build command |
| `requirements/` | Fixed host-tool dependencies |
| `tests/`, `tools/` | CPU contracts and publication checks |
| `LICENSES/` | Preserved upstream license texts |
| `state/`, `records/` | Local mutable state and experiment evidence, excluded from distribution |

The CLI imports GPU dependencies only when the selected command actually needs them. Help, configuration and CPU tests work without Torch or vLLM installed on the host. GPU programs execute inside the pinned image.

Model ID and revision have one configuration source: [runtime.lock.json](../config/runtime.lock.json). Mutable files stay rooted at the checkout, independently of the caller's working directory. Run the toolkit from a maintained checkout; it is not offered as a general Python library.

The reference image modifies two vLLM source files only after checking their full hashes. It keeps all selected attention candidates. The runtime math and the validation harness are separate modules so moving CLI code does not change the mathematical implementation.

## Validation boundaries

Download completion, checksum success, GPU smoke, config interpretation, attention parity, fixture integration and full-model TP=2 qualification are different evidence types. A result from one level cannot substitute for another. In particular, the one-GPU fixture cannot open the TP=2 launch gate.

## Migrating the earlier development layout

Root-level scripts have been replaced by `python -m glm53_setup <command>`. The lock moved to `config/`, site examples to `examples/`, fixed dependencies to `requirements/`, and the Dockerfile to `docker/`. Local `state/` and `records/` keep their existing locations. Historical run commands remain historical evidence; new runs use the current CLI and a fresh output directory.
