# Architecture

[日本語](architecture.ja.md)

The project is a checkout-local operator toolkit. It does not contain model weights or a remotely managed service.

| Location | Responsibility |
|---|---|
| `glm53_setup/__main__.py` | Fixed command dispatch; no dynamic user-supplied module loading |
| `glm53_setup/config.py` | Checkout paths and validated pinned configuration |
| `glm53_setup/startup.py`, `startup_config.py` | Experimental launch/client orchestration and categorized TOML settings |
| `glm53_setup/download.py`, `images.py`, `build_reference.py`, `service.py` | Asset preparation and guarded local operations |
| `glm53_setup/validation/` | Explicit CPU/GPU inspection, fixture creation and assessment |
| `glm53_setup/runtime/` | Source-pinned NoPE adaptation and candidate-preserving reference calculation |
| `glm53_setup/runtime/lpa.py`, `lpa_query.py` | LPA worker control, attention-input approximation and request-scoped query omission |
| `glm53_setup/validation/run_lpa.py`, `lpa_corpus.py`, `train_lpa.py` | LPA fixture verification, corpus preparation and projector fitting |
| `config/` | Model/image pins; no credentials or measured site configuration |
| `examples/` | Site template containing illustrative values only |
| `docker/` | Image construction; base digest supplied from the lock by the build command |
| `requirements/` | Fixed host-tool dependencies |
| `tests/`, `tools/` | CPU contracts and publication checks |
| `LICENSES/` | Preserved upstream license texts |
| `state/`, `records/` | Local mutable state and experiment evidence, excluded from distribution |

The CLI imports GPU dependencies only when the selected command actually needs them. Help, configuration and CPU tests work without Torch or vLLM installed on the host. GPU programs execute inside the pinned image.

The commented `examples/startup.example.toml` doubles as the complete startup schema. Full TOML validation happens at `startup_config.load` and the independently callable `startup.command` boundary. `serve_args` consumes an already validated profile and does not reread the schema; it is an internal assembly step, not an input-validation entry point. Small fabric-specific guards remain independent.

Model ID and revision have one configuration source: [runtime.lock.json](../config/runtime.lock.json). Mutable files stay rooted at the checkout, independently of the caller's working directory. Run the toolkit from a maintained checkout; it is not offered as a general Python library.

The reference image modifies two vLLM source files only after checking their full hashes. It keeps all selected attention candidates. The runtime math and the validation harness are separate modules so moving CLI code does not change the mathematical implementation.

## Validation boundaries

Download completion, checksum success, GPU smoke, config interpretation, attention parity, fixture integration and full-model TP=2 qualification are different evidence types. A result from one level cannot substitute for another. In particular, the one-GPU fixture cannot open the TP=2 launch gate.
