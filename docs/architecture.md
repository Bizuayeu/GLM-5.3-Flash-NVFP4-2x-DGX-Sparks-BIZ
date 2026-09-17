# Architecture

[日本語](architecture.ja.md)

The project is a checkout-local operator toolkit. Source archives contain no weights or remotely managed service. The optional LPA weights are a separate Release asset; [operations](operations.md#artifact-storage-and-paths) owns the package layout and installation paths.

| Location | Responsibility |
|---|---|
| `glm53_setup/__main__.py` | Fixed command dispatch; no dynamic user-supplied module loading |
| `glm53_setup/config.py` | Checkout paths and validated pinned configuration |
| `glm53_setup/server.py`, `server_config.py`, `capacity.py`, `warmup.py`, `mojibake.py` | Launch/client orchestration, categorized TOML settings, KV boot-line decomposition, the post-readiness request ladder and the Japanese/Korean broken-character check |
| `glm53_setup/host.py` | Host-side helpers shared by the launcher: site validation, serve arguments, fabric checks, snapshot resolution, subprocess execution |
| `glm53_setup/download.py`, `images.py`, `build_reference.py` | Asset preparation and guarded local operations |
| `glm53_setup/validation/` | Explicit CPU/GPU inspection, fixture creation and assessment |
| `glm53_setup/runtime/` | Source-pinned NoPE adaptation and candidate-preserving reference calculation |
| `glm53_setup/runtime/lpa.py`, `lpa_query.py` | LPA worker control, attention-input approximation and request-scoped query omission |
| `glm53_setup/validation/run_lpa.py`, `lpa_corpus.py`, `train_lpa.py` | LPA fixture verification, corpus preparation and projector fitting |
| `config/` | Model/image pins and `lpa-projector.lock.json` (Release URL, checksum, teacher and training provenance); no credentials or measured site configuration |
| `examples/` | Server profile and MTP speculative templates containing illustrative values only |
| `docker/` | Image construction; base digest supplied from the lock by the build command |
| `requirements/` | Fixed host-tool dependencies |
| `glm53_setup/validation/freedombench.py`, `freedom_scoring.py`, `apc_history.py`, `profile_trace.py`, `benchmark_*.py` | FreedomBench runner and scoring, APC history regression, trace event accounting and component benchmarks |
| `glm53_setup/runtime/candidate_order.py` | Canonical logical candidate order at the shared sparse-MLA boundary ([candidate order](candidate-order.md)) |
| `glm53_setup/runtime/apc_policy.py`, `apc_runtime.py`, `apc_worker.py`, `patch_apc_lpa.py` | APC-first LPA admission, exact-only prefix publication and worker dispatch ([design contract](apc-lpa-design.md)) |
| `glm53_setup/runtime/fused_unpack.py`, `fused_nope*.py`, `graph_policy.py`, `patch_graph_prefill.py` | Fused FP8 unpack kernel, experimental fused NoPE attention prototypes and the decode-Graph policy |
| `glm53_setup/runtime/indexer_*.py`, `component_worker.py` | CSA2 indexer observation and reuse components, and the exclusive component diagnostics worker ([indexer reuse](indexer-reuse.md)) |
| `glm53_setup/runtime/pipeline_state.py`, `patch_pipeline*.py` | PP fixture transport and layout patches (P17) |
| `glm53_setup/cluster.py`, `switch.py`, `launch_assets.py`, `fabric.py` | Two-rank pre-stop checks, owned switch/recovery transaction, read-only launch identities and RoCE rail checks ([launch contracts](launch-safety.md)) |
| `glm53_setup/model_http.py`, `io.py` | Model-API-scoped HTTP transport that never follows redirects; durable local state helpers |
| `tests/` | CPU contracts |
| `tools/` | `check_publication.py` (publication audit), `assess_benchmark.py`, `check_prefix_cache.py`, `nccl_probe.py`, `prepare_mtp_view.py` |
| `examples/zcode-hooks/` | ZCode existing-file guard hook and its setup ([harnesses](harnesses.md)) |
| `LICENSES/` | Preserved upstream license texts |
| `state/`, `records/` | Local mutable state and experiment evidence, excluded from distribution |

The CLI imports GPU dependencies only when the selected command actually needs them. Help, configuration and CPU tests work without Torch or vLLM installed on the host. GPU programs execute inside the pinned image.

The commented `examples/server.example.toml` doubles as the complete server profile schema. Full TOML validation happens at `server_config.load` and the independently callable `server.command` boundary. `serve_args` consumes an already validated profile and does not reread the schema; it is an internal assembly step, not an input-validation entry point. Small fabric-specific guards remain independent.

Model ID and revision have one configuration source: [runtime.lock.json](../config/runtime.lock.json). Mutable files stay rooted at the checkout, independently of the caller's working directory. Run the toolkit from a maintained checkout; it is not offered as a general Python library.

The reference image modifies two vLLM source files only after checking their full hashes. It keeps all selected attention candidates. The runtime math and the validation harness are separate modules so moving CLI code does not change the mathematical implementation.

## Validation boundaries

Download completion, checksum success, GPU smoke, config interpretation, attention parity, fixture integration and full-model TP=2 qualification are different evidence types. A result from one level cannot substitute for another. In particular, the one-GPU fixture cannot stand in for two-rank evidence.
