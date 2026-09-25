# Architecture

[日本語](architecture.ja.md)

The project is a checkout-local operator toolkit. Source archives contain no weights or remotely managed service. The optional LPA weights are a separate Release asset; [operations](operations.md#artifact-storage-and-paths) owns the package layout and installation paths.

## Where to start reading

Four kinds of code, distinguished by what a test can do with them rather than by any framework:

| | What it holds | Named seams |
|---|---|---|
| **Entry** | The argument interface, and which handler an action reaches | `server.ACTIONS`, `cluster.ACTIONS`, `__main__.COMMANDS` |
| **Assembly** | The order steps run in, and what a failure rolls back | `switch.switch`, `server.act_launch`, `cluster.act_switch` |
| **Decision** | Settings validated, arguments built, reports shaped — pure functions | `server_config.VALIDATORS`, `server_config.SERVE_STEPS`, `validation.*.engine_kwargs`, `runtime.apc_policy`, `runtime.patch_*.patch_text` |
| **Side effect** | docker, HTTP, subprocess, torch, vLLM | `host.run`, `model_http`, the lazy GPU imports inside each runner |

The decision layer is where the numbers live that make one measurement comparable to the next, so it is the layer kept importable without torch or vLLM: `engine_kwargs(args)` states what a fixture runner launches under, on a host that cannot run it. Entry and assembly take their side effects as arguments, so a test substitutes them; the side-effect layer is the substitution point, not the thing under test.

Order is part of the contract in two places. `VALIDATORS` runs the profile rules in a fixed sequence because the first raise is the sentence the operator reads. `SERVE_STEPS` writes into one argument list and later steps index into what earlier ones left.

| Location | Responsibility |
|---|---|
| `glm53_setup/__main__.py` | Fixed command dispatch; no dynamic user-supplied module loading |
| `glm53_setup/config.py` | Checkout paths and validated pinned configuration |
| `glm53_setup/server.py`, `server_config.py`, `capacity.py`, `warmup.py`, `mojibake.py`, `agreement.py` | Launch/client orchestration, categorized TOML settings, KV boot-line decomposition, the post-readiness request ladder, the Japanese/Korean broken-character check and the per-token agreement with a reference run (`server agreement`) |
| `glm53_setup/host.py` | Host-side helpers shared by the launcher: site validation, serve arguments, fabric checks, snapshot resolution, subprocess execution |
| `glm53_setup/download.py`, `verify_download.py`, `images.py`, `build_reference.py` | Asset preparation (pinned download, checksum verification that waits for the downloader, base-image inspection, reference-image build) and guarded local operations |
| `glm53_setup/cluster.py`, `switch.py`, `launch_assets.py`, `fabric.py` | Two-rank pre-stop checks, owned switch/recovery transaction, read-only launch identities and RoCE rail checks ([launch contracts](launch-safety.md)) |
| `glm53_setup/model_http.py`, `io.py` | Model-API-scoped HTTP transport that never follows redirects; durable local state helpers |
| `glm53_setup/runtime/pinned_patch.py`, `patch_*.py` | The source-pinned vLLM patches the image build applies: `pinned_patch` holds what they share (the hash gate on the pinned file, the `--package`/`--check` command, the record written beside the package); each `patch_*` module states its target, its pin and its anchors as a pure `patch_text(text)` that refuses a drifted or already patched source |
| `glm53_setup/runtime/reference_attention.py`, `patch_nope_reference.py`, `fa2_attention.py` | Candidate-preserving eager NoPE MLA reference, its source-pinned installation, and the FA2 path for calls of more than six rows (`runtime.fa2_attention`) |
| `glm53_setup/runtime/candidate_order.py` | Canonical logical candidate order at the shared sparse-MLA boundary ([candidate order](candidate-order.md)) |
| `glm53_setup/runtime/moe_token_order.py`, `patch_moe_order.py` | One token order inside each expert before the Marlin MoE kernel (`runtime.canonical_moe_order`) and its source-pinned patch |
| `glm53_setup/runtime/stable_topk.py`, `patch_indexer_topk.py` | The kpool indexer's top-k with ties settled (`runtime.stable_indexer_topk`) and its source-pinned patch |
| `glm53_setup/runtime/prefix_dedup.py`, `patch_prefix_dedup.py` | One cached prefix page per content (`runtime.prefix_page_dedup`) and its source-pinned patch |
| `glm53_setup/runtime/patch_slot_mapping.py` | Source-pinned patch: the slot-mapping kernel reads a block table only inside its row |
| `glm53_setup/runtime/patch_kpool_seed.py` | Source-pinned patch: the kpool prefill seed addresses tail blocks by the tail's strides (vLLM #57477) |
| `glm53_setup/runtime/mla_decode_cpb.py`, `patch_mla_decode_cpb.py` | Retired, kept until the next image build: the sparse-MLA decode's `chunks_per_block` from one sequence (`runtime.mla_decode_cpb`) and its source-pinned patch, never reached in serving because the reference attention returns first |
| `glm53_setup/runtime/inductor_pin.py`, `inductor_pin_pth.txt` | Keeps Inductor's deterministic mode on through Dynamo's state restore, which turns it off after the first compiled frame (`runtime.inductor_deterministic`); the text file, mounted as `glm53-inductor-pin.pth` in the image's site directory, runs it at interpreter start |
| `glm53_setup/runtime/lpa.py`, `lpa_query.py` | LPA worker control, attention-input approximation and request-scoped query omission |
| `glm53_setup/runtime/apc_policy.py`, `apc_runtime.py`, `apc_worker.py`, `patch_apc_lpa.py` | APC-first LPA admission, exact-only prefix publication and worker dispatch ([design contract](apc-lpa-design.md)) |
| `glm53_setup/runtime/fused_unpack.py`, `fused_nope*.py`, `graph_policy.py` | Fused FP8 unpack kernel, experimental fused NoPE attention prototypes and LPA's eager-execution requirement |
| `glm53_setup/runtime/indexer_*.py`, `component_worker.py`, `memory_probe.py` | CSA2 indexer observation and reuse components, the exclusive component diagnostics worker and the probe of a serving worker (allocator readout, weight digest, request trace, indexer kernel hashes) ([indexer reuse](indexer-reuse.md)) |
| `glm53_setup/runtime/pipeline_state.py`, `patch_pipeline*.py` | PP fixture transport and layout patches (P17) |
| `glm53_setup/validation/make_fixture.py`, `run_fixture.py`, `summarize_fixture.py`, `inspect_runtime.py`, `probe_attention.py`, `reference_check.py` | Fixture build, run and assessment, in-container inspection, the NoPE dispatch probe and reference attention parity ([validation](validation.md)) |
| `glm53_setup/validation/run_agreement_fixture.py`, `compare_agreement.py`, `quant_error.py`, `run_repeat_trace.py` | Requantization checks on the fixture and the first module that differs between repeated passes ([validation](validation.md#full-model-tp2-experimental-scope)) |
| `glm53_setup/validation/run_components.py`, `run_graph_fixture.py`, `run_indexer_fixture.py`, `run_apc_lpa_fixture.py`, `indexer_overlap.py`, `expert_worker.py`, `pipeline_worker.py`, `apc_fixture_worker.py` | Component A/B/A, Graph, indexer, APC/LPA, EP and PP fixtures and their fixture-only workers ([component validation](component-validation.md)) |
| `glm53_setup/validation/run_lpa.py`, `lpa_corpus.py`, `train_lpa.py` | LPA fixture verification, corpus preparation and projector fitting |
| `glm53_setup/validation/freedombench.py`, `freedom_scoring.py`, `apc_history.py`, `profile_trace.py`, `benchmark_*.py` | FreedomBench runner and scoring, APC history regression, trace event accounting and component benchmarks |
| `config/` | Model/image pins and `lpa-projector.lock.json` (Release URL, checksum, teacher and training provenance); no credentials or measured site configuration |
| `examples/` | Server profile and MTP speculative templates containing illustrative values only |
| `examples/zcode-hooks/` | ZCode existing-file guard hook and its setup ([harnesses](harnesses.md)) |
| `overlays/` | The two vLLM source overlays that the published option's checkpoint needs, with their manifest ([overlays/README.md](../overlays/README.md)) |
| `docker/` | Image construction; base digest supplied from the lock by the build command |
| `requirements/` | Fixed host-tool dependencies |
| `tests/` | CPU contracts |
| `tools/` | `check_publication.py` (publication audit), `release_notes.py` (the Changelog section a tag publishes), `kernel_hashes.py` (the indexer's kernels hashed inside both serving workers), `assess_benchmark.py`, `check_prefix_cache.py`, `decode_check.py`, `decode_divergence.py` and `weight_digest.py` (the decode check and the weight digest after a switch, [launch contracts](launch-safety.md#after-a-switch-the-decode-check)), `nccl_probe.py`, `prepare_mtp_view.py` |
| `.github/workflows/` | CI (CPU tests, Ruff, publication audit on Linux and Windows) and the tag-driven GitHub Release |
| `LICENSES/` | Preserved upstream license texts |
| `state/`, `records/` | Local mutable state and experiment evidence, excluded from distribution |

The CLI imports GPU dependencies only when the selected command actually needs them. Help, configuration and CPU tests work without Torch or vLLM installed on the host. GPU programs execute inside the pinned image.

The commented `examples/server.example.toml` doubles as the complete server profile schema. Full TOML validation happens at `server_config.load` and the independently callable `server.command` boundary. `serve_args` consumes an already validated profile and does not reread the schema; it is an internal assembly step, not an input-validation entry point. Small fabric-specific guards remain independent.

Model ID and revision have one configuration source: [runtime.lock.json](../config/runtime.lock.json). Mutable files stay rooted at the checkout, independently of the caller's working directory. Run the toolkit from a maintained checkout; it is not offered as a general Python library.

Every build-time patch of the reference image checks the full SHA-256 of the pinned vLLM file it modifies before changing it, and the overlays for the published option are checked the same way at launch. The image keeps all selected attention candidates. The runtime math and the validation harness are separate modules so moving CLI code does not change the mathematical implementation.

## Validation boundaries

Download completion, checksum success, GPU smoke, config interpretation, attention parity, fixture integration and full-model TP=2 qualification are different evidence types. A result from one level cannot substitute for another. In particular, the one-GPU fixture cannot stand in for two-rank evidence.
