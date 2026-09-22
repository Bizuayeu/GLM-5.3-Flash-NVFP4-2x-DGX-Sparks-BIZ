# Third-party components and licensing

Original project code is licensed under [Apache-2.0](LICENSE). This does not replace upstream licenses. Preserve [NOTICE](NOTICE) and applicable license texts when distributing derived code or images.

For commercial use, modification and distribution obligations by artifact, see [the licensing guide](docs/licensing.md) ([日本語](docs/licensing.ja.md)).

| Component | License / source | Treatment |
|---|---|---|
| NVIDIA GLM-5.3-Flash-NVFP4 weights | MIT stated in the [pinned model card](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/423acf37583782c51c142d145aef733d72943d93/README.md) | Downloaded separately; not redistributed here |
| Upstream Z.AI GLM model notice | [MIT text](LICENSES/ZAI-GLM-MIT.txt), [source revision](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/LICENSE) | Upstream notice for operator reference; the NVIDIA snapshot has no standalone LICENSE file |
| LPA training sources: LLM-jp Corpus v3 | Japanese/English Wikipedia: CC-BY-SA-3.0; C++: per-repository permissive metadata. Source URLs, revision and attribution in [projector lock](config/lpa-projector.lock.json) | Dataset/captures not redistributed; auxiliary projector offered separately under Apache-2.0. See [license scope](docs/licensing.md#lpa-projector) |
| vLLM | [Apache-2.0](LICENSES/vllm-Apache-2.0.txt), [pinned source](https://github.com/vllm-project/vllm/tree/385dce36bcee42309924a5ece951a96db3dce7f2) | Official image pinned by digest; two source files (`glm5next/nvidia/kda.py`, `model.py`) adapted and shipped as `overlays/` (modification notices inside the files and in `overlays/README.md`), mounted over the reference image |
| FreedomBench | [Apache-2.0](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/LICENSE), Lore Hex Corp | Choice extraction and prompt/option construction adapted in the local evaluator; question data acquired separately at the pinned hash; see NOTICE |
| NoPE zero-padding recipe | [MIT, kingjones30 / Jones Lab](LICENSES/kingjones-MIT.txt) | Follows the recipe preserved in [amasu's pinned patch](https://github.com/amasu/glm53-flash-cluster/blob/0ab7ca7cb1067d4fcece9d525e5d90a9bbe33773/docker/labbuild/patch_mla.py) |
| NoPE patch structure | [Apache-2.0, amasu](LICENSES/amasu-Apache-2.0.txt), same pinned patch above | Adapted zero-padding patch structure with changed guards/reference fallback; upstream license and attribution retained |
| CUDA, FlashInfer, Torch, NCCL and other dependencies | Respective upstream licenses and image notices | Existing notices remain in the image; not all dependencies are Apache/MIT |

The NoPE adaptation zero-pads the unsupported positional portion and uses a candidate-preserving eager reference calculation. The recipe's candidate-removal portions are **not included**. Source hashes are checked before applying the patch; the image retains a manifest of modified-file hashes.

Project notices are included under `/opt/glm53/`, with upstream texts in `/opt/glm53/LICENSES/`.

## Intentionally absent

- Mia's current AGPL distribution is not a dependency.
- EXL3/TR3 weights with ShapleyMCG terms are not used.
- DFlash2 draft weights with non-commercial/no-derivatives terms are not used.

These are dependency choices, not claims about every possible use of those projects. A similarly named model or image is not automatically covered by this license.

## Distribution boundary

This repository distributes setup code, tests, pinned references and reviewed summaries through Git, and the optional LPA projector through a separate Release asset. The source archive contains no weights. The NVIDIA checkpoint, credentials, local site settings and raw private logs are not redistributed. Redistributors of built containers or weights must preserve the terms and notices applicable to those artifacts.
