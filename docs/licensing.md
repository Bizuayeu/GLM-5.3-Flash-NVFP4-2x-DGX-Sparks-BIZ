# Use, modification and redistribution

[日本語](licensing.ja.md) · [Provenance and notices](../THIRD_PARTY_NOTICES.md)

**Original repository code and documentation are Apache-2.0: commercial use, modification, and paid or free redistribution are permitted subject to its conditions. Weights, containers and harnesses retain separate terms.** This is an operational summary; the applicable license text and artifact version govern.

## Permissions by artifact

| Artifact | Business use / paid service | Private modification | Redistribution / sale | Obligations |
|---|---|---|---|---|
| Original setup code and docs | Permitted | Permitted; no general source-publication requirement | Permitted in source or object form | Apache text, applicable notices, prominent modified-file notices |
| vLLM and adapted files | Permitted under Apache-2.0 | Same | Conditional permission | Preserve upstream notices and mark changes |
| MIT portions from kingjones | Permitted | Permitted; no publication requirement | Permitted, including sale/sublicensing | Retain copyright and permission notice in copies/substantial portions |
| Pinned NVIDIA GLM NVFP4 weights | Model card explicitly permits commercial/non-commercial use under MIT | MIT permits modification, including further training/conversion | Permitted subject to MIT notices | Preserve model card and upstream copyright/MIT notice; identify provenance |
| Complete image containing CUDA and other dependencies | Subject to each component's terms | Do not treat SDK modification rights as Apache | **No blanket clearance for the image** | Review the actual component inventory, versions, redistributables and notices |
| ZCode / Claude Code binaries | Subject to their product/service terms | No modification rights granted by this repository | No redistribution rights granted by this repository | Install separately from official sources; do not bundle credentials |

Sources: [Apache sections 2–4 and 6–9](../LICENSE), [MIT text](../LICENSES/kingjones-MIT.txt), [pinned NVIDIA model card](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/423acf37583782c51c142d145aef733d72943d93/README.md).

## MIT and Apache-2.0 compared

Both are permissive: commercial use, modification, redistribution and inclusion in closed products are permitted, and neither requires publishing modified source for internal use or hosted services. They differ in three places.

| Point | MIT | Apache-2.0 |
|---|---|---|
| Patents | Silent; no express patent grant | Express patent license from contributors, terminated for a party that starts patent litigation over the work |
| Changed files | Keep the copyright and permission notice | Additionally mark each modified file as changed (this project's vLLM patching writes that comment into the patched source) |
| Trademarks | Not addressed | No trademark rights are granted |

Original code here is Apache-2.0 because the express patent grant and the change-marking rule make review easier for adopting organizations. The weights are MIT because that is what the upstream Z.ai model and the NVIDIA model card state; this project does not relicense them.

## Distributing Apache-covered material

Supply the license, retain applicable upstream notices, and reproduce NOTICE attribution readably. **Mark each modified file as the table above describes; a changelog alone does not replace that requirement.** Different terms for your modifications or a closed product can coexist with retained upstream obligations.

Warranty is disclaimed; paid support can be offered on your own responsibility. Private use or providing an API does not itself create a general Apache/MIT source-publication obligation; delivering artifacts to customers triggers their distribution conditions.

## Weight notices

The pinned NVIDIA snapshot has no standalone LICENSE file; its README states MIT and commercial eligibility. We preserve the [upstream Z.AI license](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/LICENSE) as [LICENSES/ZAI-GLM-MIT.txt](../LICENSES/ZAI-GLM-MIT.txt), without claiming that file came from the NVIDIA snapshot.

When redistributing weights, attach the model card and upstream MIT notice, identify the source/quantization/revision, and account for any additional code or training data you used. Place notices outside the exact Hugging Face snapshot so its official checksum/extra-file check remains reproducible. MIT does not guarantee output ownership, non-infringement or rights to your input data.

This project applied those rules once: the attention and `lm_head` W4A16 repack that the reference pair serves is published as [Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16) under MIT, with NVIDIA's model card kept beside it as `README.nvidia.md`, the Z.AI notice referenced, the source revision, tool commit and target recorded in its card and in `config.json`, and no training data added (weight-only conversion). The conversion tool's Apache-2.0 terms apply to the tool, not to the weights.

## LPA projector

The independently fitted [LPA cut32 auxiliary weights](lpa.md#download-the-trained-projector) are offered under **Apache-2.0**, to the extent of the maintainer's rights. The Release package includes LICENSE, NOTICE, teacher provenance and training-data attribution. This does not relicense NVIDIA's checkpoint or the training dataset.

The [projector lock](../config/lpa-projector.lock.json) records the sampled LLM-jp Corpus v3 subsets: Japanese/English Wikipedia labeled CC-BY-SA-3.0, and C++ filtered by per-repository MIT/Apache/BSD/ISC metadata. LLM-jp provenance does not make every underlying text Apache-2.0. Corpus text and teacher captures are not distributed with the projector.

The Apache offer covers the auxiliary artifact, not third-party expression that might be reproduced by a model. Whether training or an artifact requires permission for such expression is a separate, fact-dependent question: [Creative Commons' guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) distinguishes copyright exceptions from uses that trigger its conditions. The license choice is not a legal finding that those conditions can never apply.

## Images and harnesses

The recommended distribution is **source, pinned references and build instructions**, with users obtaining weights, the official base and harnesses separately. Redistributing a completed image requires checking its actual dependencies, including CUDA redistributables and OS copyleft components. The [CUDA 13.0 SDK agreement](https://docs.nvidia.com/cuda/archive/13.0.0/eula/index.html) is one relevant source, not a substitute for every dependency's terms.

ZCode follows its [terms](https://zcode.z.ai/en/terms); Claude Code follows the agreement referenced by its [LICENSE](https://github.com/anthropics/claude-code/blob/main/LICENSE.md). Apache-covered integration configuration does not relicense those applications. Local model licensing and hosted cloud-plan terms are separate.

## Distribution checklist

- [ ] Identify whether the deliverable contains source, weights, images or harnesses.
- [ ] Keep LICENSE, NOTICE and applicable LICENSES texts.
- [ ] Mark modified files and record provenance/revisions.
- [ ] For weights, attach model/MIT notices and review added materials.
- [ ] For images, review the actual component inventory and redistribution terms.
- [ ] Exclude harness binaries, keys, credentials and private logs unless separately authorized and licensed.

Connection design and technical acceptance are covered in [harness integration](harnesses.md).
