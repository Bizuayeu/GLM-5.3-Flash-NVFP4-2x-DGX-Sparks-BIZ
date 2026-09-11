# 実行環境の出所

実行候補の固定値は [runtime.lock.json](runtime.lock.json)。このリポジトリにモデル重みやCUDAバイナリを同梱しない。

| 対象 | 出所・条件 | 今回の扱い |
|---|---|---|
| NVIDIA GLM-5.3-Flash-NVFP4 | [固定モデルカード](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/423acf37583782c51c142d145aef733d72943d93/README.md)、MIT表記 | 公式cacheから読み取り。重みの再配布なし |
| vLLM | [公式source commit](https://github.com/vllm-project/vllm/tree/385dce36bcee42309924a5ece951a96db3dce7f2)、Apache-2.0 | 公式コンテナをdigest固定。未改変 |
| CUDA等コンテナ内依存 | 公式イメージに含まれる各ライセンス・通知 | イメージ内の通知を保持。全依存がApache/MITとは主張しない |

Miaの現行AGPL実装、EXL3/TR3重み、DFlash2重みは実行環境に追加していない。workspaceなどMia固有の候補パッチは未採用。

## 参照attention用の候補イメージ

`Dockerfile.reference`は固定公式baseにNoPE用のゼロ埋めと、全候補を保持する参照attentionを追加する。`patch_nope_reference.py`のゼロ埋め適合は、[kingjones30のMIT版](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark)を出所とするamasuの[固定patch](https://github.com/amasu/glm53-flash-cluster/blob/0ab7ca7cb1067d4fcece9d525e5d90a9bbe33773/docker/labbuild/patch_mla.py)を参考にしている。top-k候補削減部分は採用しない。

元の[MIT通知](LICENSES/kingjones-MIT.txt)と[vLLM Apache-2.0通知](LICENSES/vllm-Apache-2.0.txt)を保持し、イメージの`/opt/glm53/LICENSES`にも配置する。変更対象の元source hashはパッチ内、変更後hashはイメージ内`glm53-reference-patch.json`に記録する。現在のGPU検証はattention単体とbackend経由までで、実モデル全体は未検証。
