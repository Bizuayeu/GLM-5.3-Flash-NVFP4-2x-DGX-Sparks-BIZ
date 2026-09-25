# Indexerの再利用と候補限定の再採点

[English](indexer-reuse.md)

## 結果：コストの門で中止（2026-09-21）

**実験用の部品であり、モデルのservingには未統合です。** 検証済みの4層fixture（indexer層は1つ、eager、chunk 512、出力1 token）で、indexer自身の演算はprefill 501 msのうち0.43 ms（2,048 token）、1,999 msのうち6.26 ms（8,192）、8,077 msのうち40.2 ms（32,768）＝0.09%・0.31%・0.50%で、prefillが線形に伸びる間にn^1.5程度で伸びます。全モデルには同じ形のindexer層が11あるので、indexer全体は32Kのprefillの約1.7%、200Kで約4%、再利用が削れる採点・選択はその半分ほどで、圧縮keyの書き込みとtailの更新は残ります。再利用の仕組みに見合う改善に届かないため、再利用は作りませんでした。設計ではコスト、overlap、正しさ、速度、課題の順に門を置いており、最初のコストで止まりました。それ以前の2K／8Kのoverlap観測は[部品検証](component-validation.ja.md#indexerの観測)にあります。

## 再利用が保つ必要のあったもの

- NVIDIAの設定では全層の `indexer_types` が `full` で、`index_share_for_mtp_iteration=true` はdraftの反復にだけ適用され、target層には適用されません。
- [kpool indexer](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/layers/sparse_attn_indexer_kpool.py)は `topk_tokens // index_kpool` 個のpoolを選び、論理tokenの候補と未完のtailへ展開します。kpool=4で2,048 tokenの予算は512 poolで、2,048 poolではありません。
- indexerは圧縮keyの書き込みとtail状態の初期化・更新も行い、後のnative decodeがそれを使います。[モデル](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py)は共有のTop-K bufferをMLA層へ渡すので、選択された行は別の層が上書きする前にsnapshotする必要があります。物理cacheのアドレスは層・要求・rankをまたいで渡しません。
- 候補IDの複写はKVの共有ではありません。GLMの各層は固有の学習済みprojectionとpool gateを持ちます。[IndexCache](https://arxiv.org/abs/2603.12201)とその[参照patch](https://github.com/THUDM/IndexCache)（対象はGLM-5の `GlmMoeDsaForCausalLM` で、この `Glm5Next` ではない）と[ReTopK](https://arxiv.org/abs/2607.27692)が動機でしたが、その数値はGLM-5.3-Flashの結果ではありません。

## 保持した部品

- `runtime/indexer_capture.py`（`runtime/indexer_worker.py` がLPA／MTPと独立に取り付ける）：束ねたkpoolモジュールで、範囲を限定したCUDA eventの計時と論理候補のcaptureを行い、バイト数・event数に上限があります。4層fixtureの2K／8Kの実行ではlayer 3の候補を採取し、native／capture／復帰後の出力tokenは一致しました。
- `runtime/indexer_candidates.py`：要求内の選択snapshot、層対の検証、候補だけを対象とするFP32のscore参照実装、完全なpoolと未完tailの展開。同点は小さいpool IDを採ります。
- `runtime/indexer_reindex.py`：固定版のFP8・32 head／128特徴の契約で、与えたpoolを採点する単体のTriton実装。候補1／17／128件で独立なFP64の密oracleと照合しました（`rtol=2e-5`、`atol=2e-4`）。
- `runtime/indexer_shared_pool.py`：共有候補poolを、target自身のK／scaleで固定版のnative Tensor Core採点器により採点します。

query 512件・選択pool 1,024個では、共有pool経路は約0.129 msで、nativeによる8,192 pool（kpool=4で約32K token）の採点0.234 msより速い一方、2,048 pool（約8K）の0.068 msに対しては悪化でした。`benchmark_reindex` がこれらを記録し、選択されたscoreをnativeと照合します。queryごとに採点する最初のTriton試作は正しかったものの、nativeより大幅に遅いものでした。CPUの `indexer-overlap` コマンドは、揃えた `source`／`target` 行（`request_id`・`query_position`・`coordinate_space="logical_tokens"`・`indices`、source側の任意の `candidate_pool`）を比較し、整列していない入力、因果的でない入力、物理slotの入力は拒否します。

観測器の確認は、現在のソースをマウントした固定版GPUイメージ内で、LPA・MTP・prefix cachingを無効にして `python -m glm53_setup.validation.run_indexer_fixture --fixture /verified/four-layer-fixture --output /new/record` を実行して再現します。
