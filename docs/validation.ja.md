# 検証範囲

[English](validation.ja.md)

## 証拠であり、本番認定ではない

以下の観測は、GB10 GPU、vLLMのsource commit `385dce36bcee42309924a5ece951a96db3dce7f2`、NVIDIAのモデルrevision `423acf37583782c51c142d145aef733d72943d93` で取得しました。非公開の生runは配布せず、本書はそれをレビューした要約です。

| 試験 | 観測結果 | 限界 |
|---|---|---|
| 実packed FP8 cacheでの参照Attention | 候補幅63／64／65／2048／2051／2176を検査。padding・空行を処理し、意図的なtail除去も検出 | 部品の検査であり、モデル全体の正当性ではない |
| 4層checkpoint | 選択した3,591件のtensorをすべてバイト単位で検証。元の幅と288 expertsを保持 | 言語品質のベンチマークではない |
| Marlin W4A16、GPU 1台 | ロード、生成、A→B→Aの再実行、2要求batchのtoken比較、prefill強制の試験に合格 | 入力は限定的で、本番信頼性の主張ではない |
| Marlin、8,705 tokenの入力 | 実測した8,704 tokenのattention manager blockを跨いだ。prefill強制の次tokenは一致し、選択したlogprobの差は0.0031653 | すべての境界やcontext容量の上限を網羅しない |
| CUTLASS W4A4 | 生成は完了したが、数値不変性の検査で差が出た | 原因は切り分けていない |
| batch不変モード | SM120の疎MLAが拒否。Triton MLAは疎に未対応 | 固定した本スタックでは使えない |

確率の許容差は、参照logprobの大きさに対するBF16イプシロン2つ分という暫定の上限でした。厳密な再実行、tokenの一致、許容差による比較は、それぞれ別の検査です。層を切り詰めたモデルは、数値差を拡大することがあります。

パッケージ化したCLIと再構成したDocker buildもGB10で確認しました。実cacheの部品試験と、context 16,384でのMarlin 4層試験は、8,705 tokenの境界入力を含めて合格しました。両試験containerともOOMなしで正常終了しました。これが検証するのは新しいパッケージ・workerのimport経路であり、run間のビット一致やフルモデルTP=2ではありません。

Marlinは演算そのものを変えます。[linear kernel](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/kernels/linear/nvfp4/marlin.py)はW4A16で、[MoEのselector](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py)は汎用の `use_a16` フラグとは独立にMARLIN向けのW4A16を選びます。このフラグだけから精度を推定しないでください。

vLLMは[既定での再現性を保証していません](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/usage/reproducibility.md)。ただし、これは本リポジトリのアダプタが正しいことの証明にもなりません。W4A4の差は、引き続き調査が必要な観測のままです。

## GPU 1台のfixtureを再現する

Linux版のGB10ホストと、検証済みのcheckpointを使います。checkoutのルートから実行してください。例は既定のHugging Face cacheを前提とするため、環境が異なる場合はホスト側のmountを調整します。

~~~sh
python -m glm53_setup build-reference
mkdir -p state records/fixture-check state/fixture-cache
IMAGE=glm53-enterprise:reference
HF_CACHE=$HOME/.cache/huggingface
REVISION=$(python -c 'from glm53_setup.config import REVISION; print(REVISION)')
~~~

元のcacheを変更せずに読み取り、別の4層checkpointを作ります。fixtureにはおよそ7.46 GiB分のtensorが入ります。

~~~sh
docker run --name glm53-fixture-build --network none --memory 24g --memory-swap 24g -v "$HF_CACHE:/hf:ro" -v "$PWD/state:/data" --entrypoint python3 "$IMAGE" -m glm53_setup fixture-build --source "/hf/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/$REVISION" --output /data/four-layer
~~~

新しい実験では、出力ディレクトリとcontainer名を新規に用意します。既存のfixture出力を上書きすることはありません。

~~~sh
docker run --name glm53-fixture-check --gpus all --network none --memory 32g --memory-swap 32g --shm-size 2g -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_HOST_IP=127.0.0.1 -e GLOO_SOCKET_IFNAME=lo -e NVIDIA_TF32_OVERRIDE=0 -v "$PWD/state/four-layer:/fixture:ro" -v "$PWD/records/fixture-check:/out" -v "$PWD/state/fixture-cache:/root/.cache" --entrypoint python3 "$IMAGE" -m glm53_setup fixture-run --fixture /fixture --output /out --backend marlin --context 16384 --chunk 512
python -m glm53_setup fixture-assess records/fixture-check
~~~

24／32 GiBの予算は試験用の上限であり、フルモデルの必要量ではありません。実験には外部で期限を設け、超過した場合は該当する試験containerを停止します。過去の試験では15分を使いました。containerと結果は保持します。

診断のために比較する場合は、新しいrunで `--backend auto` または `--chunk 128` を使います。生成がすべて完了していても、`passed=false` は数値上の合格ではありません。

## CPUと部品の検査

~~~sh
python -m unittest discover -s tests -t . -v
python tools/check_publication.py
~~~

CPU側の検査は、GPU importなしのCLI振り分け、checkout基準の資材、revision・起動のガード、fixtureの選択、結果の判定を対象とします。CPUのCIはGPU試験を実行せず、重みもダウンロードしません。

CLIは `inspect-runtime`・`probe-attention`・`test-reference` も提供します。reference imageの中で、それぞれの `--help` を参照してください。これらの部品検査は、TP=2の検収の代わりにはなりません。

## 残る検収項目

[FreedomBench・政治的文脈の評価](freedombench.ja.md)は、エンタープライズ向けの必須評価項目です。英語原版の1構成には予備実測がありますが、4構成の一覧、日本語・長文への拡張、人手監査は未検収のままです。結果とそのLPA迂回の制約は、リンク先の文書が正典です。

公式ZCodeとClaude Code CLIは、[ハーネス受け入れ一覧](harnesses.ja.md)における別々の必須対象です。両者の端から端までのケースはNOT RUNです。後述の基礎APIスモークは、一覧全体もクライアント連携のケースも終わらせません。

[2台でのNCCL通信検証](nccl-validation.ja.md)は、固定したbase imageで試験したcollectiveのパターンに合格しています。その範囲はtransportと合成データの正当性であり、参照Attentionやフルモデルとは別です。

[同時2系列の独立評価](benchmarks.ja.md#標準batchingの独立評価)には、限定した課題・throughput・16K×2容量の結果があります。より広いフルモデルの数値・品質評価、持続的な混在負荷、本番復旧、batchingの組合せ、k=1／k=3以外のMTP深さ、Graphs、画像入力は未検証のままです。prefix cachingには[範囲を限定した独立の結果](benchmarks.ja.md#全モデルのprefix-caching独立評価p19)があり、APC／LPAとMTPの併用は別の[P22の契約](apc-lpa-design.ja.md)に従います。fixture、APIスモーク、collectiveの結果を、無制限の `tp2-kernel-validation` 証跡に変えないでください。

## フルモデルTP=2の実験範囲

reference imageは、2台のGB10ホストで45層の言語層すべてを、Marlin W4A16、eager実行、同時1系列、context 16,384、rankあたり1 GiBのKVでロードしました。直列TP=2の4層fixtureは、既存の状態検査をすべて通過しました。fixtureを同時2系列にした場合、greedy経路の1本がほぼ同値の箇所で分岐しました。この生の診断は失敗のままであり、課題水準の受け入れとは別です。

フルモデルは、served IDの基礎確認、英語・日本語の最終回答、OpenAIのSSE、無害な自動ツールの呼出し・引数・戻り、AnthropicのMessages／count_tokensのスモーク検査に合格しました。reasoning effortを低くしたチャットの受け入れ試験も、これらの最終回答・ツールの基準を満たしました。再実行では推論文が異なりました。これは診断として残すものであり、自由記述の逐語一致を要求するものではありません。未対応のthinking offを指定した要求ではparserと本文が混ざったため、受け入れ済みの構成ではありません。[ハーネスの設定](harnesses.ja.md)を参照してください。

[vLLM公式の合成ベンチマーク](benchmarks.ja.md)は、計画した計測要求をすべて完了しました。試験containerはその後停止しました。これらの結果は、現行の通常ランチャーを解放するものでも、アプリケーション全体の品質を示すものでも、本番デプロイを認定するものでもありません。

その後、[MTP k=1の候補](speculative-decoding.ja.md)は、小規模なロードfixture、同条件のフルモデルベンチ5ケースすべて、同じ11項目の基礎API検査に合格しました。別のメタデータviewは、元のcheckpointを保ったまま、そのBF16 MTP層を全体のNVFP4から除外します。手順書には、draftの受理率、追加メモリ、負荷に依存する改善、未解決の分散停止の制約を記載しています。実際のZCode・Claude Codeの受け入れは、引き続き別扱いです。

同じフルモデルのベンチとAPIのケースは、k=3でも合格しました。[k=3の比較結果](speculative-decoding.ja.md#k3の比較結果)には、位置ごとの受理率と実効的なcache整列を記録しています。今後の実験評価ではk=3を優先しますが、最適な投機深さを示すものではなく、残るゲートを閉じるものでもありません。
