# 検証範囲

[English](validation.md)

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
IMAGE=$(python -c 'import json; print(json.load(open("config/runtime.lock.json"))["reference_candidate"]["tag"])')
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

[FreedomBench・政治的文脈の評価](freedombench.ja.md)は、業務利用向けの必須評価項目です。英語原版の1構成には予備実測がありますが、4構成の一覧、日本語・長文への拡張、人手監査は未検収のままです。結果とそのLPA迂回の制約は、リンク先の文書が正典です。

公式ZCodeとClaude Code CLIは、[ハーネス受け入れ一覧](harnesses.ja.md)における別々の必須対象で、ケース別の状態はその一覧が正典です。後述の基礎APIスモークは一覧のAPI群に反映され、クライアント連携のケースを終わらせません。

[2台でのNCCL通信検証](nccl-validation.ja.md)は、固定したbase imageで試験したcollectiveのパターンに合格しています。その範囲はtransportと合成データの正当性であり、参照Attentionやフルモデルとは別です。

[同時2系列の独立評価](benchmarks.ja.md#標準batchingの独立評価)には、限定した課題・throughput・16K×2容量の結果があります。より広いフルモデルの数値・品質評価、持続的な混在負荷、本番復旧、batchingの組合せ、k=1／k=3以外のMTP深さとGraphsは未検証のままで、画像入力には[200Kでの画像入力](vision.ja.md)の限定した証拠しかありません。prefix cachingには[範囲を限定した独立の結果](benchmarks.ja.md#全モデルのprefix-caching独立評価p19)があり、APC／LPAとMTPの併用は別の[P22の契約](apc-lpa-design.ja.md)に従います。fixture、APIスモーク、collectiveの結果を、無制限の `tp2-kernel-validation` 証跡に変えないでください。

## フルモデルTP=2の実験範囲

reference imageは、2台のGB10ホストで45層の言語層すべてを、Marlin W4A16、eager実行、同時1系列、context 16,384、rankあたり1 GiBのKVでロードしました。直列TP=2の4層fixtureは、既存の状態検査をすべて通過しました。fixtureを同時2系列にした場合、greedy経路の1本がほぼ同値の箇所で分岐しました。この生の診断は失敗のままであり、課題水準の受け入れとは別です。

フルモデルは、served IDの基礎確認、英語・日本語の最終回答、OpenAIのSSE、無害な自動ツールの呼出し・引数・戻り、AnthropicのMessages／count_tokensのスモーク検査に合格しました。reasoning effortを低くしたチャットの受け入れ試験も、これらの最終回答・ツールの基準を満たしました。再実行では推論文が異なりました。これは診断として残すものであり、自由記述の逐語一致を要求するものではありません。未対応のthinking offを指定した要求ではparserと本文が混ざったため、受け入れ済みの構成ではありません。[ハーネスの設定](harnesses.ja.md)を参照してください。

**多バイト文字の出力。** gate射影とup射影のglobal scaleが食い違うModelOpt NVFP4 checkpointでは、多バイト文字が化けると報告されています（[vLLM #54150](https://github.com/vllm-project/vllm/issues/54150)）。固定したNVIDIAのcheckpointは該当しません。フルモデルのどのログにも `w1_weight_scale_2 must match` の警告は出ておらず、4層fixtureのlayer 3ではexpert 288個すべてでgateとupのscaleが一致しています。それでも rank 0 の `server mojibake` で監視します。日本語と韓国語で400文字以上の回答をtemperature 0で3回ずつ求め、回答とreasoningの両方でU+FFFD・孤立サロゲート・改行とタブ以外の制御文字を数えます。短い回答、別の言語の回答、空の回答、形の壊れた応答、失敗した要求は判定不能とし、合格にはしません。回答の全文は `records/<stamp>-mojibake-r0/result.json` に残ります。配布用1.3.1のprofile（2026-09-17）では6回すべて合格しました：852〜1,024文字、対象言語の文字が93〜95%、該当文字なし、すべて `stop` で終了。reasoningはprofileのlow effortで空だったため、reasoningの文字列は検査できていません。

**fixtureでの再量子化検査。** 3つのコマンドで、4層fixtureを再量子化した複製を元と比べます。`quant-error` はBF16から置き換わったNVFP4テンソルをすべて復元し、相対Frobenius誤差・最大絶対誤差・最悪の出力行を表にし、他のテンソルがbyte一致であることを確かめます。`agreement-fixture` はfixtureを教師強制で読み、top-5の行、全語彙のfloat32 log確率、8,192 token promptの93クエリ位置でのlayer 3のsparse-MLA候補集合を残します。`agreement-compare` はその記録2つから全語彙KL・argmax一致・候補集合のJaccardを出します。fixtureは言語モデルではない（教師強制top-1は1〜3%）ので、読むのは元からの移動だけで、しかも元自身の再起動間の差を物差しにします。参照imageでは、無改変fixtureの2回の起動で93個の候補集合がすべて違い（Jaccard平均0.986）、同一プロセス内の反復はbit一致でした。`agreement-fixture` は8層fixtureも読め、2つのMLA層の候補集合を別々に報告します。8層では、無改変fixtureの3回の起動がそれぞれ違う状態になり、同一プロセス内の反復もbit一致しなくなりました（argmax一致0.99〜1.00、8,192 token promptでの全語彙KLは最大0.03）。GPU 1枚・MTPなし・prefix cacheなしでの話です。配信中の全モデルは同じ挙動をより強く示す（[`server agreement`](server-configuration.ja.md#コマンド)）ので、これは層数とともに大きくなるもので、2台構成・MTP・prefix cacheが作っているものではありません。fixtureでは `python -m glm53_setup.validation.run_repeat_trace` が出どころを名指しします。2回のpassで最初に出力がずれるmoduleは、常にどこかのMoE層のrouted expertsで、その手前はrouterを含めてすべてbit一致です。`moe_align_block_size` は、入力が同じでも呼ぶたびにexpertごとのtokenの並び順を変えてMarlinのMoE kernelに渡し、kernelの結果は行の位置に依存します。約100万要素の出力のうち1〜37要素が1e-4〜1.5e-2動き、後段のrouterがそれを増幅します。kernelの作業バッファをゼロ化しても変わらず、kernelに渡す前にexpert内の並び順を固定する（`--canonical-align`）と、5文すべてで全passがbit一致になりました。これはfixture上の診断です。runtimeのpatchは入れておらず、全モデルでのコストは未計測です。

[vLLM公式の合成ベンチマーク](benchmarks.ja.md)は、計画した計測要求をすべて完了しました。試験containerはその後停止しました。これらの結果は、アプリケーション全体の品質を示すものでも、本番デプロイを認定するものでもありません。

その後、[MTP k=1の候補](speculative-decoding.ja.md)は、小規模なロードfixture、同条件のフルモデルベンチ5ケースすべて、同じ11項目の基礎API検査に合格しました。別のメタデータviewは、元のcheckpointを保ったまま、そのBF16 MTP層を全体のNVFP4から除外します。手順書には、draftの受理率、追加メモリ、負荷に依存する改善、未解決の分散停止の制約を記載しています。実際のZCode・Claude Codeの受け入れは、引き続き別扱いです。

同じフルモデルのベンチとAPIのケースは、k=3でも合格しました。[k=3の比較結果](speculative-decoding.ja.md#k3の比較結果)には、位置ごとの受理率と実効的なcache整列を記録しています。今後の実験評価ではk=3を優先しますが、最適な投機深さを示すものではなく、残るゲートを閉じるものでもありません。
