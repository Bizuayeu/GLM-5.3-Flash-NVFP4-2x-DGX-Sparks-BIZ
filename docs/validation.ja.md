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
| batch不変モード | SM120の疎MLAが拒否。Triton MLAは疎に未対応 | 固定した本スタックでは使えない |

確率の許容差は、参照logprobの大きさに対するBF16イプシロン2つ分という暫定の上限でした。厳密な再実行、tokenの一致、許容差による比較は、それぞれ別の検査です。層を切り詰めたモデルは、数値差を拡大することがあります。

パッケージ化したCLIと再構成したDocker buildもGB10で確認しました。実cacheの部品試験と、context 16,384でのMarlin 4層試験は、8,705 tokenの境界入力を含めて合格しました。両試験containerともOOMなしで正常終了しました。これが検証するのは新しいパッケージ・workerのimport経路であり、run間のビット一致やフルモデルTP=2ではありません。

Marlinは演算そのものを変えます。[linear kernel](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/kernels/linear/nvfp4/marlin.py)はW4A16で、[MoEのselector](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py)は汎用の `use_a16` フラグとは独立にMARLIN向けのW4A16を選びます。このフラグだけから精度を推定しないでください。

**NVIDIAのモデルカードはこの配信を記述しません。** [固定したモデルカード](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)は、そのcheckpointについてBF16対NVFP4の精度表を載せています。その数値はNVIDIA GB200上でvLLMとSGLangを通し、temperature 1.0でサンプリングし、カードの事後量子化recipe（`nvfp4_experts_dense_mlp-kv_fp8_cast`、W4A4の経路）で測ったもので、その機体のその経路を記述します。このスタックは同じ重みをGB10上のMarlin W4A16で、独自のprefill・decode・投機の経路で動かすので、カードの表はここでは再現も主張もしません。この配信を記述する数値は、[ベンチマーク](benchmarks.ja.md)の教師強制NLLの行と長い入力の確認、[FreedomBench](freedombench.ja.md)の結果、[ハーネス](harnesses.ja.md)の各ケースで、いずれも測ったimageとprofileを添えています。

vLLMは[既定での再現性を保証していません](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/usage/reproducibility.md)。ただし、これは本リポジトリのアダプタが正しいことの証明にもなりません。上の検査は証拠であって証明ではありません。

## 候補と無改変対照の比較

差を判定する前に同じ無改変armを2本以上測り、全反復の値・中央値・範囲を残します。候補に再起動が必要なら対照も別起動を含めます。差が対照のばらつきの内側なら、同等の証明ではなく**この分解能では判定不能**です。少数反復では遅い側の尾が無いことも証明できません。

- 実行物を特定します。モデル／tokenizerのrevision、image ID、sourceと読み込まれたoverlay、実効設定、対象kernelのdispatchを確認し、意図した実装が動いた証拠のない比較は無効にします。起動の識別子、sampling／thinking、入出力長、同時数、warmup、APC履歴も結果と保存します。
- cold prefillとAPC hitを分けます。coldの入力長の梯子では各要求の早い位置に固有nonceを置き、短い入力が次の入力のprefixになるのを避け、tokenizerかusageで長さを測ります。共通system／tool部分はヒットし得るためcached-token数とサーバーログを照合し、証拠の欠落は不明のままにします。
- 同一出力を前提とする速度比較ではcompletion hashを比べます。異なるcompletionは別条件として扱い、性能・品質の結果や失敗を残し、同一出力での改善という主張へ混ぜません。
- 重み・量子化・投機の深さを変える場合は種類ごとに複数promptを使い、調整用と評価用の入力を分けます。種類ごとに3つ以上を仮の出発点とし、無改変armのprompt間のばらつきから数の根拠を決めます。tok/sと採択長、step/sの目安（decode tok/s÷採択長）を並べます。completionが変わると、演算自体の速さが変わらなくてもdraftの採択が変わるためです。step/sを比べるのは同じ深さのarmどうしに限ります。深いdraftは1 stepで検証する位置が多いためです。1.6.0のroute gと深さの掃引は種類ごとに1 promptで、1.7.0の掃引は調整用・評価用に分けた10入力です（[両方のcheckpointで深さ3](speculative-decoding.ja.md#両方のcheckpointで深さ32026-09-21)）。
- draft側の変更はcompletionを変えるものと見込みます。このstackではdraftした候補が検証のbatchに入り、targetのBF16 logitsの同点が別の側に倒れるため、同じ深さでもdraftを変えると（top-kの使い回し、関門、再量子化した `lm_head`）ほとんどの入力で文章が変わり、無改変のarmは自分の文章を反復しました。そうした変更は採択と教師強制NLLで判定し、文章の一致は要件ではなく余得として扱います。
- 同じ計測窓の送信／完了要求数・出力token数とサーバーcounter差分を照合します。背景要求の混入や説明できない不一致がある窓は制御比較から除き、値と理由を残します。counterのない既存記録は、隔離を確認できた範囲を明示します。Mia PR #139の独立TP2報告がこの区別の実例です。同報告のadaptive policy・graph範囲・scratchサイズ変更は合算で測られています。
- 採択率の分母を明示します。生成draft数・検証候補数・実採択draft数（bonusを除く）は異なります。検証prefixを短くすれば、固定深さでの予測能力が改善しなくても率が上がり得ます（Mia PR #235）。DFlash2の結果を標準MTPへ、greedyの結果をsamplingへ転用しません。

`server agreement --reference` が受け取る参照結果は1本で、対照2本には対応しません。`self_agreement` は起動内の反復を測ります。無改変→候補→無改変の比較では3本を保存し、既存の `agreement.compare_records` 関数で対照同士も確認してください。起動内の安定を起動間の安定と読み替えません。2本目の参照を受けるCLIオプションは設けていません。

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

[FreedomBench・政治的文脈の評価](freedombench.ja.md)は配信profileで完了しています（2026-09-22：固定の英語原版60問が初回で60問正解・拒否ゼロ、長文付きpilotが6問中6問）。4構成の一覧とLPAの項目は配信profileが一つになったことで退役し、日本語訳の本体・対立的な言い回し・長距離の証拠配置は未実施です。結果と範囲はリンク先の文書が正典です。

ハーネスの受け入れはケース別に[ハーネス受け入れ一覧](harnesses.ja.md)に記録し、ケース別の状態と2026-09-22の受け入れ経路の判断はその一覧が正典です。後述の基礎APIスモークは一覧のAPI群に反映され、クライアント連携のケースを終わらせません。

[2台でのNCCL通信検証](nccl-validation.ja.md)は、固定したbase imageで試験したcollectiveのパターンに合格しています。その範囲はtransportと合成データの正当性であり、参照Attentionやフルモデルとは別です。

[同時2系列の独立評価](benchmarks.ja.md#標準batchingの独立評価)には、限定した課題・throughput・16K×2容量の結果があります。これは範囲を限った実測で、同時2系列以上は受け入れた範囲の外です（[同時実行の範囲](#同時実行の範囲)）。持続的な混在負荷とbatchingの組合せは未検証のままです。制御された停止・再起動と対の復旧は、[起動安全](launch-safety.ja.md)と[ベンチマーク](benchmarks.ja.md)に記録した `cluster switch` の演習で確認しています。MTPの深さ1〜5とdecodeのGraphsには、[投機的デコーディング](speculative-decoding.ja.md)と[施策台帳](optimization-catalog.ja.md)が所有する限定した比較がありますが、これは実測であって本番の検収ではありません。画像入力には[画像入力](vision.ja.md)の限定した証拠しかありません。prefix cachingには[範囲を限定した独立の結果](benchmarks.ja.md#全モデルのprefix-caching独立評価p19)があり、APC／LPAとMTPの併用は別の[P22の契約](apc-lpa-design.ja.md)に従います。fixture、APIスモーク、collectiveの結果を、[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)の受け入れが宣言していない範囲の証拠に変えないでください。

## フルモデルTP=2の実験範囲

**状態（2026-09-22）。** 基準の2台の配信profile——TP=2、同時1系列、256K context、[起動設定](server-configuration.ja.md)の設定——は、[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)、[ベンチマーク](benchmarks.ja.md)、[ハーネスの受け入れ試験一覧](harnesses.ja.md#受け入れ試験一覧と実施状態)に記録した証拠で**通常運用として受け入れ済み**です。以下の段落は、その証拠を集めた順に書いた経緯です。「検収を確立しない」「未検証のまま」と書いてある箇所は、その結果単体では確立しなかったという意味で、宣言した範囲の外（他のハードウェア、配布既定での同時2系列以上、公開した任意設定での同時3系列以上、動画入力、未対応の要求設定）は何も検収していません。2026-09-23からは公開した任意設定の同時2系列profileも、[同時実行の範囲](#同時実行の範囲)に述べた範囲で受け入れ済みです。READMEの状態表と本節は同じことを言っています。食い違ったら[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)が記録です。

### 同時実行の範囲

配布既定は**同時1系列**（`max_num_seqs = 1`）で配信します。それを超える要求は順番待ちになり、これが宣言した挙動です。**同時2系列以上はこのprofileでは非対応**です。KV予算（rankあたり3 GiB）は256Kの1系列向けで、上の同時2系列の評価は16K×2・rankあたり1 GiBでの実測であって検収ではありません。同時実行には系列ごとにKVを確保する必要があり、固定の重みではそれは2台で予算を増やすことではなくrankを増やすことです。公開した任意設定は例外です：再パックした重みはrankあたり4.4 GiB軽く、[その例のprofile](../examples/server.axl.example.toml)はrankあたり6 GiBから同時2系列を配信します（606,881トークン、256K要求の2.32倍。2026-09-23に同時2要求を配信）。2026-09-23にこのprofileは参照対の1起動で同時2系列のタスク単位の検査に合格しました（[1.10.2での測定](benchmarks.ja.md#1102での測定)）：約200Kの合言葉要求2本を同時に送って両方正答・preemptionなし（2本で330 s、1本単独で166 s、KV使用率の最大54%、headの空き6.46 GiB）、tool呼び出し2本の同時で両方が正しい呼び出し、画像1本と散文1本の同時で両方回答。同時2系列ではcompletionは同じ要求を単独で送った時のものと一致しません：数え上げと散文は同時に走ると単独時と別のcompletionになり、同じ散文要求2本を同時に送ると別の文章が2つ返り、同じ2本の組でも回によって組が変わりました。このbackendではbatch-invariant modeが使えないためです（[証拠の表](#証拠であり本番認定ではない)）。単独で走る要求は同じ起動の中でbit一致で反復します。同時2系列のdecodeは数え上げ32.1 tok/s・散文21.8で、単独は45.6・28.2です。**このprofileは2026-09-23から通常運用として受け入れ済み**です。範囲は同時2系列・1要求あたり約200K tokenまで（実測した範囲）。2026-09-23の3起動（数値状態は2・1・2）がそれぞれ切替後の定型（両rankの重みのdigestが初回起動と一致、decode検査、要求のtrace。3起動目はkernel hashも両rankで一致）を通り、上のタスク単位の検査は1起動目、[1.10.4](benchmarks.ja.md#1104での測定)のsparkDashとtool-evalは2起動目で走り、3回の切替はいずれも復旧なしで完了しました。項目と記録先は[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)にあります。範囲の外：境界262,144 tokenの要求2本の同時は未測定、キャンセルは[H-06 PARTIAL](harnesses.ja.md#受け入れ試験一覧と実施状態)のまま、新しい起動は配布既定と同じく仮定せず検査します。同時2系列でcompletionが変わることは宣言した挙動であって、受け入れた欠陥ではありません：固定したbackendのsparse MLAにはbatch-invariant modeが無く、同じstepを共有する他の要求からcompletionを独立にするsource固定patchが書けるかは、下の起動状態の追及と併せて検証中です。patchは実測と新しいfingerprintを伴う版として出します。**同時配信にはTP=4（GB10×4）を推奨**します。TP=3は推奨しません。モデルの幾何（KDAの64 headほか、shardする幅は2と4で割れて3で割れない）のため、測る前に広範な調整が要ります。どちらの構成もここでは測っていません。

reference imageは、2台のGB10ホストで45層の言語層すべてを、Marlin W4A16、eager実行、同時1系列、context 16,384、rankあたり1 GiBのKVでロードしました。直列TP=2の4層fixtureは、既存の状態検査をすべて通過しました。fixtureを同時2系列にした場合、greedy経路の1本がほぼ同値の箇所で分岐しました。この生の診断は失敗のままであり、課題水準の受け入れとは別です。

フルモデルは、served IDの基礎確認、英語・日本語の最終回答、OpenAIのSSE、無害な自動ツールの呼出し・引数・戻り、AnthropicのMessages／count_tokensのスモーク検査に合格しました。reasoning effortを低くしたチャットの受け入れ試験も、これらの最終回答・ツールの基準を満たしました。再実行では推論文が異なりました。これは診断として残すものであり、自由記述の逐語一致を要求するものではありません。未対応のthinking offを指定した要求ではparserと本文が混ざったため、受け入れ済みの構成ではありません。[ハーネスの設定](harnesses.ja.md)を参照してください。

**多バイト文字の出力。** gate射影とup射影のglobal scaleが食い違うModelOpt NVFP4 checkpointでは、多バイト文字が化けると報告されています（[vLLM #54150](https://github.com/vllm-project/vllm/issues/54150)）。固定したNVIDIAのcheckpointは該当しません。フルモデルのどのログにも `w1_weight_scale_2 must match` の警告は出ておらず、4層fixtureのlayer 3ではexpert 288個すべてでgateとupのscaleが一致しています。それでも rank 0 の `server mojibake` で監視します。日本語と韓国語で400文字以上の回答をtemperature 0で3回ずつ求め、回答とreasoningの両方でU+FFFD・孤立サロゲート・改行とタブ以外の制御文字を数えます。短い回答、別の言語の回答、空の回答、形の壊れた応答、失敗した要求は判定不能とし、合格にはしません。回答の全文は `records/<stamp>-mojibake-r0/result.json` に残ります。配布用1.3.1のprofile（2026-09-17）では6回すべて合格しました：852〜1,024文字、対象言語の文字が93〜95%、該当文字なし、すべて `stop` で終了。reasoningはprofileのlow effortで空だったため、reasoningの文字列は検査できていません。

**fixtureでの再量子化検査。** 3つのコマンドで、4層fixtureを再量子化した複製を元と比べます。`quant-error` はBF16から置き換わったNVFP4テンソルをすべて復元し、相対Frobenius誤差・最大絶対誤差・最悪の出力行を表にし、他のテンソルがbyte一致であることを確かめます。`agreement-fixture` はfixtureを教師強制で読み、top-5の行、全語彙のfloat32 log確率、8,192 token promptの93クエリ位置でのlayer 3のsparse-MLA候補集合を残します。`agreement-compare` はその記録2つから全語彙KL・argmax一致・候補集合のJaccardを出します。fixtureは言語モデルではない（教師強制top-1は1〜3%）ので、読むのは元からの移動だけで、しかも元自身の再起動間の差を物差しにします。参照imageでは、無改変fixtureの2回の起動で93個の候補集合がすべて違い（Jaccard平均0.986）、同一プロセス内の反復はbit一致でした。`agreement-fixture` は8層fixtureも読め、2つのMLA層の候補集合を別々に報告します。8層では、無改変fixtureの3回の起動がそれぞれ違う状態になり、同一プロセス内の反復もbit一致しなくなりました（argmax一致0.99〜1.00、8,192 token promptでの全語彙KLは最大0.03）。GPU 1枚・MTPなし・prefix cacheなしでの話です。配信中の全モデルは同じ挙動をより強く示す（[`server agreement`](server-configuration.ja.md#コマンド)）ので、これは層数とともに大きくなるもので、2台構成・MTP・prefix cacheが作っているものではありません。fixtureでは `python -m glm53_setup.validation.run_repeat_trace` が出どころを名指しします。2回のpassで最初に出力がずれるmoduleは、常にどこかのMoE層のrouted expertsで、その手前はrouterを含めてすべてbit一致です。`moe_align_block_size` は、入力が同じでも呼ぶたびにexpertごとのtokenの並び順を変えてMarlinのMoE kernelに渡し、kernelの結果は行の位置に依存します。約100万要素の出力のうち1〜37要素が1e-4〜1.5e-2動き、後段のrouterがそれを増幅します。kernelの作業バッファをゼロ化しても変わらず、kernelに渡す前にexpert内の並び順を固定する（`--canonical-align`）と、5文すべてで全passがbit一致になりました。並び順を固定したpassは、5文中4文で固定前のpassとbit一致し、残り1文は普段の揺れと同じ大きさの差でした。固定が変えるのは、それが取り除く揺れの範囲だけです。fixture（MoE 5層）では、固定を入れても2,048 tokenのprefillは変わらず（無改変1.231 sと1.233 s、固定1.234 sと1.235 s）、128 tokenのdecodeは約1%遅くなりました（無改変3.090 sと3.098 s、固定3.122 sと3.132 s）。MoE 1層・1 stepあたり約0.05 msです。固定版vLLMのalign kernelは多数のCUDAスレッドからatomic addでスロットを割り当てるので、並びはスレッドのスケジューリングに従い、expertが64個を超えると決定的な小規模経路は使われません。上流ではvLLM issue #52525として追跡されており、修正（#52532、#48032）は未マージです。この版から、参照imageは `runtime.canonical_moe_order`（[起動設定](server-configuration.ja.md#配布用の既定設定)）の下でこの固定を入れ、既定で有効にします。参照機（TP=2、MTP k=3、image `e7a2a606…`、armごとに1回起動、他は同一）では、固定を入れると `server agreement` がbit一致で反復しました。4文すべてでargmax一致1.0・log確率の移動0、別々の2回の実行でNLLが小数4桁まで同一です。同じimageで固定を切ると一致は0.926〜0.977でした。decodeは遅くなっていません（固定あり30.81／31.27／31.06 tok/s、なし31.02／25.80／31.17、prefillは568.9対572.1 tok/s）。MTPの平均採択長は3.09から3.35に、199,652 tokenの合言葉要求（358.2 sで正答）では3.21から4.00に上がり、ソートの費用はそこに吸収されました。

**indexerのtop-kの同点。** expert内の順序を固定した後も、割れの出所がもう一つ残っていました。基準の2台でMTPの深さを4にすると（attention projectionを再量子化、prefillはFA2）、同じprose要求の9回の反復が割れました。log確率は位置298から最大0.178動き、tokenは位置320で分かれます。countとcodeの要求、および深さ3の3種は、bit一致で反復しました。稼働中のworkerの中でmoduleの指紋をtraceすると、最初に食い違う呼び出しは毎回同じで、5行の検証stepにおける2つ目のMLA層の出力projectionでした（16回中3回）。原因はその上流にあります。kpool indexerはquery行ごとに512 poolを選びますが、固定しているvLLMの `persistent_topk`（decode）と `top_k_per_row_prefill` は、512位の境界にpoolの同点があると、同じ入力から違う集合を返します。GB10 1台では、decodeのkernelが同一の呼び出し1,200回に3通り、prefillのkernelが18,000行に4通りの集合を返しました。4層のMTP fixtureでは、同一要求48本のうち17本が同じindexerの呼び出しで最初に食い違い、同点を低いpool indexに決めると36本中0本でした。同じ設定でも、ある起動では12本中0本、別の起動では24本中12本でした。割れが見えないことは、割れが無いことの証拠になりません。投機の深さは、completionがどの位置を通るかを決めるだけです。割れる起動の一つで、その瞬間を捕まえました。540 poolの行で513個が512位の値に届いており、kernelが要求の間で入れ替えた二つのpool（214と274）のscoreはbit単位で同じでした。その呼び出しまで、有効な列のlogitsは要求の間で同一でした。`runtime.stable_indexer_topk`（[サーバー設定](server-configuration.ja.md#配布用の既定設定)）がこの同点の規則を入れます。基準の2台（image `1b7dc6fa…`、深さ4）では、prose・count・codeが9回中9回、log確率の移動0で反復しました。decodeはproseで24.68 tok/s、countで45.99（修正前は24.28と45.13）、prefillは38,962 tokenで1,248.1 tok/s（修正前1,255.5）、199,652 tokenの合言葉要求は169.1 sで正答（修正前164.3 s）、その間のheadの空きは10.09 GiB以上でした。同じprofile・同じimageの二回目の起動（間に別のprofileの起動を三回挟んだ後）でも、proseとcountのcompletionは一回目と同じでした（hashが一致、各9回中9回）。翌朝の三回目の起動も同じで、コードのcompletionも一致しました。標本は3起動です。上流では、vLLM pull request #55122（決定的な `persistent_topk`）の著者が2026-09-21に目的を言い直しました：そのプロジェクト自身のトラフィックでは16K文脈の6,192行に512位の同点が一つも無く、kernelの契約は集合の一致だけとされ、pull requestは性能を主張し決定性は副次と位置づけています。このstackはそれに当たりません。上記の同点はkernelで再現し、実際のprose要求でも捕まえたので、同点の規則は維持します。そのkernelがここで規則の代わりになるかは別の問いで、変更の前にfixtureで測ります。

**起動状態の名指し（2026-09-23）。** 同時2系列profile（[1.10.2](benchmarks.ja.md#1102での測定)）の起動が前の起動と別の状態で計算し、probeが二つの問いに順に答えた。両rankの重みのdigestは前の起動と同一（各2,382 tensor）＝同じbitを違う計算で処理した。要求のtraceを前の起動の同じ散文・コード要求のtraceと比べると、最初に違う呼び出しは**rank 1のlayer 19のkpool indexer**で、decodeの検証step（4行。散文要求のstep 88、コード要求のstep 132）だった。hidden state・queryの射影・行ごとの候補数は同一で、候補の集合が違う。差は5呼び出し後にattention出力のall-reduceでrank 0へ届き、分岐がtraceした64トークンの先にある数え上げ要求は両rankでbit一致した。indexerはshardされず複製されている：全rankが同じcacheに同じ射影を走らせるので、二つの写しはbit一致すべきものである。前の起動では一致していなかった：rank 1のlayer 19のindexerは両要求ともまさにその呼び出しでrank 0と食い違い、新しい起動では両rankが互いにも前の起動のrank 0とも一致した。よってこの二つの起動の差は、複製されたindexerの一方のrankの写しが、同じ入力のほぼ同点を他方と違って採点したことである。pool集合が違えば同点規則は揃えられず、そのpoolが効く箇所でcompletionが分岐する。三つ目の状態は数え上げ要求の64トークンでしかtraceしておらず、そこではどの状態も一致するので、同じrank・同じlayerかは分かっていない。二つのprocessの間で違うのはindexerの採点の内側か採点するcache（cuBLASによるfp32のhead gate、融合したFWHT量子化、DeepGEMMのprocessごとにJITされるMQA-logits kernel、そして前の呼び出しがcompress-insert経路で書いたfp8 pool cache。そのbyteはtraceがfingerprintしない）で、traceはそこを分解しない。2026-09-22の13 processの検査は `lm_head` とMTPのGEMMとKDAのkernelを見たもので、これらは見ていない。この四つの計算を、参照imageの中でGB10 1台の新しいprocess 13本で、配信の形状・固定の入力で回した（fp32のhead gateとbf16のgate scoreを4行と2,048行で、融合したFWHT量子化を両方で、pool cacheのprefillのcompress-and-writeとdecodeのtail update、そのcacheに対するDeepGEMMのpaged MQA logitsと行ごとのstable top-k）。13本すべてで全出力がbit一致した（pinを入れた3本も同じ。[ベンチマーク1.9.0](benchmarks.ja.md#190での測定)）。GPU 1台の新しいprocessでは差が再現しないので、差は配信processそのものの状態（そのメモリとstreamの条件下でJITが選んだもの、あるいはその起動で違って書かれたpool cacheのbyte）に属する。そこでprobeに、固定入力でこれらの計算を稼働中の両rankで走らせhashを比べるmethodを足した（`kernel_hashes`、[1.11.0](../CHANGELOG.ja.md)）。次の起動（Probe4、03:34、1.11.0のcheckout）は再び状態2で、digest・decode検査・trace 3種は前の状態2の起動と行ごとに等しく、固定入力のhashは両rankで全部一致した。続いてhookを3段深くし入力もfingerprintしたtraceで、その起動の複製されたindexerの内側を両rankで比べた：射影の出力・fp8のquery・head gate・hidden stateは792呼び出しすべて一致し、**indexer opに渡すkeyがprefillで11のMLA層のうち8層（7・11・19・23・27・31・35・43）とdecodeの2 stepで両rankで違った**。opの候補集合が違ったのはlayer 19の2呼び出しだけで、keyの差は小さく同点でしか表に出ない。射影とopの間にあるのは、compileされたlayer norm（`_fused_indexer_k_norm`、`torch.compile` のInductor kernel。両hostのcacheに同じ4 variant）とindexerのrotary embeddingで、queryも同じrotaryを通ってfp8のqueryは一致したので、第一候補はInductorのlayer norm kernel、第二候補はrotaryのkey側。1.11.1からprobeは一つの層の実際の重みでこれらの段階もhashする（compiledのnormとeager fp32、rotaryのqueryとkey、射影）ので、次の起動がkernelを名指しする。その起動（Probe5、04:03、1.11.1のcheckout）は状態1で、両rankは層の実際の重みでのcompiledのlayer normを含む16のhashすべてで一致し、深いtraceでは両rankのindexerがすべての呼び出しで同じkeyを受け取っていた。状態2の起動と比べると、rank 0のprefillは両状態で行ごとに同一で、rank 1の最初の差はprefillの最初のMLA層（layer 7）のkeyである：起動の状態とは、rank 1のprocessがindexerのkeyをrank 0と同じbitで作るか否かである。compiledのnormは状態1の起動では固定入力で両rank一致したので、決め手は状態2の起動での同じhashで、違えばInductorのkernel、一致すればrotaryのkey側かnormが読むstride付きviewが残る。

**原因と修正（2026-09-24〜25）。** 配信の呼び出しが正規化するkeyは、投影出力のstrided view（`kw[:, :head_dim]`）で、Inductorはこれを専用のkernelにcompileします。1.11.1のstage hashは連続したtensorを渡しており、それは別のkernelを動かすので、rank間の一致は配信のkernelについて何も示していませんでした。配信のkernelは候補configが三つ（`XBLOCK` 1・8・32、warp 2）のpersistent reductionです。固定入力では8と32が同じbitで、1は2,048行中2行が違い、両GB10で同じでした。各workerで生きているInductorのautotunerを読むprobeのmethod（`autotuners`）で、各rankが使っているconfigが見えました。ホストに保存された選択は変わらないのに、それは起動ごとに変わっていました。永続cacheのgraphは2026-09-15に停止したcontainerの `/tmp/torchinductor_root` から写したもので、そのディレクトリの名前を持ったままだったため、起動のたびに計測し直し、選択をcontainerの中に保存していました。三つの状態はclassの三通りの組です：両rankが{8, 32}なら状態1、rank 1だけが1なら状態2、rank 0だけが1なら状態3。三つともprobeで観測し、最後の一つは2026-09-24でした。modelの読み込み前にこのkernelのconfigだけを指定すると、状態を狙って作れました。rank 0を8・rank 1を1にすると状態2になり、深いtraceは以前の状態2の起動と行ごとに一致しました。両方8にすると状態1に戻りました。二つの間で動いたhashは、rank 1のこのkernelの出力だけでした。`runtime.inductor_deterministic`（[サーバー設定](server-configuration.ja.md#配布用の既定設定)、1.12.0）はこの計測をなくします。keyを付けると、両rankがこのkernelを `XBLOCK` 8・候補一つで動かし、すべての呼び出しで同じkeyを受け取りました。keyを付けた2系列profileの3起動（2026-09-25）はすべて状態1で計算した：3種の課題とも同じcompletion（各3反復）、decodeはprose・count・codeで27.95〜28.44・46.05〜46.27・38.39〜38.80 tok/s、以前の状態1の起動の幅（27.7〜28.6・45.0〜47.9・37.7〜40.1）の中。最初の起動がindexerのleafを新しいcacheにcompileし、残りの2回はそれを読み込んで何もcompileしなかった。 途中で障害を二つ取り除く必要があり、keyはその両方を含みます：torch 2.13は最初にcompileしたframeの後でモードを切ること（[pytorch/pytorch#198563](https://github.com/pytorch/pytorch/issues/198563) として報告）と、モードなしでcompileしたgraphが古いcacheから計測の候補ごと戻ってくることです。TP rankに複製され、bitまで一致すべきなのに一致させる仕組みのない計算として、vLLMに [vllm-project/vllm#58636](https://github.com/vllm-project/vllm/issues/58636) で報告しました。keyを付けた配布既定（出荷どおりのテンプレート、2026-09-25に3起動）も同じふるまいでした：両rankがこのkernelを `XBLOCK` 8・候補一つで動かし、すべての呼び出しで同じkeyを受け取り、rank間で違うInductorのconfigはなく、3起動とも同じcompletionで、decodeはcount・prose・codeで32.80〜32.84・20.76〜20.82・27.53〜27.64 tok/s（公開している配布既定の値は32.01・20.67・26.68）。

[vLLM公式の合成ベンチマーク](benchmarks.ja.md)は、計画した計測要求をすべて完了しました。試験containerはその後停止しました。これらの結果は、アプリケーション全体の品質を示すものでも、本番デプロイを認定するものでもありません。宣言した範囲での通常運用としての受け入れは、その後の2026-09-22に、本節冒頭に挙げた証拠で成立しました。

その後、[MTP k=1の候補](speculative-decoding.ja.md)は、小規模なロードfixture、同条件のフルモデルベンチ5ケースすべて、同じ11項目の基礎API検査に合格しました。別のメタデータviewは、元のcheckpointを保ったまま、そのBF16 MTP層を全体のNVFP4から除外します。手順書には、draftの受理率、追加メモリ、負荷に依存する改善、未解決の分散停止の制約を記載しています。実際のZCode・Claude Codeの受け入れは、引き続き別扱いです。

同じフルモデルのベンチとAPIのケースは、k=3でも合格しました。[k=3の比較結果](speculative-decoding.ja.md#k3の比較結果)には、位置ごとの受理率と実効的なcache整列を記録しています。今後の実験評価ではk=3を優先しますが、最適な投機深さを示すものではなく、残るゲートを閉じるものでもありません。
