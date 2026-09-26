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

### kpool tail ringの再現

`glm53_setup/validation/kpool_ring_repro.py` は、参照imageのkpool decode kernelをGPU 1台で重みなしに動かします。poolを完成させるdraftを、その後ろのdraftがstashされた後で棄却し、やり直しが書くpoolを、正しいkeyに対するprefill側の書き込み結果（投機なしの基準）とbyte単位で比べます。vLLMのpull request #58454の回帰テストを元にしています。`patch_kpool_ring` を持つimageでの期待は、1 pool分のring（4 slot、patch前の配置）が一致せず、MTP 3のring（8 slot）が一致し、対照runはどちらでも一致することです。そうでなければ0以外で終了します。

~~~sh
python3 -m glm53_setup.validation.kpool_ring_repro --output /tmp/kpool-ring.json
~~~

未実行です。作り直したimageとGPUが要ります。確かめるのはkernelだけで、モデルの出力ではありません。

## 残る検収項目

[FreedomBench・政治的文脈の評価](freedombench.ja.md)は配信profileで完了しています（2026-09-22：固定の英語原版60問が初回で60問正解・拒否ゼロ、長文付きpilotが6問中6問）。4構成の一覧とLPAの項目は配信profileが一つになったことで退役し、日本語訳の本体・対立的な言い回し・長距離の証拠配置は未実施です。結果と範囲はリンク先の文書が正典です。

ハーネスの受け入れはケース別に[ハーネス受け入れ一覧](harnesses.ja.md)に記録し、ケース別の状態と2026-09-22の受け入れ経路の判断はその一覧が正典です。後述の基礎APIスモークは一覧のAPI群に反映され、クライアント連携のケースを終わらせません。

[2台でのNCCL通信検証](nccl-validation.ja.md)は、固定したbase imageで試験したcollectiveのパターンに合格しています。その範囲はtransportと合成データの正当性であり、参照Attentionやフルモデルとは別です。

2026-09-12の[同時2系列の独立評価](benchmarks.ja.md#標準batchingの独立評価)は、固定の重み・16K×2で範囲を限った実測です。受け入れた同時実行のprofileは公開した任意設定のものだけです（[同時実行の範囲](#同時実行の範囲)）。持続的な混在負荷とbatchingの組合せは未検証のままです。制御された停止・再起動と対の復旧は、[起動安全](launch-safety.ja.md)と[ベンチマーク](benchmarks.ja.md)に記録した `cluster switch` の演習で確認しています。MTPの深さとdecodeのGraphsには、[投機的デコーディング](speculative-decoding.ja.md)と[施策台帳](optimization-catalog.ja.md)が所有する限定した比較があります。画像入力には[画像入力](vision.ja.md)の限定した証拠しかありません。prefix cachingには[範囲を限定した独立の結果](benchmarks.ja.md#全モデルのprefix-caching独立評価p19)があり、APC／LPAとMTPの併用は別の[P22の契約](apc-lpa-design.ja.md)に従います。fixture、APIスモーク、collectiveの結果を、[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)の受け入れが宣言していない範囲の証拠に変えないでください。

## フルモデルTP=2の実験範囲

**状態。** 基準の2台の配信profile——TP=2、256K context、[起動設定](server-configuration.ja.md)の設定——は**通常運用として受け入れ済み**です：両profileでの同時1系列は2026-09-22から（2026-09-23の生成AIなんでも展示会#6での公開に向けて受け入れ、以後の通常運用も同じ範囲）、公開した任意設定での同時2系列は2026-09-23から（[同時実行の範囲](#同時実行の範囲)）。その受け入れと各項目の証拠の所在は[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)が記録で、本ページやREADMEが食い違ったら手順6が優先します。その範囲の外（他のハードウェア、それを超える同時数、動画入力、未対応の要求設定）は何も検収していません。同時実行の範囲に続く小節は、証拠をどう集めたかの記録です。

### 同時実行の範囲

| profile | 同時系列数 | 状態 |
|---|---|---|
| 配布既定 | 1（`max_num_seqs = 1`）。それを超える要求は順番待ちになり、これが宣言した挙動 | 2026-09-22に受け入れ。同時2系列以上は**非対応**：rankあたり3 GiBのKVは256Kの1系列向けで、固定の重みでの同時実行には、2台で予算を増やすことではなくrankを増やすことが要る |
| 公開した任意設定（[例のprofile](../examples/server.axl.example.toml)） | 2。rankあたり6 GiBのKVから。再パックした重みはrankあたり4.4 GiB軽い | 2026-09-22に同時1系列で、2026-09-23に同時2系列・1要求あたり約200K tokenまで（実測した範囲）で受け入れ |
| それを超える同時数 | — | **TP=4（GB10×4）を推奨**。TP=3は推奨しない：KDAの64 headほか、shardする幅は2と4で割れて3で割れない。どちらもここでは測っていない |

**同時2系列profileの証拠。** 2026-09-23の1起動で、約200Kの合言葉要求2本を同時に送って両方正答・preemptionなし、tool呼び出し2本の同時で両方が正しい呼び出し、画像1本と散文1本の同時で両方回答しました（[1.10.2での測定](benchmarks.ja.md#1102での測定)。時間・decode速度・メモリもそこにあります）。同日の3起動（数値状態は2・1・2）がそれぞれ切替後の定型——両rankの重みのdigestが初回起動と一致、decode検査、要求のtrace、3起動目はkernel hashも——を通り、[1.10.4](benchmarks.ja.md#1104での測定)のsparkDashとtool-evalは2起動目で走り、3回の切替はいずれも復旧なしで完了しました。項目は[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)にあります。未測定：境界262,144 tokenの要求2本の同時。キャンセルは[H-06 PARTIAL](harnesses.ja.md#受け入れ試験一覧と実施状態)のまま、新しい起動は配布既定と同じく仮定せず検査します。

**同時2系列での反復性。** 単独の要求はbit一致で反復します。他の要求とstepを共有した要求は違うcompletionになりえます。固定したbackendにはbatch-invariant modeが無く（[証拠の表](#証拠であり本番認定ではない)）、呼び出しを共有するものが要求の結果を変えるためです：NVFP4のMarlin MoEはK方向をexpert block数で分け、prefillと共有したstepはprefill用のkernelを通り、MTPの深さ3では相方がいるとdecodeのstepが8行になり、6行を超えるのでattentionが参照計算の代わりにFA2を通ります（[起動設定](server-configuration.ja.md#attentionとcacheとcheckpoint)）。attentionは主因ではありません：すべての呼び出しを参照計算に切り替えても2系列の差の多くは残り、容疑者の先頭はMoEです（[servingでの到達性](benchmarks.ja.md#servingでの到達性2026-09-26)）。どちらの重みでも同じで、同じ順に送った組は反復します（[1.14.0での測定](benchmarks.ja.md#1140での測定)）。これは宣言した挙動であって、受け入れた欠陥ではありません。**どんな負荷でも反復するcompletionが要る場合は `max_num_seqs = 1` で配信します**。同時に送った要求は待ち行列に入り、それぞれ単独のcompletionになります。要求が重なるときの処理量は2系列の方が約4分の1多くなります。

### 読み込み・API・ベンチマークの確認

reference imageは、2台のGB10ホストで45層の言語層すべてを、Marlin W4A16、eager実行、同時1系列、context 16,384、rankあたり1 GiBのKVでロードしました。直列TP=2の4層fixtureは、既存の状態検査をすべて通過しました。fixtureを同時2系列にした場合、greedy経路の1本がほぼ同値の箇所で分岐しました。この生の診断は失敗のままであり、課題水準の受け入れとは別です。

フルモデルは、served IDの基礎確認、英語・日本語の最終回答、OpenAIのSSE、無害な自動ツールの呼出し・引数・戻り、AnthropicのMessages／count_tokensのスモーク検査に合格しました。reasoning effortを低くしたチャットの受け入れ試験も、これらの最終回答・ツールの基準を満たしました。再実行では推論文が異なりました。これは診断として残すものであり、自由記述の逐語一致を要求するものではありません。未対応のthinking offを指定した要求ではparserと本文が混ざったため、受け入れ済みの構成ではありません。[ハーネスの設定](harnesses.ja.md)を参照してください。

[vLLM公式の合成ベンチマーク](benchmarks.ja.md)は、計画した計測要求をすべて完了しました。MTPはテンプレートに入る前に、k=1とk=3で同じベンチとAPIのケースに合格しました。別のメタデータviewは、元のcheckpointを保ったまま、そのBF16 MTP層を全体のNVFP4から除外します。深さ1〜5と3の選定は[投機的デコーディング](speculative-decoding.ja.md)にあります。これらの確認はどれも、単体ではアプリケーションの品質を示しません。宣言した範囲での通常運用は、[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)に記録した受け入れに拠ります。

### マルチバイト出力

gate射影とup射影のglobal scaleが食い違うModelOpt NVFP4 checkpointでは、多バイト文字が化けると報告されています（[vLLM #54150](https://github.com/vllm-project/vllm/issues/54150)）。固定したNVIDIAのcheckpointは該当しません。フルモデルのどのログにも `w1_weight_scale_2 must match` の警告は出ておらず、4層fixtureのlayer 3ではexpert 288個すべてでgateとupのscaleが一致しています。それでも rank 0 の `server mojibake` で監視します。日本語と韓国語で400文字以上の回答をtemperature 0で3回ずつ求め、回答とreasoningの両方でU+FFFD・孤立サロゲート・改行とタブ以外の制御文字を数えます。短い回答、別の言語の回答、空の回答、形の壊れた応答、失敗した要求は判定不能とし、合格にはしません。回答の全文は `records/<stamp>-mojibake-r0/result.json` に残ります。配布用1.3.1のprofile（2026-09-17）では6回すべて合格しました：852〜1,024文字、対象言語の文字が93〜95%、該当文字なし、すべて `stop` で終了。reasoningはprofileのlow effortで空だったため、reasoningの文字列は検査できていません。同じモデルの別のlauncherは、化けの原因を別に読んでいます。tonyd2wildのcheckpoint guard（commit `abb38bb`、2026-09-24、コードは採用しない）は、attentionを量子化したModelOptのbuildを化ける側として拒否します。BIZ AXLはattention射影を量子化している（W4A16、ModelOptではなく本リポジトリでのrepack）ので、そのguardが拒否する形に当たります。AXLのprofileでは `server mojibake` が2026-09-21と2026-09-25に合格しました（後者は `runtime.inductor_deterministic` 付きの2系列profile：日本語1,154文字・韓国語1,309文字、対象言語の文字が93〜94%、該当文字なし、すべて `stop` で終了）。temperature 0の6回答では、二つの読みのどちらが正しいかは決まりません。

### 再現性

**fixtureでの再量子化検査。** 3つのコマンドで、4層fixtureを再量子化した複製を元と比べます。`quant-error` はBF16から置き換わったNVFP4テンソルをすべて復元し、相対Frobenius誤差・最大絶対誤差・最悪の出力行を表にし、他のテンソルがbyte一致であることを確かめます。`agreement-fixture` はfixtureを教師強制で読み、top-5の行、全語彙のfloat32 log確率、8,192 token promptの93クエリ位置でのlayer 3のsparse-MLA候補集合を残します。`agreement-compare` はその記録2つから全語彙KL・argmax一致・候補集合のJaccardを出します。fixtureは言語モデルではない（教師強制top-1は1〜3%）ので、読むのは元からの移動だけで、しかも元自身の再起動間の差を物差しにします。参照imageでは、無改変fixtureの2回の起動で93個の候補集合がすべて違い（Jaccard平均0.986）、同一プロセス内の反復はbit一致でした。`agreement-fixture` は8層fixtureも読め、2つのMLA層の候補集合を別々に報告します。8層では、無改変fixtureの3回の起動がそれぞれ違う状態になり、同一プロセス内の反復もbit一致しなくなりました（argmax一致0.99〜1.00、8,192 token promptでの全語彙KLは最大0.03）。GPU 1枚・MTPなし・prefix cacheなしでの話です。配信中の全モデルは同じ挙動をより強く示す（[`server agreement`](server-configuration.ja.md#コマンド)）ので、これは層数とともに大きくなるもので、2台構成・MTP・prefix cacheが作っているものではありません。fixtureでは `python -m glm53_setup.validation.run_repeat_trace` が出どころを名指しします。2回のpassで最初に出力がずれるmoduleは、常にどこかのMoE層のrouted expertsで、その手前はrouterを含めてすべてbit一致です。`moe_align_block_size` は、入力が同じでも呼ぶたびにexpertごとのtokenの並び順を変えてMarlinのMoE kernelに渡し、kernelの結果は行の位置に依存します。約100万要素の出力のうち1〜37要素が1e-4〜1.5e-2動き、後段のrouterがそれを増幅します。kernelの作業バッファをゼロ化しても変わらず、kernelに渡す前にexpert内の並び順を固定する（`--canonical-align`）と、5文すべてで全passがbit一致になりました。並び順を固定したpassは、5文中4文で固定前のpassとbit一致し、残り1文は普段の揺れと同じ大きさの差でした。固定が変えるのは、それが取り除く揺れの範囲だけです。fixture（MoE 5層）では、固定を入れても2,048 tokenのprefillは変わらず（無改変1.231 sと1.233 s、固定1.234 sと1.235 s）、128 tokenのdecodeは約1%遅くなりました（無改変3.090 sと3.098 s、固定3.122 sと3.132 s）。MoE 1層・1 stepあたり約0.05 msです。固定版vLLMのalign kernelは多数のCUDAスレッドからatomic addでスロットを割り当てるので、並びはスレッドのスケジューリングに従い、expertが64個を超えると決定的な小規模経路は使われません。上流ではvLLM issue #52525として追跡されており、修正（#52532、#48032）は未マージです。1.6.0から、参照imageは `runtime.canonical_moe_order`（[起動設定](server-configuration.ja.md#配布用の既定設定)）の下でこの固定を入れ、既定で有効にします。参照機（TP=2、MTP k=3、image `e7a2a606…`、armごとに1回起動、他は同一）では、固定を入れると `server agreement` がbit一致で反復しました。4文すべてでargmax一致1.0・log確率の移動0、別々の2回の実行でNLLが小数4桁まで同一です。同じimageで固定を切ると一致は0.926〜0.977でした。decodeは遅くなっていません（固定あり30.81／31.27／31.06 tok/s、なし31.02／25.80／31.17、prefillは568.9対572.1 tok/s）。MTPの平均採択長は3.09から3.35に、199,652 tokenの合言葉要求（358.2 sで正答）では3.21から4.00に上がり、ソートの費用はそこに吸収されました。

**indexerのtop-kの同点。** expert内の順序を固定した後も、割れの出所がもう一つ残っていました。基準の2台でMTPの深さを4にすると（attention projectionを再量子化、prefillはFA2）、同じprose要求の9回の反復が割れました。log確率は位置298から最大0.178動き、tokenは位置320で分かれます。countとcodeの要求、および深さ3の3種は、bit一致で反復しました。稼働中のworkerの中でmoduleの指紋をtraceすると、最初に食い違う呼び出しは毎回同じで、5行の検証stepにおける2つ目のMLA層の出力projectionでした（16回中3回）。原因はその上流にあります。kpool indexerはquery行ごとに512 poolを選びますが、固定しているvLLMの `persistent_topk`（decode）と `top_k_per_row_prefill` は、512位の境界にpoolの同点があると、同じ入力から違う集合を返します。GB10 1台では、decodeのkernelが同一の呼び出し1,200回に3通り、prefillのkernelが18,000行に4通りの集合を返しました。4層のMTP fixtureでは、同一要求48本のうち17本が同じindexerの呼び出しで最初に食い違い、同点を低いpool indexに決めると36本中0本でした。同じ設定でも、ある起動では12本中0本、別の起動では24本中12本でした。割れが見えないことは、割れが無いことの証拠になりません。投機の深さは、completionがどの位置を通るかを決めるだけです。割れる起動の一つで、その瞬間を捕まえました。540 poolの行で513個が512位の値に届いており、kernelが要求の間で入れ替えた二つのpool（214と274）のscoreはbit単位で同じでした。その呼び出しまで、有効な列のlogitsは要求の間で同一でした。`runtime.stable_indexer_topk`（[サーバー設定](server-configuration.ja.md#配布用の既定設定)）がこの同点の規則を入れます。基準の2台（image `1b7dc6fa…`、深さ4）では、prose・count・codeが9回中9回、log確率の移動0で反復しました。decodeはproseで24.68 tok/s、countで45.99（修正前は24.28と45.13）、prefillは38,962 tokenで1,248.1 tok/s（修正前1,255.5）、199,652 tokenの合言葉要求は169.1 sで正答（修正前164.3 s）、その間のheadの空きは10.09 GiB以上でした。同じprofile・同じimageの二回目の起動（間に別のprofileの起動を三回挟んだ後）でも、proseとcountのcompletionは一回目と同じでした（hashが一致、各9回中9回）。翌朝の三回目の起動も同じで、コードのcompletionも一致しました。標本は3起動です。上流では、vLLM pull request #55122（決定的な `persistent_topk`）の著者が2026-09-21に目的を言い直しました：そのプロジェクト自身のトラフィックでは16K文脈の6,192行に512位の同点が一つも無く、kernelの契約は集合の一致だけとされ、pull requestは性能を主張し決定性は副次と位置づけています。このstackはそれに当たりません。上記の同点はkernelで再現し、実際のprose要求でも捕まえたので、同点の規則は維持します。そのkernelがここで規則の代わりになるかは別の問いで、変更の前にfixtureで測ります。

**起動状態（1.12.0で解決）。** 1.12.0までは、対は起動を跨ぐと三つの数値状態のどれかで計算していました（状態を確かめた配信imageの16起動で11・3・2）。それぞれの状態の中ではbit一致で反復します。重みのdigestは両rankが同じbitを読んだことを示し、要求のtraceは最初に違う呼び出しを名指ししました：rank 1のlayer 19の複製されたkpool indexerで、rank 0と下位bitで違うkeyを受け取っていたため、候補集合はpoolが同点の所でだけ違いました。keyは、投影出力のstrided view（`kw[:, :head_dim]`）に対してInductorがcompileしたpersistent reductionで正規化され、その候補configは三つ（`XBLOCK` 1・8・32）です。8と32は同じbitで、1は2,048行中2行が違い、両GB10で同じでした。永続Inductor cacheのgraphは2026-09-15にcontainerの `/tmp/torchinductor_root` から写したもので、選択をまだそこに保存していたため、各rankは起動のたびに計測で選んでいました。三つの状態はclassの三通りの組で（両rankが{8, 32}なら状態1、rank 1だけが1なら状態2、rank 0だけが1なら状態3）、modelの読み込み前にこのkernelのconfigだけを指定すると、状態1と2を狙って作れました。`runtime.inductor_deterministic`（[サーバー設定](server-configuration.ja.md#再現性のスイッチ)）はこの計測をなくします：keyを付けた各profileの3起動は、両rankでこのkernelを `XBLOCK` 8で動かし、すべての呼び出しで同じkeyを受け取って同じcompletionを返し、decodeは以前の状態1の起動の幅の中でした。各起動の状態とdecodeの数値は[1.9.0での測定](benchmarks.ja.md#190での測定)に、探索の段階（重みのdigest、固定入力のkernel hash、3段深いtrace）は1.10.0〜1.12.2の[CHANGELOG](../CHANGELOG.ja.md)にあります。TP rankに複製され、bitまで一致すべきなのに一致させる仕組みのない計算として、vLLMに [vllm-project/vllm#58636](https://github.com/vllm-project/vllm/issues/58636) で報告しました。
