# Indexerの再利用と候補限定の再採点

[English](indexer-reuse.md)

**実験用の部品であり、モデルのservingには未統合です。2026-09-21に下の第一の門で中止しました。** 検証済みの4層fixture（indexer層は1つ、eager、chunk 512、出力1 token）で、indexer自身の演算はprefill 501 msのうち0.43 ms（2,048 token）、1,999 msのうち6.26 ms（8,192）、8,077 msのうち40.2 ms（32,768）＝0.09%・0.31%・0.50%で、prefillが線形に伸びる間にn^1.5程度で伸びます。全モデルには同じ形のindexer層が11あるので、indexer全体は32Kのprefillの約1.7%、200Kで約4%、再利用が削れる採点・選択はその半分ほどで、圧縮keyの書き込みとtailの更新は残ります。再利用の仕組みに見合う改善に届かないため、再利用は作りませんでした。[launch／throughputの調査](performance-investigation.ja.md)を補完しますが、削減量と品質への影響がLPAと加算になるとは前提しません。

## 固定版GLMで確認した契約

取得したNVIDIAの設定では、全層のエントリで `indexer_types` が `full` になっています。`index_share_for_mtp_iteration=true` はdraftの反復に適用されるもので、target層のindex共有ではありません。共有モードを許すスキーマがあることは、このcheckpointやこのserving経路がそれを使う証拠にはなりません。target層の実験はdraftモデルと分けて扱います。

[IndexCache](https://arxiv.org/abs/2603.12201)とその[参照patch](https://github.com/THUDM/IndexCache)は層をまたぐ選択の再利用の動機になりますが、そこに挙げられたGLM-5のアーキテクチャは `GlmMoeDsaForCausalLM` であり、このhybridな `Glm5Next` の実装ではありません。公表された削減・overlapの数値はGLM-5.3-Flashの結果ではありません。[ReTopK](https://arxiv.org/abs/2607.27692)はこれと異なり、過去のqueryの支持集合を、現在のqueryでの再ランク、直近窓のカバー、厳密な更新とともに再利用します。KVは保持したままであり、GLMが学習したKV値を共有する根拠にはなりません。時間方向の再利用では、投機・棄却されたprefixのエントリを無効化し、要求ごとに分離したままにする必要もあります。

[kpool indexer](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/model_executor/layers/sparse_attn_indexer_kpool.py)は `topk_tokens // index_kpool` 個のpoolを選び、それを論理tokenの候補へ展開し、未完のtail処理を加えます。したがって、kpool=4で2,048 tokenの予算は、選択されるpoolが512個であり、2,048個のpoolではありません。短いコンテキストの経路では、通常のscore／Top-K処理を経ずに因果的な全tokenを使うことがあります。

indexerは圧縮したkeyの書き込みと、tail状態の初期化・更新も行います。prefillだけで再利用する経路でも、その後のnative decodeのためにこれらの書き込みを残す必要があります。[モデル](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py)は共有のTop-K bufferをMLA層へ渡します。別の層が上書きする前に、選択された行をsnapshotします。物理cacheのアドレスそのものを、層・要求・rankをまたいで渡しません。

## 段階的な実験

1. **計時とcapture：** まずindexer全体を上限として測り、次に採点・選択を、projectionと必須のcache書き込みから切り離します。prefillに占める割合は計時のみの実行で、overlapは別のcapture実行で測ります。captureするqueryの位置とバイト数には上限を置きます。hidden state全体のcaptureは、MTPを含む全モデルで既に予約量を超えました。文書種別、入力長、query位置、rank、層、座標系、nativeな選択数をラベルとして記録します。
2. **Overlap：** 提案する層の対ごとに、揃えた論理候補を比較し、paddingと重複するIDは除きます。Jaccardとtargetのカバー率を、文書種別と2K／8K入力別に報告します。因果的な全tokenを選んだ場合は自明なoverlapとして印を付け、疎性に関する主張から除外します。より広いsource側のTop-2K／4K候補集合は、別の診断bufferで採取する必要があります。nativeなTop-K出力だけでは復元できません。
3. **再利用：** 有用なコストとoverlapが確認できた後にだけ、target自身のkey・tail・cacheの更新を保つ境界で、targetの選択結果を置き換えます。要求内に限定した層対のmapを明示的に適用し、エラー時は復帰させます。KVの値は共有しません。最初の受け入れ範囲はnative decodeのままです。
4. **再採点・階層pool：** 候補を扱えるkernel／gather経路で、採取したsource候補集合の内側だけでtargetを採点します。全コンテキストのscore行列を計算した後にmaskしても、その採点コストは減りません。因果性、tailの候補、padding、重複の扱い、backendのbuffer上限を維持します。第1層の固定poolは品質リスクの異なる別案であり、隣接層の再利用と自動的に同等ではありません。

NoPEであることやlatent幅が等しいことは、候補選択が層に依存しないことを示しません。LPAはattentionの入力を変え、indexerの入力も変えうるため、候補の誤差とLPAの誤差は相互作用します。

共有を前提に学習したアーキテクチャのように、主KVやpool化したindexerのKを共有することは、候補IDを複写することとは別の介入です。GLMの各層は固有の学習済みprojectionとpool gateを持ち、NoPEであっても因果・tail・pool内の位置の意味は消えません。この違いをqueryの向きが変わるだけの話として扱いません。示したJaccardの閾値は仮説であり、GLMで確立した受け入れ基準ではありません。現在試験済みのコンテキスト範囲の内側から始めます。32K／128K／256Kと時間方向の再利用には、容量・状態・品質の別途検証が必要です。

## 受け入れの順序

| 段階 | 証拠／中止条件 |
|---|---|
| コスト | 採点・選択の時間をprefill全体の時間と比較します。削減可能な割合が、既に宣言した実用上の改善目標に届かない場合は、再利用を作る前に中止します。必須のcache書き込みを削減可能な処理に数えません。 |
| Overlap | 層の対と候補予算は、品質試験の前にvalidationデータで選び、カバー率の低い対は棄却します。held-outの結果を見てから閾値を作りません。 |
| 正しさ | capture → off → 再利用 → 復帰offの順で、tokenの一致、共通tokenのlogprob差、KDA／indexer／MLAの状態を検査します。baselineと復帰後が一致しない場合、決定性と実装上の原因を切り分けるまで因果の帰属は成立しません。ただちにコードの不具合の証明になるわけでもありません。 |
| 速度 | LPA off／on × 再利用 off／onで、出力1 tokenのprefill、warmup除外、既存の5回測定の手順に従い、ばらつきも報告します。誤差と区別できない改善や、LPA単独に劣る改善は棄却します。 |
| 課題 | 長文の参照、コード、tool往復を再実行します。既存の8課題の回帰比較は維持し、過去に使った課題を未使用のholdoutと呼びません。最終の受け入れには封をした未使用のテスト文書を使い、それで再調整しません。 |

学習した重みは固定のままです。転送エラー、打切り、欠落した課題は分けて報告します。CPUの `indexer-overlap` コマンドは、`request_id`・`query_position`・`coordinate_space="logical_tokens"`・`indices` を持つ `source`／`target` 行を含むJSONの `pairs` を比較します。source側の任意の `candidate_pool` は、拡張したpoolでのrecallに対応します。整列していない入力、因果的でない入力、物理slotの入力は拒否します。

## 統合前の部品

- `runtime/indexer_capture.py`：明示的に束ねたkpoolモジュールへ、範囲を限定したhookを掛けます。opをCUDA eventで挟み、選択された論理行は共有bufferが上書きされる前にcloneします。位置の選択を空にすると計時のみの観測になり、captureにはバイト数・event数の上限があります。例外時の後始末でhookを外します。`runtime/indexer_worker.py` は、この観測器をLPA／MTPと独立に取り付けます。検証済みの4層GLM fixtureでは、2K／8K入力の実行でlayer 3の実際の候補とCUDA eventを採取し、native／capture／復帰後の出力tokenは一致しました。このfixtureにはsparse層が1つしかなく、層をまたぐoverlapとvalidationコーパスの採取は未了です。
- `runtime/indexer_candidates.py`：要求内に閉じた不変の選択snapshot、前方向の層対の検証、後始末を提供します。本ライブラリがtarget側のcache書き込みを飛ばすことはありません。候補だけを対象とするFP32のscore参照実装、検査付きの候補の重複除去・Top-K選択、完全なpoolとqueryの未完tailの展開も備えます。選択の同点はpool IDの昇順とし、backendの同点処理は異なる場合があります。論理tokenのsnapshotと、論理poolのscore入力は別のインターフェースです。
- `runtime/indexer_reindex.py`：与えられたpoolを対象とする単体のTriton採点で、固定版のFP8・32 head／128特徴の入力契約に従います。与えられたK行だけをgatherし、重み付きReLUの内積を計算します。GPU試験では、候補1／17／128件について、因果maskとpadding maskを含めて独立なFP64の密oracleと比較します（`rtol=2e-5`、`atol=2e-4`）。これは試験した範囲のscore算術を検証するもので、DeepGEMMのTop-Kとの厳密一致やモデル品質を検証するものではありません。検査した単体の境界でも、入力検証のためにhost同期を行います。
- `runtime/indexer_shared_pool.py`：整列した共有候補poolでは、target自身のK／scaleをgatherした後、固定版のnative Tensor Core採点器を使います。`searchsorted` で各queryの因果的な上限を選択poolへ対応付けます。GPU試験には、連続しないpool IDと、因果的なprefixが空の場合を含みます。これは階層poolの採点部品であり、有用な共有poolを構成する方針は別です。

queryごとに採点する最初のTriton試作は、数値的には正しかったものの、nativeの全pool採点より大幅に遅く、**serving候補ではありません**。query 512件・選択pool 1,024個の合成比較では、native poolが8,192個の条件で、検査とgatherを含む共有pool経路が約0.234→0.129 msでしたが、native poolが2,048個では0.068→0.129 msでした。後者は悪化です。poolの個数は圧縮された座標であり、kpool=4ではおよそ32K tokenと8K tokenに相当します。検収済みのモデルのコンテキスト長ではありません。`benchmark_reindex` はnative／限定／共有の測定を記録し、選択されたscoreをnativeと照合します。これらの観測は、候補のrecall、適切な層のスケジュール、モデル全体での利得を示すものではありません。

選択・展開の参照実装、K／tail書き込みの保持、要求内に限定したengineのdispatchを統合して試験するまでは、servingのオプションを有効にしません。モデルの層を再利用する判断の前には、依然として計時・overlapのゲートが置かれます。部品の試験は、GLMで有用なoverlapやモデル全体の高速化を示しません。主KVは各層に閉じたままです。

観測器の確認は、現在のソースをマウントした固定版GPUイメージ内で `python -m glm53_setup.validation.run_indexer_fixture --fixture /verified/four-layer-fixture --output /new/record` を実行して再現します。この部品の実行では、LPA・MTP・prefix cachingを無効にします。試験の記録には、ソース、baseイメージ、fixtureの識別情報、資源ガードを明記します。
