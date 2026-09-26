# GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ

**略称：NVFP4 BIZ**（引用は「NVFP4 BIZ 1.17.0」の形）。この配信スタックの呼び名で、NVIDIAの固定checkpointを配布のまま配信します。公開している任意設定の重みは **NVFP4 BIZ AXL**（AXL：attention projectionと `lm_head` をW4A16にしたもの。Hugging Faceの [Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16) で、リポジトリ名は中身の記述）。リポジトリ名はどちらもそのままです。

**BIZ**は保守者の印（Bizuayeu）であり、意図を示す語です。商用利用できるライセンス、資産の固定、検査結果の記録、戻せる運用を整えた**業務利用向けの構成**という意味で、製品ティア・サポート・保証・認定を意味しません。

[English](README.md) · [セットアップ手順書](SETUP.ja.md) · [運用手順](docs/operations.ja.md) · [検証範囲](docs/validation.ja.md) · [構成](docs/architecture.ja.md) · [文書一覧](docs/README.ja.md)

## 要約

- **何であるか。** NVIDIAのGLM-5.3-Flash NVFP4 checkpointを、**DGX SparkおよびGB10を搭載する互換機2台**でQSFP/RoCE越しにTP=2で分割し、reference imageに組み込んだ固定版vLLMで配信するコミュニティ製のセットアップ・検証ツールです。掲載した実測はMSI EdgeXpert（MS-C931）2台のものです。商用利用できるライセンス、資産の固定、検査結果の記録、戻せる運用を軸にします。
- **状態。** 両profileとも**2026-09-22から同時1系列の範囲で**（2026-09-23の生成AIなんでも展示会#6での公開に向けて受け入れ、以後の通常運用も同じ範囲）、公開した任意設定の同時2系列profileは**2026-09-23から同時2系列・1要求あたり約200K tokenまでの範囲で**、通常運用として受け入れ済みです。それぞれの受け入れが何に拠るかは[SETUP手順6](SETUP.ja.md#6-フルモデルの検証)が記録し、ハーネスの受け入れはケース別に[ハーネス](docs/harnesses.ja.md#受け入れ試験一覧と実施状態)に記録しています。他の機体、それを超える同時数、動画入力は受け入れた範囲の外です（[範囲ごとの状態](#範囲ごとの状態)）。
- **配信する二つのprofile。** **配布既定**はNVIDIA配布の固定の重みをそのまま配信します。**公開した任意設定（NVFP4 BIZ AXL）**はattention projectionと `lm_head` をW4A16に再パックしたもので、decodeが速い代わりに実測した品質の費用があり、運用者が有効にします。[確認した範囲](#確認した範囲)が両者を比較し、範囲ごとの状態を並べています。
- **精度。** 配信はGB10上のMarlin W4A16で動きます。NVIDIAのモデルカードは別のrecipe・別の機体でcheckpointを評価しているため、その精度表はこのスタックを記述しません。どの数値がこの配信を記述するかは[検証範囲](docs/validation.ja.md#証拠であり本番認定ではない)にあります。
- **ライセンス。** コードはApache-2.0、重みは運用者が取得するMITで同梱しません。資産ごとに条件が異なります（[ライセンスの早見表](#ライセンスの早見表)）。
- **未検証。** 公開した任意設定の同時2系列を超える同時実行の配信、動画入力、アプリケーション全体の品質、本番の信頼性、最大性能（[範囲ごとの状態](#範囲ごとの状態)）。

## 導入するものと対応機体

構成は **Z.aiの原モデル → NVIDIA配布のNVFP4量子化重み → 本リポジトリのGB10向け実行・検証環境**です。

| 項目 | 導入時に確認する内容 |
|---|---|
| 原モデル | [Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| 使用する重み・取得元 | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)。固定revisionは [runtime.lock.json](config/runtime.lock.json) が正典 |
| この配布物の役割 | 重みの取得・検証、GB10向けruntime適合、起動と性能・品質検証。配布の既定は、NVIDIA配布のcheckpointをそのまま配信し、独自の再量子化・追加学習は行わない。再量子化したコピーの配信は運用者が有効にする[任意の設定](docs/server-configuration.ja.md#配布用の既定設定)で、再量子化した重みは本リポジトリに同梱しない |
| 機体 | 1台あたりGB10・128 GB級統合メモリ、Linux ARM64、NVIDIA GPU対応Dockerを備える2台。TP=2でモデルを分割し、QSFP/RoCEで接続する |
| 検証範囲 | MSI EdgeXpertでの結果を掲載。他のDGX Spark互換機も機種名だけで対応済みとはせず、ドライバー・GPU・メモリ・通信を[導入手順](SETUP.ja.md#1-必要情報を集め2台とも現状確認する)で検収する。Windowsは管理・CPU検査用で、推論はLinux実機上で行う |
| 保管と容量 | 重みは各Linux機のHugging Face cacheに置く。各台に約205 GBのディスク容量と、別途イメージ・作業領域が必要。TP=2でも各台には完全なcheckpointを置き、ロード時に分割する。[保管場所と確認方法](docs/operations.ja.md#資材の保管場所とパス) |
| 起動設定 | [一つの起動設定TOML](docs/server-configuration.ja.md)に、コンテキスト長・キャッシュ・MTP・LPA・生成既定値・ノード設定をまとめ、ランチャーと専用クライアントから使う。配布用テンプレートは固定の重みで直列最適化構成を有効にする。[既定値と必要な資材](docs/server-configuration.ja.md#配布用の既定設定) |

ソースcheckoutにはコード・固定参照・ビルド手順を含みます。本体checkpointと完成Dockerイメージは利用者の環境で取得・構築します。MTPはcheckpoint内の重みを別メタデータviewで利用し、LPAの学習済み補助器は独立した[GitHub Release添付物](docs/lpa.ja.md#学習済みprojectorの取得)として提供します。[資材の区別・配布ファイル構成・配置](docs/operations.ja.md#資材の保管場所とパス)を確認してください。

NVFP4は取得する重みの形式です。検証済みの参照構成はMarlin **W4A16**で実行しており、NVIDIAのW4A4 recipeとは演算精度が異なります。NVIDIAのモデルカードの精度表はそのrecipeで、別の機体・別のengine経路で測ったもので、この配信の品質の主張ではありません。モデルカードの数値が何を記述し、どの数値がこのスタックを記述するかは[精度と検証範囲](docs/validation.ja.md#証拠であり本番認定ではない)を参照してください。

[LPA（後段Prefill近似）](docs/lpa.ja.md)は配布テンプレートでは無効で、バッチ用のopt-inです（近似した要求は共有prefix cacheに登録されないため）。教師状態の復元・コーパス採取・補助器学習の道具はその経路向けに同梱しています。その品質・速度の検収は、下記の確認した範囲とは別に扱います。

### ライセンスの早見表

対象ごとに条件が違い、義務と選定理由は[ライセンス整理](docs/licensing.ja.md)、出所は[第三者通知](THIRD_PARTY_NOTICES.md)が正典です。

| 対象 | ライセンス | 出所 |
|---|---|---|
| 独自のセットアップコード・文書 | **Apache-2.0** | 本リポジトリ |
| GLM-5.3-Flash NVFP4 重み | **MIT**（固定NVIDIAモデルカードの表記。上流Z.aiモデルもMIT） | 利用者が取得。同梱しない |
| attentionと `lm_head` のW4A16再パック（公開した任意設定） | **MIT**。NVIDIAのモデルカードを併置 | 任意の[Hugging Face配布の重み](docs/licensing.ja.md#重みのmit通知)。Git追跡外 |
| LPA cut32補助重み | **Apache-2.0**。学習データの通知は別途保持 | 任意の[Release添付物](docs/lpa.ja.md#学習済みprojectorの取得)。Git追跡外 |
| 完成Dockerイメージ | 同梱物ごと（CUDA・Torch・NCCL等）。一括して一色とは扱わない | 利用者が固定の公式base imageから構築 |
| ZCode／Claude Codeハーネス | 各製品の規約 | 別途導入。本リポジトリで再許諾しない |

本リポジトリをソース・固定参照・ビルド手順として配る場合の義務は、Apache-2.0の条件と、取り込んだ第三者コード（MIT・Apache）の著作権表示・許諾文の保持です。重みや完成イメージを再配布する場合に、それぞれの条件が加わります。EXL3/TR3重み、DFlash2重み、Mia現行AGPL版を導入する構成ではありません。対象別の許諾範囲と義務は[商用利用・改造・再配布の整理](docs/licensing.ja.md)にまとめています。

## 必要な環境

- ツール用にPython 3.11以上。CPU検査はWindows・Linuxで実行可能。
- GPU検証にはLinux ARM64、NVIDIA GPU対応Docker、GB10。
- TP=2には2台の実機と、検証済みのQSFP/RoCE接続。
- ホストカーネル：実測は `6.17.0-1032-nvidia`。現在の DGX OS の更新で入る `7.0.0-1019-nvidia` は、既定設定のままだと2台間のRoCEが失敗することがあるため、旧カーネルを使い続けるか `kho=off` で起動する。[ホストカーネルと複数ノードRoCE](docs/operations.ja.md#ホストカーネルと複数ノードroce)を参照。
- 各配置先に約205 GBの重み、加えてイメージ・cache・任意のfixtureを保存できる容量。全checkpointは128 GBの1台には収まりません。

## checkoutから準備する

対象Linuxホストで、このリポジトリのルートから実行します。

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/huggingface.lock.txt
python -m glm53_setup --help
python -m glm53_setup --version
~~~

本リポジトリはcheckoutから使う運用ツールです。PyPI配布パッケージとしての提供ではありません。機体の現状確認から受け入れまでの手順は[セットアップ手順書](SETUP.ja.md)が順に示します。

### 資産の準備

~~~sh
python -m glm53_setup download --background
python -m glm53_setup verify-download --hf .venv/bin/hf --output records/checksum --wait
python -m glm53_setup prepare-image --background
python -m glm53_setup build-reference
~~~

既存のHugging Face cacheを再利用します。取得処理はprocess lockで重複を防ぎ、状態をatomicに更新します。休止中の取得を検証待ちが勝手に再開することはありません。これらは実際に処理を開始するコマンドなので、同じcacheを転送中に別の取得処理を開始しないでください。

モデルrevision・base image digest・ローカル参照タグは [config/runtime.lock.json](config/runtime.lock.json) が正典です。イメージのbuildは推論開始でも、TP=2合格でもありません。[sparse候補の順序正規化](docs/candidate-order.ja.md)をはじめruntimeへのpatchはimageの中にあり、source更新後は再ビルドが必要です。導入先で再ビルドしたruntimeも検収が必要です。

### 推論の前に検証する

[GPU 1台のfixture手順](docs/validation.ja.md#gpu-1台のfixtureを再現する)で、実行完了・再現性・数値差を分けて確認できます。

TP=2の参照profileは**実測済みで、通常運用として受け入れ済み（2026-09-22）**です。`server preflight` は起動前に各ホストで資材・fabric・image・GPUの専有・メモリを検査しますが、品質や可用性を保証するものではありません。検査の内容は[運用手順](docs/operations.ja.md#フルモデルの起動検査)、受け入れ各項目の証拠の所在は[セットアップ手順](SETUP.ja.md#6-フルモデルの検証)を参照してください。

## 確認した範囲

**配布既定は、画像入力を受ける256K（262,144 token）・KV各3 GiB・保護3 GiB・時間制限なしの直列最適化構成です（動画入力は拒否）。** この既定の裏付けは[画像入力](docs/vision.ja.md)と[1.6.0での測定](docs/benchmarks.ja.md#160での測定)、テキスト専用の代替は[256Kの実入力確認](docs/benchmarks.ja.md#256kでの実入力確認)、従来の速度・tool-evalの結果とSafety Gate未達は[リリース候補の測定](docs/benchmarks.ja.md#リリース候補の測定)が保持しています。

### 主要な測定値（1.15.0）

GB10×2、TP=2、prefillはFA2、expert内のtoken順を固定、indexerのtop-kの同点を決定、MTP k=3。二つのprofile：**配布既定**（固定のNVIDIA重み。テンプレートが配信するもの）と、**公開した任意設定**（attention projectionと `lm_head` をW4A16 NVFP4に再パックし、KDAのinput projectionを分割して宣言した `runtime.derived_checkpoint` で配信。重みは[Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16)）。任意設定の列は参照対が配信するprofile＝[同時2系列のAXL profile](examples/server.axl.example.toml)を2026-09-23に測ったもの（[1.10.4での測定](docs/benchmarks.ja.md#1104での測定)。NLLは2026-09-25、[1.13.0での測定](docs/benchmarks.ja.md#1130での測定)）で、その夜に測り直していない行は最後に測った夜の値を日付つきで残しています。配布既定の列は2026-09-22の夜です。3回または9回の中央値で、幅・条件・旧版はすべて[ベンチマーク](docs/benchmarks.ja.md)にあります。文種は常に 数え上げ／散文／コード の順に並べます。

| 分類 | 測定 | 配布既定（固定の重みでのNVFP4 BIZ。注記がなければ2026-09-22） | 公開した任意設定（NVFP4 BIZ AXL、配信中の同時2系列profile。注記がなければ2026-09-23） |
|---|---|---|---|
| prefill | prefill（38,962 tokenのprompt） | 1,232.8 tok/s（1.7.1の夜は1,277.0） | **1,294.8 tok/s**（2026-09-22） |
| decode | decode（2,048 tokenのpromptの後）：数え上げ／散文／コード | 32.01／20.67／26.68 tok/s | **45.04／28.16／37.67 tok/s**（もう一方が走っている間は数え上げ32.09／散文21.81） |
| decode | decode（固定の短いpromptの後の512 token） | 26.87〜27.31 tok/s | **41.8 tok/s**（2026-09-22） |
| decode | sparkDash DecodeBench（128 token）：structured／prose／code／json | 36.24／26.68／31.67／26.25 tok/s（1.5.0） | **48.23／31.38／41.28／34.88 tok/s** |
| 長文入力 | 約200K token入力、中央の合言葉1個 | 173.5 sと173.6 s、正答（199,652 token） | **165.7 s、正答**（199,649 token）。同種の2本を同時に：330.2 s、両方正答、preemptionなし |
| 長文入力 | 255,950 token入力、中央の合言葉1個 | 217.3 s、正答 | 220.9 s、正答（2026-09-22） |
| 長文入力 | 261,573 tokenの3か所参照、背景を囲んだprompt（1.13.0からの正典） | 234.2 s、3回とも正答（2026-09-25） | **230.7 s、3回とも正答**（2026-09-25） |
| 長文入力 | 最大容量（入力262,080＋出力64 token） | 240.3 s、logprobは有限 | 245.7 s、logprobは有限（2026-09-22） |
| 品質 | 教師強制のNLL：日本語／英語／コード／数学 | 1.5963／2.0241／0.9479／0.5931 | 1.6270／1.9946／0.9601／0.6275（2026-09-25。同時1系列のprofileでは1.6645／2.0024／1.0031／0.6279、2026-09-21） |
| 品質 | tool-eval-bench、標準69シナリオ | 90／100（1.0.0、2026-09-14） | 88／100、failは同じ3件、Safety Gate未達 |
| 反復性 | temperature 0での同一要求 | 9回中9回同じcompletion、log確率の移動0。1.12.0からはどの起動も同じ数値状態（3起動中3起動） | 単独の要求なら同じ。要求が2本走っている間のcompletionはその要求の単独時と違いうる。`max_num_seqs = 1` なら同時に送った要求は待ち行列に入り、それぞれ単独のcompletionを繰り返す（[1.14.0での測定](docs/benchmarks.ja.md#1140での測定)） |
| メモリ | bench中のheadの最小空きメモリ | 5.53 GiB（KV 3 GiB） | 200K 2本の同時で6.46 GiB、sparkDash中は7.24 GiB（KV 6 GiB） |

| | 配布既定 | 公開した任意設定 |
|---|---|---|
| **長所** | 固定のNVIDIA重みに対してlossless。テンプレートに入っており、追加の取得が要らない | decode stepが全入力で12〜13 ms短い（深さ3・10入力の平均で78 ms）。prefillも同じ夜の既定より2〜5%速い（projectionの分割で以前の代価が消えた）。同一要求のbit一致の反復は保たれ、256Kでheadの空きが約4 GiB多く残る |
| **短所** | どの文種でも二つのうち遅い方 | losslessではない：NLLが4文のうち3文で1〜6%上がる。テンプレートの外で、二つ目のcheckpointを取得・配置する必要がある。約250K tokenを超える要求には、1.7.0以降でbuildしたimageが持つslot-mapping guardが要る |
| **向く用途** | コード・ツール利用と、固定の重みと一致させたい用途全般 | 日本語散文をはじめ、NLLの代価を許せる生成主体の直列用途 |

MTPのdecodeの速さは、文がどれだけ予測しやすいかで決まります。同じprofileでも、数え上げは45 tok/s、散文は28 tok/sです。数値は[深さ3の公開した任意設定](docs/benchmarks.ja.md#基準の2台の配信profileattentionと-lm_head-の再パック深さ3)と[両方のcheckpointで深さ3](docs/speculative-decoding.ja.md#両方のcheckpointで深さ32026-09-21)に、選定は[用途別の構成](docs/optimization-overview.ja.md#用途別の構成)にあります。decodeの数値はすべて両rankのworkerが高性能コアにいるときのもので、どちらかのrankが高効率コアにいるとdecodeは約3分の1に落ちます。これは[`nodes[].cpuset_cpus`](docs/server-configuration.ja.md#cpu配置の任意指定)で防げます（[1.15.0での測定](docs/benchmarks.ja.md#1150での測定)）。

### 範囲ごとの状態

各行は状態と、証拠を持つ文書を示します。経緯はその文書にあります。断りが無ければ同時1系列です。

| 区分 | 対象 | 状態 |
|---|---|---|
| ツール | 固定checkpointの取得・公式checksum確認。公式ARM64イメージの準備・参照イメージのbuild | 実装済み |
| fixture | 候補tokenを削らないNoPE参照attention | GPU検証済み |
| fixture | Marlin W4A16による4層・GPU 1台のfixture | 生成・状態比較を通過。8,705-token入力も確認。[検証範囲](docs/validation.ja.md) |
| fixture | 固定SM120 sparse MLAでのbatch-invariant mode | 非対応 |
| 全モデル | 固定ベースによる2台のNCCL collective | RoCE経路で試験パターン合格。[実測条件と制約](docs/nccl-validation.ja.md) |
| 全モデル | 45層TP=2の参照profile | ロード・基礎APIのテキスト／ツールを確認。[ベンチマーク](docs/benchmarks.ja.md) |
| 全モデル | temperature 0での同一要求 | 同じ起動の中でbit一致で反復し、1.12.0からは起動を跨いでも同じ：両テンプレートでonの[再現性のスイッチ](docs/server-configuration.ja.md#再現性のスイッチ)が、expert内のtoken順、indexerのtop-kの同点、Inductorのrankごとのconfigの選択を固定する。どちらのprofileでも3起動とも同じcompletion。新しい起動は今も切替ごとに確かめる（重みのdigest、decode検査、kernel hash）。[それぞれの原因を見つけた経緯](docs/validation.ja.md#再現性) |
| 全モデル | 200K・256Kでの画像入力（Vision） | 合成画像1枚に両方の長さで正答、テキスト・ツールの回帰は合格、動画は拒否。大きな画像とハーネス画面への直接添付は未確認。[実測と限界](docs/vision.ja.md) |
| 全モデル | 日本語・韓国語の長い出力 | 852〜1,024文字の回答6件で化け文字なし。reasoningの文字列は未検査。[検査と限界](docs/validation.ja.md#フルモデルtp2の実験範囲) |
| 同時実行 | 同時2系列以上 | 配布既定では**非対応**（`max_num_seqs = 1`。要求は順番待ち）。公開した任意設定の例はrankあたり6 GiBから同時2系列を配信し、1要求あたり約200K tokenまでの同時2系列で**2026-09-23から通常運用として受け入れ済み**。他の要求とstepを共有した要求は違うcompletionになりうる（宣言した挙動）。どんな負荷でも反復するcompletionが要るなら `max_num_seqs = 1` で配信する。それを超える同時数：TP=4を推奨、TP=3は非推奨。どちらもここでは未計測。[同時実行の範囲](docs/validation.ja.md#同時実行の範囲) |
| ハーネス | ZCode／Claude Codeの連携 | 基礎API群は合格。共通群H-01〜H-11はnpm版ZCode CLIで一巡（PASS 5・PARTIAL 6）。公式ZCode Desktopは**BLOCKED**（同梱CLIが対話起動できない。[feedback #270](https://github.com/zai-org/feedback/issues/270)）、Claude Codeは**判断で見送り**（同じ機体のAnthropicサブスクリプション設定と競合する）。受け入れた経路はnpm版ZCode CLI。[受け入れ試験一覧](docs/harnesses.ja.md#受け入れ試験一覧と実施状態) |
| テンプレートで有効 | prefillのFA2（`runtime.fa2_attention`） | 採用。prefillは1.5.0の2.2倍、1系列のdecodeは参照経路のまま、LPAとは排他。[測定](docs/benchmarks.ja.md#160での測定) |
| テンプレートで有効 | BF16 draftのMTP k=3 | 10入力で、再量子化したcheckpointでは深さ1〜5を、固定のcheckpointでは1・3・4を測定。k=3を両方に採用。[投機デコード](docs/speculative-decoding.ja.md#両方のcheckpointで深さ32026-09-21) |
| テンプレートで有効 | Prefix caching（APC） | 実測した直列の長文prefix再利用の実験用途で受入。[実測](docs/benchmarks.ja.md#全モデルのprefix-caching独立評価p19) |
| テンプレートで有効 | checkpoint保持 | 履歴試験とA/B/Aを経て、通常priming済みの途中編集用途で採用（実測は標準の間隔4,352。block幅に依存しない`dense`は実測した配置で同等、最終併用の検収は別）。[契約](docs/launch-safety.ja.md) |
| テンプレートで有効 | unpack融合・非同期index検査 | それぞれ独立に実測して有効化。[全体像](docs/optimization-overview.ja.md) |
| 任意・既定off | 再量子化したattention projectionと `lm_head`（`runtime.derived_checkpoint`、P23） | 上の公開した任意設定。shared expertsを足す変種は測って不採用。[実測](docs/benchmarks.ja.md#基準の2台の配信profileattentionと-lm_head-の再パック深さ3)／[施策台帳](docs/optimization-catalog.ja.md) |
| 任意・既定off | APC優先LPA（P22） | 校正・MTP／融合／非同期検査との併用・held-out文書での確認まで完了。バッチ用opt-in。[契約](docs/apc-lpa-design.ja.md) |
| 測って不採用 | Expert Parallel、PP2、decodeのCUDA Graphs、採択履歴による深さ、draftの確信度の関門、draft側の設定二つ | それぞれ全モデルで測り、数値は所有文書にある。[全体像](docs/optimization-overview.ja.md)、[投機デコード](docs/speculative-decoding.ja.md#固定の深さの先2026-09-21) |
| 測って不採用 | 層間のindexer再利用（CSA2、P16） | コストの門で中止。indexerはfixtureでprefillの1%未満、全モデルの射影で200Kでも約4%。[設計と結果](docs/indexer-reuse.ja.md) |
| 未検証 | 動画入力・アプリ全体の品質・本番信頼性・最大性能 | **未検証** |

fixtureは元の幅・experts・選択したtensor bytesを保持しますが、層を切り詰めたモデルです。言語品質の評価には使えません。Marlin W4A16とNVIDIAのW4A4 recipeも同一の演算ではありません。[検証結果と限界](docs/validation.ja.md)を区別して利用してください。

## 業務利用に向けた取り組み（BIZ）

本プロジェクトは、**`nvidia/GLM-5.3-Flash-NVFP4`をDGX Spark相当の2台構成で、業務で評価・改造・運用しやすくすること**を目的としています。次の三点を一体として整備します。

- **ライセンスと出所の選択：** 商用利用できるMIT/Apache系の構成要素を優先し、採用元・版・通知を固定します。コード・重み・コンテナ・ハーネスそれぞれの条件は[ライセンス整理](docs/licensing.ja.md)に示します。
- **政治的な偏りと資料への忠実さの検証：** [FreedomBenchと業務文脈の追加試験](docs/freedombench.ja.md)で、政治的な問いへの回答・拒否・資料にない主張の挿入を調べます。対象範囲と失敗も示し、スコアだけで普遍的な思想的中立性を証明したとは扱いません。配信profileでは2026-09-22に完了しています（固定の英語原版60問が初回で60問正解・拒否ゼロ、長文付きpilotが6問中6問）。日本語訳の本体・対立的な言い回し・証拠配置は未実施です。
- **実測に基づく性能調整：** MTP・LPA・prefix caching・CUDA融合・batching・並列方式を、タスク品質・メモリ・復旧と併せて検証します。[推論最適化の全体像](docs/optimization-overview.ja.md)に各施策が効く段階と用途別の構成を、[性能・品質施策台帳](docs/optimization-catalog.ja.md)に候補・証拠・保留理由をまとめ、次のGLMでも振り返れる比較基準を残します。

速度改善には、外部draftモデルを追加せず、**checkpoint同梱の標準MTPを使い、先読みトークン数は3（k=3）を選定**し、両方のcheckpointに共通の深さとしました。理由は[両方のcheckpointで深さ3](docs/speculative-decoding.ja.md#両方のcheckpointで深さ32026-09-21)にあります。

業務利用に適するかは検収によって判断します。BIZの語を認定済みの意味にはせず、確認済みの範囲と残る条件を上記と各検証文書に示します。

## 本リポジトリ外の関連研究

**Euryale**は、凍結したモデルの中間表現から軽い補助器で複数のdraft tokenを提案する独立した非公開の研究プロジェクトで、GLM-5.3-Flash／GB10 2台を最初の対象としています。本配布物には含まれず、この研究から生まれた[候補順序の正規化](docs/candidate-order.ja.md)を除いて本リポジトリのcheckpoint・runtime・既定値を変更しません。checkpoint同梱の標準MTPに対する速度・品質・メモリの優位は未実証で、4層fixtureで実効投機幅5〜12を限定検収した段階です。全モデルの教師採取・補助器学習・同条件比較は未着手です。既定の投機経路をMTP k=3からEuryaleへ切り替えるのは、同条件比較で品質・性能・メモリ・復旧のゲートを通し、[施策台帳の区別](docs/optimization-catalog.ja.md#機能受入と既定設定)（機能受入・性能採用・既定値・併用検収）で判定した場合に限ります。それまではMTP k=3が実測済みの候補です。

### DGX Spark向けの他のGLM-5.3-Flashレシピ

同じモデルを同じ級の機体で動かす公開レシピが複数あり、エンジン・量子化・割り切りがそれぞれ違います。選ぶ前に比べる価値があります。各レシピのリンク、2026-09-18時点（0xSeroは2026-09-20）で確認したライセンス、本リポジトリが取り込んだものは、この表が正典です。他の文書は名前とPR番号だけで引用します。コードを取り込んだものの表示は[第三者表示](THIRD_PARTY_NOTICES.md)にあります。

| レシピ | ライセンス | 本リポジトリが取り込んだもの |
|---|---|---|
| [amasu/glm53-flash-cluster](https://github.com/amasu/glm53-flash-cluster)（kingjones30のレシピを保持） | Apache-2.0／MIT | **コードを改変して採用：** NoPEゼロ埋めpatchの構造とレシピ |
| [tenhkspark/glm53-flash-nvfp4-2node](https://github.com/tenhkspark/glm53-flash-nvfp4-2node)と[Wabi checkpoint](https://huggingface.co/tenhkspark/GLM-5.3-Flash-NVFP4-Wabi) | Apache-2.0（コード）、MIT（重み） | コードも重みも採用しない。BF16のattention射影をW4A16 NVFP4へ再量子化する方式をP23として評価し、上の公開した任意設定（attentionと `lm_head`）へ育てた。測定は[施策台帳](docs/optimization-catalog.ja.md) |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | AGPL-3.0 | コードは採用しない。機構と測定：warmup ladder、停滞検知、KV容量の読み取り、NCCLチャネル設定、起動安全の要件、現場の手順記録 |
| [sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark) | MIT | コードは採用しない。SM90 attention経路と他container検出の起動ガードを参照点として |
| [drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated) | Apache-2.0 | コードは採用しない。zero-RoPE shimと `index_topk` 削減をattention検証の比較対象として |
| [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) | なし | コードは採用しない。測定と現場報告：GB10のメモリ挙動、checksum中の電源断、平均採択長、同時実行の結果。2026-09-20のattention／MLP射影の量子化の記録は、P23と同じテンソル集合に独立に到達している（TP=4、品質は未測定、[施策台帳](docs/optimization-catalog.ja.md)） |
| [0xSero/GLM-5.3-Flash-EXL3-1x-DGX-Spark](https://github.com/0xSero/GLM-5.3-Flash-EXL3-1x-DGX-Spark)と[EXL3 Spark mosaic](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark) | MIT（リポジトリのコード）、MIT（別配布の重みのmodel card表記） | コード・重みは未採用。mosaicの品質パネル、cold／warm計測、overlayが実際に読み込まれたことの確認を参照。単機mcgのMTPレシピとmul1のmosaicは配布物・runtimeが異なり、速度・品質・MTP結果を合算しない |

## ローカルデータと開発

`state/`・`records/`・認証情報・実サイトの設定・重みをGitとDocker build contextへ含めません。公開するのはレビュー済みの要約です。

CPU検査と公開境界の確認は [CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)、変更履歴は [CHANGELOG.ja.md](CHANGELOG.ja.md)、ライセンスは [LICENSE](LICENSE)・[NOTICE](NOTICE) を参照してください。
