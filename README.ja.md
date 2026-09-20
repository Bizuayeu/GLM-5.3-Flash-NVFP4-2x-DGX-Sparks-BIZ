# GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ

**BIZ**は保守者の印（Bizuayeu）であり、意図を示す語です。商用利用できるライセンス、資産の固定、検査結果の記録、戻せる運用を整えた**業務利用向けの構成**という意味で、製品ティア・サポート・保証・認定を意味しません。

**全モデルTP=2の参照profileを試験・実測済みです。通常運用としての受け入れは未了で、ハーネスの受け入れ状態はケース別に[ハーネス](docs/harnesses.ja.md#受け入れ試験一覧と実施状態)に記録しています。**

[English](README.md) · [セットアップ手順書](SETUP.ja.md) · [運用手順](docs/operations.ja.md) · [検証範囲](docs/validation.ja.md) · [構成](docs/architecture.ja.md) · [文書一覧](docs/README.ja.md)

NVIDIAのGLM-5.3-Flash NVFP4を、**DGX SparkおよびGB10を搭載する互換機2台**で動かすためのコミュニティ製セットアップ・検証ツールです。実測には**MSI EdgeXpert（MS-C931）2台**を使用しています。商用利用できるライセンスを軸に、資産の固定、検査結果の記録、戻せる運用を重視します。

## 導入するものと対応機体

構成は **Z.aiの原モデル → NVIDIA配布のNVFP4量子化重み → 本リポジトリのGB10向け実行・検証環境**です。

| 項目 | 導入時に確認する内容 |
|---|---|
| 原モデル | [Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| 使用する重み・取得元 | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)。固定revisionは [runtime.lock.json](config/runtime.lock.json) が正典 |
| この配布物の役割 | 重みの取得・検証、GB10向けruntime適合、起動と性能・品質検証。配布の既定は、NVIDIA配布のcheckpointをそのまま配信し、独自の再量子化・追加学習は行わない。手元で再量子化したコピーの配信は、運用者が自分で作って有効にする[任意の設定](docs/server-configuration.ja.md#配布用の既定設定)で、再量子化した重みはここでは配布しない |
| 機体 | 1台あたりGB10・128 GB級統合メモリ、Linux ARM64、NVIDIA GPU対応Dockerを備える2台。TP=2でモデルを分割し、QSFP/RoCEで接続する |
| 検証範囲 | MSI EdgeXpertでの結果を掲載。他のDGX Spark互換機も機種名だけで対応済みとはせず、ドライバー・GPU・メモリ・通信を[導入手順](SETUP.ja.md#1-必要情報を集め2台とも現状確認する)で検収する。Windowsは管理・CPU検査用で、推論はLinux実機上で行う |
| 保管と容量 | 重みは各Linux機のHugging Face cacheに置く。各台に約205 GBのディスク容量と、別途イメージ・作業領域が必要。TP=2でも各台には完全なcheckpointを置き、ロード時に分割する。[保管場所と確認方法](docs/operations.ja.md#資材の保管場所とパス) |

ソースcheckoutにはコード・固定参照・ビルド手順を含みます。本体checkpointと完成Dockerイメージは利用者の環境で取得・構築します。MTPはcheckpoint内の重みを別メタデータviewで利用し、LPAの学習済み補助器は独立した[GitHub Release添付物](docs/lpa.ja.md#学習済みprojectorの取得)として提供します。[資材の区別・配布ファイル構成・配置](docs/operations.ja.md#資材の保管場所とパス)を確認してください。

NVFP4は取得する重みの形式です。検証済みの参照構成はMarlin **W4A16**で実行しており、NVIDIAのW4A4 recipeとは演算精度が異なります。[精度と検証範囲](docs/validation.ja.md)を参照してください。

ライセンスの早見表。対象ごとに条件が違い、義務と選定理由は[ライセンス整理](docs/licensing.ja.md)、出所は[第三者通知](THIRD_PARTY_NOTICES.md)が正典です。

| 対象 | ライセンス | 出所 |
|---|---|---|
| 独自のセットアップコード・文書 | **Apache-2.0** | 本リポジトリ |
| GLM-5.3-Flash NVFP4 重み | **MIT**（固定NVIDIAモデルカードの表記。上流Z.aiモデルもMIT） | 利用者が取得。同梱しない |
| LPA cut32補助重み | **Apache-2.0**。学習データの通知は別途保持 | 任意の[Release添付物](docs/lpa.ja.md#学習済みprojectorの取得)。Git追跡外 |
| 完成Dockerイメージ | 同梱物ごと（CUDA・Torch・NCCL等）。一括して一色とは扱わない | 利用者が固定の公式base imageから構築 |
| ZCode／Claude Codeハーネス | 各製品の規約 | 別途導入。本リポジトリで再許諾しない |

本リポジトリをソース・固定参照・ビルド手順として配る場合の義務は、Apache-2.0の条件と、取り込んだ第三者コード（MIT・Apache）の著作権表示・許諾文の保持です。重みや完成イメージを再配布する場合に、それぞれの条件が加わります。EXL3/TR3重み、DFlash2重み、Mia現行AGPL版を導入する構成ではありません。

[商用利用・改造・再配布の整理](docs/licensing.ja.md)に、対象別の許諾範囲と義務をまとめています。[ハーネス連携](docs/harnesses.ja.md)では公式ZCodeと実験的なClaude Code接続を扱い、両方を必須の受け入れ対象にしています。実施状態の正典はその文書です。

## 業務利用に向けた取り組み（BIZ）

本プロジェクトは、**`nvidia/GLM-5.3-Flash-NVFP4`をDGX Spark相当の2台構成で、業務で評価・改造・運用しやすくすること**を目的としています。次の三点を一体として整備します。

- **ライセンスと出所の選択：** 商用利用できるMIT/Apache系の構成要素を優先し、採用元・版・通知を固定します。コード・重み・コンテナ・ハーネスそれぞれの条件は[ライセンス整理](docs/licensing.ja.md)に示します。
- **政治的な偏りと資料への忠実さの検証：** [FreedomBenchと業務文脈の追加試験](docs/freedombench.ja.md)で、政治的な問いへの回答・拒否・資料にない主張の挿入を調べます。対象範囲と失敗も示し、スコアだけで普遍的な思想的中立性を証明したとは扱いません。評価全体の検収は未了です。
- **実測に基づく性能調整：** MTP・LPA・prefix caching・CUDA融合・batching・並列方式を、タスク品質・メモリ・復旧と併せて検証します。[推論最適化の全体像](docs/optimization-overview.ja.md)に各施策が効く段階と用途別の構成を、[性能・品質施策台帳](docs/optimization-catalog.ja.md)に候補・証拠・保留理由をまとめ、次のGLMでも振り返れる比較基準を残します。

速度改善には、外部draftモデルを追加せず、**checkpoint同梱の標準MTPを使い、先読みトークン数は3（k=3）を選定**しました。深さ1〜5を測定済みです。3は、数え上げ・散文・コードのどれでも最下位にならない深さで、4が効くのは、手元で再量子化したattention projectionと組にした時だけです（[深さ1〜5](docs/speculative-decoding.ja.md#深さ152026-09-1920)）。配布用の起動テンプレートもこの直列最適化構成を有効にします。[既定値と必要な資材](docs/server-configuration.ja.md#配布用の既定設定)を確認してください。

業務利用に適するかは検収によって判断します。BIZの語を認定済みの意味にはせず、確認済みの範囲と残る条件を以下と各検証文書に示します。

## 確認した範囲

**配布既定は、画像入力を受ける256K（262,144 token）・KV各3 GiB・保護3 GiB・時間制限なしの直列最適化構成です（動画入力は拒否）。** [リリース候補の測定](docs/benchmarks.ja.md#リリース候補の測定)に従来の速度・tool-evalの結果とSafety Gate未達を保持し、現在の既定の確認は[画像入力](docs/vision.ja.md)と[1.6.0での測定](docs/benchmarks.ja.md#160での測定)、テキスト専用の代替は[256Kの実入力確認](docs/benchmarks.ja.md#256kでの実入力確認)にまとめています。

### 主要な測定値（1.6.0）

GB10×2、TP=2、配布既定の構成（256K・画像入力・FP8 KV各rank 3 GiB・MTP k=3・APC・chunk 2048・prefillはFA2・expert内のtoken順を固定・indexerのtop-kの同点を決定）、同時1系列です。3回または9回の中央値で、幅・条件・旧版との比較は数値の正典である[1.6.0での測定](docs/benchmarks.ja.md#160での測定)にあります。長い入力の行は、同点の規則を書く前日に、それを含まないprofileで測ったものです。

| 測定 | 結果 |
|---|---|
| temperature 0での同一要求 | 9回中9回同じcompletion、log確率の移動0（散文・数え上げ・コード） |
| prefill（38,962 tokenのprompt） | 1,271.6 tok/s（1.5.0は569.8） |
| decode（固定の短いpromptの後の512 token） | 27.17 tok/s（27.14–27.69） |
| decode（2,048 tokenのpromptの後）：数え上げ／散文／コード | 32.50／21.00／28.30 tok/s |
| 255,950 token入力、中央の合言葉1個 | 207.4 s、正答（2026-09-19、同点の規則の前。1.5.0は462.8 s） |
| 最大容量（入力262,080＋出力64 token） | 228.9 s、logprobは有限 |
| 3か所参照（256K） | 不安定：3回中2回正答、待機後の1回は誤答 |
| 256Kでの最小空きメモリ | head 5.38 GiB |

**基準の2台が配信に使っている、任意の設定二つを足した場合**（attention projectionを手元でW4A16 NVFP4に再量子化して `runtime.derived_checkpoint` で配信、MTPの深さは4。再量子化した重みは運用者が自分で作るもので、ここでは配布しません）：

| 測定 | 結果 |
|---|---|
| decode（2,048 tokenのpromptの後）、予測しやすい文（数え上げ） | **45.4 tok/s**（既定では32.5） |
| 同、コード／散文 | 34.8／24.3 tok/s（既定では28.3／21.0） |
| decode（固定の短いpromptの後の512 token） | 38.3 tok/s（既定では27.2） |
| 199,652 token入力、中央の合言葉1個 | 169.1 s、正答 |
| rankあたりの重み／headの最小空き | 91.8 GiB／10.1 GiB（既定では95.8／6.3） |
| 代価 | 教師強制のNLLが4文のうち3文で4〜6%上がる。同一要求のbit一致の反復は保たれる |

MTPのdecodeの速さは、文がどれだけ予測しやすいかで決まります。同じprofileでも、数え上げは45 tok/s、散文は24 tok/sです。数値は[配信profile](docs/benchmarks.ja.md#基準の2台の配信profile)と[深さ1〜5](docs/speculative-decoding.ja.md#深さ152026-09-1920)にあります。

[起動設定の一括管理](docs/server-configuration.ja.md)：コンテキスト長・キャッシュ・MTP・LPA・生成既定値・ノード設定を一つのTOMLにまとめ、ランチャーと専用クライアントから使えます。

[LPA（後段Prefill近似）](docs/lpa.ja.md)は配布テンプレートでは無効で、バッチ用のopt-inです（近似した要求は共有prefix cacheに登録されないため）。教師状態の復元・コーパス採取・補助器学習の道具はその経路向けに同梱しています。品質・速度の検収は、下記の基準構成とは別に扱います。

| 対象 | 状態 |
|---|---|
| 固定checkpointの取得・公式checksum確認 | 実装済み |
| 公式ARM64イメージの準備・参照イメージのbuild | 実装済み |
| 候補tokenを削らないNoPE参照attention | GPU検証済み |
| Marlin W4A16による4層・GPU 1台のfixture | 生成・状態比較を通過。8,705-token入力も確認 |
| 標準CUTLASS W4A4のfixture | 生成は完了。検査した数値不変性の条件は未達 |
| 固定SM120 sparse MLAでのbatch-invariant mode | 非対応 |
| 固定ベースによる2台のNCCL collective | RoCE経路で試験パターン合格。[実測条件と制約](docs/nccl-validation.ja.md) |
| 全45層TP=2・同時実行1の参照profile | ロード・基礎APIのテキスト／ツールを確認。[初期ベンチ](docs/benchmarks.ja.md)を測定 |
| ZCode／Claude Codeのハーネス連携 | 基礎API群は合格。共通群H-01〜H-11はnpm版ZCode CLIで一巡（PASS 5・PARTIAL 6）。公式ZCode DesktopとClaude Codeのクライアント試験は**未実施**。ケース別の状態は[ハーネス](docs/harnesses.ja.md#受け入れ試験一覧と実施状態) |
| temperature 0での同一要求・同時実行1 | bit一致で反復する（三つのpromptで9回中9回、log確率の移動0、NLLは小数4桁まで同じ）。原因を二つ直した結果で、expert内のtoken順と、indexerのtop-kの同点。基準の2台のprofileは3回の起動で同じcompletionを返した。4層fixtureでは起動が二つの数値の状態に分かれるので、新しい起動は仮定せずに確かめる。[見つけた経緯](docs/validation.ja.md#フルモデルtp2の実験範囲) |
| prefillのFA2（`runtime.fa2_attention`）・同時実行1 | 採用し、テンプレートでon。prefillは1.5.0の2.2倍、decodeは参照経路のまま、LPAとは排他。[測定](docs/benchmarks.ja.md#160での測定) |
| BF16 draftのMTP 深さ1〜5・同時実行1 | 数え上げ・散文・コード・短いpromptで測定済み。テンプレートはk=3のまま。[深さ1〜5](docs/speculative-decoding.ja.md#深さ152026-09-1920) |
| 手元で再量子化したattention projection（`runtime.derived_checkpoint`、P23） | 任意で、テンプレートには無い。基準の2台でMTPの深さ4と組にして試験採用中。decodeは+22〜29%、rankあたり4.0 GiB減、NLLは4文のうち3文で4〜6%上がる。shared expertsを足す変種は測って不採用。[施策台帳](docs/optimization-catalog.ja.md) |
| Prefix caching（APC）・同時実行1 | 実測した直列の長文prefix再利用の実験用途で受入。起動テンプレートで有効。[実測](docs/benchmarks.ja.md#全モデルのprefix-caching独立評価p19) |
| APC優先LPA（P22）・同時実行1 | 校正・MTP／融合／非同期検査との併用・held-out文書での確認まで完了。LPA自体は配布テンプレートで無効（バッチ用opt-in）。[契約](docs/apc-lpa-design.ja.md) |
| checkpoint保持・同時実行1 | 履歴試験とA/B/Aを経て、通常priming済みの途中編集用途で採用（実測は標準の間隔4,352。block幅に依存しない`dense`は実測した配置で同等、最終併用の検収は別）。[契約](docs/launch-safety.ja.md) |
| 2系列batching・Expert Parallel・PP2・unpack融合・非同期index検査 | それぞれ独立に実測。2系列と非同期検査は範囲限定で受入、EPとPP2は不採用、unpack融合は起動テンプレートで有効。[全体像](docs/optimization-overview.ja.md) |
| 200K・256Kでの画像入力（Vision）・同時実行1 | 合成画像1枚に両方の長さで正答、テキスト・ツールの回帰は合格、動画は拒否。ZCodeのツールで読んだ画像1枚を正しく説明。大きな画像とハーネス画面への直接添付は未確認。[実測と限界](docs/vision.ja.md) |
| 日本語・韓国語の長い出力、同時1系列 | 852〜1,024文字の回答6件で化け文字なし。reasoningの文字列は未検査。[検査と限界](docs/validation.ja.md#フルモデルtp2の実験範囲) |
| 動画入力・アプリ全体の品質・本番信頼性・最大性能 | **未検証** |

fixtureは元の幅・experts・選択したtensor bytesを保持しますが、層を切り詰めたモデルです。言語品質の評価には使えません。Marlin W4A16とNVIDIAのW4A4 recipeも同一の演算ではありません。[検証結果と限界](docs/validation.ja.md)を区別して利用してください。

[sparse候補の順序正規化](docs/candidate-order.ja.md)はGLM runtime共通の変更で、新しくビルドした参照imageでは既定で有効です。source更新後は再ビルドが必要で、既存imageやcontainerには自動適用されません。[導入手順](SETUP.ja.md#4-イメージ準備と参照実装の単体検証)に必要作業として明記しています。初期の最適化比較は変更前のimageで測定しています。正規化後の全モデル併用回帰は上記の候補順序文書、従来の200K併用実測は[ベンチマーク](docs/benchmarks.ja.md#リリース候補の測定)を参照してください。現在のコンテキスト・KV既定値は[200Kでの画像入力](docs/vision.ja.md)、テキスト専用の代替は[256Kの実入力確認](docs/benchmarks.ja.md#256kでの実入力確認)を参照してください。導入先で再ビルドしたruntimeも検収が必要です。

## 本リポジトリ外の関連研究

**Euryale**は、凍結したモデルの中間表現から軽い補助器で複数のdraft tokenを提案する独立した非公開の研究プロジェクトで、GLM-5.3-Flash／GB10 2台を最初の対象としています。本配布物には含まれず、上記の候補順序の正規化を除いて本リポジトリのcheckpoint・runtime・既定値を変更しません。checkpoint同梱の標準MTPに対する速度・品質・メモリの優位は未実証で、4層fixtureで実効投機幅5〜12を限定検収した段階です。全モデルの教師採取・補助器学習・同条件比較は未着手です。上記の候補順序の正規化は、この研究から生まれたruntime共通の変更です。既定の投機経路をMTP k=3からEuryaleへ切り替えるのは、同条件比較で品質・性能・メモリ・復旧のゲートを通し、[施策台帳の区別](docs/optimization-catalog.ja.md#機能受入と既定設定)（機能受入・性能採用・既定値・併用検収）で判定した場合に限ります。それまではMTP k=3が実測済みの候補です。

### DGX Spark向けの他のGLM-5.3-Flashレシピ

同じモデルを同じ級の機体で動かす公開レシピが複数あり、エンジン・量子化・割り切りがそれぞれ違います。選ぶ前に比べる価値があります。各レシピのリンク、2026-09-18時点（0xSeroは2026-09-20）で確認したライセンス、本リポジトリが取り込んだものは、この表が正典です。他の文書は名前とPR番号だけで引用します。コードを取り込んだものの表示は[第三者表示](THIRD_PARTY_NOTICES.md)にあります。

| レシピ | ライセンス | 本リポジトリが取り込んだもの |
|---|---|---|
| [amasu/glm53-flash-cluster](https://github.com/amasu/glm53-flash-cluster)（kingjones30のレシピを保持） | Apache-2.0／MIT | **コードを改変して採用：** NoPEゼロ埋めpatchの構造とレシピ |
| [tenhkspark/glm53-flash-nvfp4-2node](https://github.com/tenhkspark/glm53-flash-nvfp4-2node)と[Wabi checkpoint](https://huggingface.co/tenhkspark/GLM-5.3-Flash-NVFP4-Wabi) | Apache-2.0（コード）、MIT（重み） | コードは採用せず、再量子化した重みもここでは配布しない。BF16のattention射影をW4A16 NVFP4へ再量子化する方式をP23として評価した。全モデルでの最初の読みでは不採用、同一要求が反復するようになってから測り直し、2026-09-19から基準の2台で `runtime.derived_checkpoint` を通して試験採用中。測定は[施策台帳](docs/optimization-catalog.ja.md) |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | AGPL-3.0 | コードは採用しない。機構と測定：warmup ladder、停滞検知、KV容量の読み取り、NCCLチャネル設定、起動安全の要件、現場の手順記録 |
| [sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark) | MIT | コードは採用しない。SM90 attention経路と他container検出の起動ガードを参照点として |
| [drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated) | Apache-2.0 | コードは採用しない。zero-RoPE shimと `index_topk` 削減をattention検証の比較対象として |
| [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) | なし | コードは採用しない。測定と現場報告：GB10のメモリ挙動、checksum中の電源断、平均採択長、同時実行の結果 |
| [0xSero/GLM-5.3-Flash-EXL3-1x-DGX-Spark](https://github.com/0xSero/GLM-5.3-Flash-EXL3-1x-DGX-Spark)と[EXL3 Spark mosaic](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark) | MIT（リポジトリのコード）、MIT（別配布の重みのmodel card表記） | コード・重みは未採用。mosaicの品質パネル、cold／warm計測、overlayが実際に読み込まれたことの確認を参照。単機mcgのMTPレシピとmul1のmosaicは配布物・runtimeが異なり、速度・品質・MTP結果を合算しない |

## 必要な環境

- ツール用にPython 3.11以上。CPU検査はWindows・Linuxで実行可能。
- GPU検証にはLinux ARM64、NVIDIA GPU対応Docker、GB10。
- TP=2には2台の実機と、検証済みのQSFP/RoCE接続。
- ホストカーネル：実測は `6.17.0-1032-nvidia`。現在の DGX OS の更新で入る `7.0.0-1019-nvidia` は、既定設定のままだと2台間のRoCEが失敗することがあるため、旧カーネルを使い続けるか `kho=off` で起動する。[ホストカーネルと複数ノードRoCE](docs/operations.ja.md#ホストカーネルと複数ノードroce)を参照。
- 各配置先に約205 GBの重み、加えてイメージ・cache・任意のfixtureを保存できる容量。全checkpointは128 GBの1台には収まりません。

## checkoutから準備する

対象Linuxホストで、このリポジトリのルートから実行します。

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/huggingface.lock.txt
python -m glm53_setup --help
python -m glm53_setup --version
```

本リポジトリはcheckoutから使う運用ツールです。PyPI配布パッケージとしての提供ではありません。

### 資産の準備

```sh
python -m glm53_setup download --background
python -m glm53_setup verify-download --hf .venv/bin/hf --output records/checksum --wait
python -m glm53_setup prepare-image --background
python -m glm53_setup build-reference
```

既存のHugging Face cacheを再利用します。取得処理はprocess lockで重複を防ぎ、状態をatomicに更新します。休止中の取得を検証待ちが勝手に再開することはありません。これらは実際に処理を開始するコマンドなので、同じcacheを転送中に別の取得処理を開始しないでください。

モデルrevision・base image digest・ローカル参照タグは [config/runtime.lock.json](config/runtime.lock.json) が正典です。イメージのbuildは推論開始でも、TP=2合格でもありません。

### 推論の前に検証する

[GPU 1台のfixture手順](docs/validation.ja.md#gpu-1台のfixtureを再現する)で、実行完了・再現性・数値差を分けて確認できます。

TP=2の参照profileは**実測済みだが通常運用としては未受け入れ**です。`server preflight` は起動前に各ホストで資材・fabric・image・GPUの専有・メモリを検査しますが、品質や可用性を保証するものではありません。検査の内容は[運用手順](docs/operations.ja.md#フルモデルの起動検査)、受け入れまでに残る項目は[セットアップ手順](SETUP.ja.md#6-フルモデルの検証)を参照してください。

## ローカルデータと開発

`state/`・`records/`・認証情報・実サイトの設定・重みをGitとDocker build contextへ含めません。公開するのはレビュー済みの要約です。

CPU検査と公開境界の確認は [CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)、変更履歴は [CHANGELOG.md](CHANGELOG.md)、ライセンスは [LICENSE](LICENSE)・[NOTICE](NOTICE) を参照してください。
