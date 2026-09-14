# GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ

**BIZ**は保守者の印（Bizuayeu）であり、意図を示す語です。商用利用できるライセンス、資産の固定、検査結果の記録、戻せる運用を整えた**業務利用向けの構成**という意味で、製品ティア・サポート・保証・認定を意味しません。

**BETA（ベータ版）— 限定した参照profileで全モデルTP=2を試験済みです。本番運用・ハーネスの検収は未完了です。**

[English](README.md) · [セットアップ手順書](SETUP.ja.md) · [運用手順](docs/operations.md) · [検証範囲](docs/validation.md) · [構成](docs/architecture.md) · [文書一覧](docs/README.ja.md)

NVIDIAのGLM-5.3-Flash NVFP4を、**DGX SparkおよびGB10を搭載する互換機2台**で動かすためのコミュニティ製セットアップ・検証ツールです。実測には**MSI EdgeXpert（MS-C931）2台**を使用しています。商用利用できるライセンスを軸に、資産の固定、検査結果の記録、戻せる運用を重視します。

## 導入するものと対応機体

構成は **Z.aiの原モデル → NVIDIA配布のNVFP4量子化重み → 本リポジトリのGB10向け実行・検証環境**です。

| 項目 | 導入時に確認する内容 |
|---|---|
| 原モデル | [Z.ai GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| 使用する重み・取得元 | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)。固定revisionは [runtime.lock.json](config/runtime.lock.json) が正典 |
| この配布物の役割 | 重みの取得・検証、GB10向けruntime適合、起動と性能・品質検証。本体checkpointはNVIDIA配布物を保持し、独自の再量子化・追加学習は行わない |
| 機体 | 1台あたりGB10・128 GB級統合メモリ、Linux ARM64、NVIDIA GPU対応Dockerを備える2台。TP=2でモデルを分割し、QSFP/RoCEで接続する |
| 検証範囲 | MSI EdgeXpertでの結果を掲載。他のDGX Spark互換機も機種名だけで対応済みとはせず、ドライバー・GPU・メモリ・通信を[導入手順](SETUP.ja.md#1-必要情報を集め2台とも現状確認する)で検収する。Windowsは管理・CPU検査用で、推論はLinux実機上で行う |
| 保管と容量 | 重みは各Linux機のHugging Face cacheに置く。各台に約205 GBのディスク容量と、別途イメージ・作業領域が必要。TP=2でも各台には完全なcheckpointを置き、ロード時に分割する。[保管場所と確認方法](docs/operations.ja.md#資材の保管場所とパス) |

配布するのはソース・固定参照・ビルド手順です。重みと完成Dockerイメージは同梱せず、利用者の環境で取得・構築します。MTPはcheckpoint内の重みを別メタデータviewで利用し、LPAは本体と別の学習済み補助器を使います。[資材の区別と配置](docs/operations.ja.md#資材の保管場所とパス)を確認してください。

NVFP4は取得する重みの形式です。検証済みの参照構成はMarlin **W4A16**で実行しており、NVIDIAのW4A4 recipeとは演算精度が異なります。[精度と検証範囲](docs/validation.md)を参照してください。

ライセンスの早見表。対象ごとに条件が違い、義務と選定理由は[ライセンス整理](docs/licensing.ja.md)、出所は[第三者通知](THIRD_PARTY_NOTICES.md)が正典です。

| 対象 | ライセンス | 出所 |
|---|---|---|
| 独自のセットアップコード・文書 | **Apache-2.0** | 本リポジトリ |
| GLM-5.3-Flash NVFP4 重み | **MIT**（固定NVIDIAモデルカードの表記。上流Z.aiモデルもMIT） | 利用者が取得。同梱しない |
| 完成Dockerイメージ | 同梱物ごと（CUDA・Torch・NCCL等）。一括して一色とは扱わない | 利用者が固定の公式base imageから構築 |
| ZCode／Claude Codeハーネス | 各製品の規約 | 別途導入。本リポジトリで再許諾しない |

本リポジトリをソース・固定参照・ビルド手順として配る限り、必要なのはApache-2.0の条件だけです。重みや完成イメージを再配布する場合に、それぞれの条件が加わります。取り込んだMIT・Apacheの通知は保持します。EXL3/TR3重み、DFlash2重み、Mia現行AGPL版を導入する構成ではありません。

[商用利用・改造・再配布の整理](docs/licensing.ja.md)に、対象別の許諾範囲と義務をまとめています。[ハーネス連携](docs/harnesses.ja.md)では公式ZCodeと実験的なClaude Code接続を扱い、両方を必須の受け入れ対象にしています。実接続テストは未実施です。

## 業務利用に向けた取り組み（BIZ）

本プロジェクトは、**`nvidia/GLM-5.3-Flash-NVFP4`をDGX Spark相当の2台構成で、業務で評価・改造・運用しやすくすること**を目的としています。次の三点を一体として整備します。

- **ライセンスと出所の選択：** 商用利用できるMIT/Apache系の構成要素を優先し、採用元・版・通知を固定します。コード・重み・コンテナ・ハーネスそれぞれの条件は[ライセンス整理](docs/licensing.ja.md)に示します。
- **政治的な偏りと資料への忠実さの検証：** [FreedomBenchと業務文脈の追加試験](docs/freedombench.ja.md)で、政治的な問いへの回答・拒否・資料にない主張の挿入を調べます。対象範囲と失敗も示し、スコアだけで普遍的な思想的中立性を証明したとは扱いません。評価全体の検収は未了です。
- **実測に基づく性能調整：** MTP・LPA・prefix caching・CUDA融合・batching・並列方式を、タスク品質・メモリ・復旧と併せて検証します。[推論最適化の全体像](docs/optimization-overview.ja.md)に各施策が効く段階と用途別の構成を、[性能・品質施策台帳](docs/optimization-catalog.ja.md)に候補・証拠・保留理由をまとめ、次のGLMでも振り返れる比較基準を残します。

速度改善には、外部draftモデルを追加せず、**checkpoint同梱の標準MTPを使い、先読みトークン数は3（k=3）を選定**しました。MTP有効時の設定例も3で、[k=1との比較結果](docs/speculative-decoding.ja.md#k3の比較結果)を根拠としています。配布用の起動テンプレートもこの直列最適化構成を有効にします。[既定値と必要な資材](docs/startup-configuration.ja.md#配布用の既定設定)を確認してください。

業務利用に適するかは検収によって判断します。BIZの語を認定済みの意味にはせず、確認済みの範囲と残る条件を以下と各検証文書に示します。

## このベータ版で確認した範囲

[起動設定の一括管理](docs/startup-configuration.ja.md)：コンテキスト長・キャッシュ・MTP・LPA・生成既定値・ノード設定を一つのTOMLにまとめ、実験用ランチャーと専用クライアントから使えます。

[LPA（後段Prefill近似）](docs/lpa.ja.md)を、教師状態の復元・コーパス採取・補助器学習を含む実験経路として用意しています。品質・速度の検収は、下記の基準構成とは別に扱います。

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
| ZCode／Claude Codeのハーネス連携 | 必須の受け入れ項目を定義。**未実施** |
| BF16 draftのMTP k=1 / k=3・同時実行1 | 基礎APIと同条件ベンチが合格。次の実験はk=3を優先。[有効化手順・効果とコスト](docs/speculative-decoding.ja.md) |
| Prefix caching（APC）・APC優先LPA・checkpoint保持・同時実行1 | APCは実測した直列の長文prefix再利用の実験用途で受入、起動テンプレートで有効。APC優先LPAは校正・MTP／融合／非同期検査との併用・held-out文書での確認まで完了。checkpoint保持は履歴試験とA/B/Aを経て、通常priming済みの途中編集用途で採用（実測は標準の間隔4,352。block幅に依存しない`dense`は実測した配置で同等、最終併用の検収は別）。[実測](docs/benchmarks.ja.md#全モデルのprefix-caching独立評価p19)と[契約](docs/launch-safety.ja.md) |
| 2系列batching・Expert Parallel・PP2・unpack融合・非同期index検査 | それぞれ独立に実測。2系列と非同期検査は範囲限定で受入、EPとPP2は不採用、unpack融合は起動テンプレートで有効。[全体像](docs/optimization-overview.ja.md) |
| 他のMTP先読み数・画像・アプリ全体の品質・本番信頼性・最大性能 | **未検証** |

fixtureは元の幅・experts・選択したtensor bytesを保持しますが、層を切り詰めたモデルです。言語品質の評価には使えません。Marlin W4A16とNVIDIAのW4A4 recipeも同一の演算ではありません。[検証結果と限界](docs/validation.md)を区別して利用してください。

[sparse候補の順序正規化](docs/candidate-order.ja.md)はGLM runtime共通の変更で、新しくビルドした参照imageでは既定で有効です。source更新後は再ビルドが必要で、既存imageやcontainerには自動適用されません。[導入手順](SETUP.ja.md#4-イメージ準備と参照実装の単体検証)に必要作業として明記しています。これまでに公開した全モデルの実測は、この変更より前にビルドしたimageで取得したものです。patched imageには4層GB10 fixtureの回帰しかなく、再ビルドしたruntimeには全モデルの回帰確認が必要です。

## 本リポジトリ外の関連研究

**Euryale**は、凍結したモデルの中間表現から軽い補助器で複数のdraft tokenを提案する独立した非公開の研究プロジェクトで、GLM-5.3-Flash／GB10 2台を最初の対象としています。本配布物には含まれず、上記の候補順序の正規化を除いて本リポジトリのcheckpoint・runtime・既定値を変更しません。checkpoint同梱の標準MTPに対する速度・品質・メモリの優位は未実証で、4層fixtureで実効投機幅5〜12を限定検収した段階です。全モデルの教師採取・補助器学習・同条件比較は未着手です。上記の候補順序の正規化は、この研究から生まれたruntime共通の変更です。既定の投機経路をMTP k=3からEuryaleへ切り替えるのは、同条件比較で品質・性能・メモリ・復旧のゲートを通し、[施策台帳の区別](docs/optimization-catalog.ja.md#機能受入と既定設定)（機能受入・性能採用・既定値・併用検収）で判定した場合に限ります。それまではMTP k=3が実測済みの候補です。

## 必要な環境

- ツール用にPython 3.11以上。CPU検査はWindows・Linuxで実行可能。
- GPU検証にはLinux ARM64、NVIDIA GPU対応Docker、GB10。
- TP=2には2台の実機と、検証済みのQSFP/RoCE接続。
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

TP=2ランチャーは**検収前**です。起動には実測したノード設定と検証記録が必要ですが、本ベータ版にはTP=2の検収を完了できる手順一式はまだありません。合格記録を手書きで作らず、`service plan`・`service preflight`で確認してください。残る検証は[運用手順](docs/operations.md)に記載しています。

## ローカルデータと開発

`state/`・`records/`・認証情報・実サイトの設定・重みをGitとDocker build contextへ含めません。公開するのはレビュー済みの要約です。

CPU検査と公開境界の確認は [CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)、変更履歴は [CHANGELOG.md](CHANGELOG.md)、ライセンスは [LICENSE](LICENSE)・[NOTICE](NOTICE) を参照してください。
