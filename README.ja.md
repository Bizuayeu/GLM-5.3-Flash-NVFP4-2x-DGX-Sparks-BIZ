# GLM-5.3-Flash on 2× DGX Spark Enterprise Setup

**BETA（ベータ版）— 限定した参照profileで全モデルTP=2を試験済みです。本番運用・ハーネスの検収は未完了です。**

[English](README.md) · [セットアップ手順書](SETUP.ja.md) · [運用手順](docs/operations.md) · [検証範囲](docs/validation.md) · [構成](docs/architecture.md)

[起動設定の一括管理](docs/startup-configuration.ja.md)：コンテキスト長・キャッシュ・MTP・LPA・生成既定値・ノード設定を一つのTOMLにまとめ、実験用ランチャーと専用クライアントから使えます。

NVIDIAのGLM-5.3-Flash NVFP4を、DGX Spark相当のGB10システム2台で動かすためのセットアップ・検証ツールです。商用利用できるライセンスを軸に、資産の固定、検査結果の記録、戻せる運用を重視します。

独自コードは **Apache-2.0**。取り込んだMIT・Apacheの通知を保持します。重みとコンテナ内依存にはそれぞれの条件が適用されます。[第三者通知](THIRD_PARTY_NOTICES.md)を参照してください。EXL3/TR3重み、DFlash2重み、Mia現行AGPL版を導入する構成ではありません。

[商用利用・改造・再配布の整理](docs/licensing.ja.md)に、対象別の許諾範囲と義務をまとめています。[ハーネス連携](docs/harnesses.ja.md)では公式ZCodeと実験的なClaude Code接続を扱い、両方を必須の受け入れ対象にしています。実接続テストは未実施です。

## 企業利用に向けた取り組み

本プロジェクトは、**`nvidia/GLM-5.3-Flash-NVFP4`をDGX Spark相当の2台構成で、企業が評価・改造・運用しやすくすること**を目的としています。次の三点を一体として整備します。

- **ライセンスと出所の選択：** 商用利用できるMIT/Apache系の構成要素を優先し、採用元・版・通知を固定します。コード・重み・コンテナ・ハーネスそれぞれの条件は[ライセンス整理](docs/licensing.ja.md)に示します。
- **政治的な偏りと資料への忠実さの検証：** [FreedomBenchと業務文脈の追加試験](docs/freedombench.ja.md)で、政治的な問いへの回答・拒否・資料にない主張の挿入を調べます。対象範囲と失敗も示し、スコアだけで普遍的な思想的中立性を証明したとは扱いません。評価全体の検収は未了です。
- **実測に基づく性能調整：** MTP・LPA・CUDA融合・batching・並列方式を、タスク品質・メモリ・復旧と併せて検証します。[性能・品質施策台帳](docs/optimization-catalog.ja.md)に候補・証拠・保留理由をまとめ、次のGLMでも振り返れる比較基準を残します。

速度改善には、外部draftモデルを追加せず、**checkpoint同梱の標準MTPを使い、先読みトークン数は3（k=3）を選定**しました。MTP有効時の設定例も3で、[k=1との比較結果](docs/speculative-decoding.ja.md#k3の比較結果)を根拠としています。MTP自体の有効化はopt-inです。

企業利用に適するかは検収によって判断します。プロジェクト名を認定済みの意味にはせず、確認済みの範囲と残る条件を以下と各検証文書に示します。

## このベータ版で確認した範囲

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
| 他のMTP先読み数・画像・アプリ全体の品質・本番信頼性・最大性能 | **未検証** |

fixtureは元の幅・experts・選択したtensor bytesを保持しますが、層を切り詰めたモデルです。言語品質の評価には使えません。Marlin W4A16とNVIDIAのW4A4 recipeも同一の演算ではありません。[検証結果と限界](docs/validation.md)を区別して利用してください。

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

[GPU 1台のfixture手順](docs/validation.md#reproduce-the-single-gpu-fixture)で、実行完了・再現性・数値差を分けて確認できます。

TP=2ランチャーは**検収前**です。起動には実測したノード設定と検証記録が必要ですが、本ベータ版にはTP=2の検収を完了できる手順一式はまだありません。合格記録を手書きで作らず、`service plan`・`service preflight`で確認してください。残る検証は[運用手順](docs/operations.md)に記載しています。

## ローカルデータと開発

`state/`・`records/`・認証情報・実サイトの設定・重みをGitとDocker build contextへ含めません。公開するのはレビュー済みの要約です。

CPU検査と公開境界の確認は [CONTRIBUTING.md](CONTRIBUTING.md)、変更履歴は [CHANGELOG.md](CHANGELOG.md)、ライセンスは [LICENSE](LICENSE)・[NOTICE](NOTICE) を参照してください。
