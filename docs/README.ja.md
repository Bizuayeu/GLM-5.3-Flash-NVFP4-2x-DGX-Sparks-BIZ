# 文書一覧

[English](README.md)

文書ごとに役割を一つ決め、他の文書は内容を複製せずリンクで参照します。利用者向けの文書はすべて英日の対（`name.md`／`name.ja.md`）です。Changelog・エージェント向け指示・ライセンスと通知の本文だけは意図して英語のみとします。利用者に見える手順を変える際は両方を更新します（[Contributing](../CONTRIBUTING.ja.md)）。

## 入口

| 文書 | 役割 | EN | JA |
|---|---|---|---|
| README | 導入するもの、対応機体、確認した範囲、業務利用の目的（BIZ） | [EN](../README.md) | [JA](../README.ja.md) |
| セットアップ手順書 | 機体確認から受け入れまでの順序付きゲート | [EN](../SETUP.md) | [JA](../SETUP.ja.md) |
| Contributing | CPU検査、公開監査、貢献の規則 | [EN](../CONTRIBUTING.md) | [JA](../CONTRIBUTING.ja.md) |
| Changelog | 変更履歴と版ごとの検証状態 | [EN](../CHANGELOG.md) | — |
| リポジトリ指示 | このcheckoutを編集するAIエージェント・運用者向けの規則 | [EN](../AGENTS.md) | — |
| ライセンス・通知 | Apache-2.0本文、帰属、第三者の出所 | [LICENSE](../LICENSE)、[NOTICE](../NOTICE)、[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)、[LICENSES/](../LICENSES/) | — |

## 導入・運用

| 文書 | 役割 | EN | JA |
|---|---|---|---|
| 運用手順 | 資材の保管場所、取得、機体準備、起動ゲート、復旧 | [EN](operations.md) | [JA](operations.ja.md) |
| 起動設定 | カテゴリ別の起動TOML、KV／RAM条件、imageの契約、LPA／MTPの制約 | [EN](startup-configuration.md) | [JA](startup-configuration.ja.md) |
| QSFPネットワーク | QSFP直結とNetworkManagerの永続profile | [EN](qsfp-network.md) | [JA](qsfp-network.ja.md) |
| NCCL検証 | 2台のcollective診断とその限界 | [EN](nccl-validation.md) | [JA](nccl-validation.ja.md) |
| 起動契約 | APIクライアント認証、allocator伝達、全レール検査、両rank切替と復旧、APC履歴検証 | [EN](launch-safety.md) | [JA](launch-safety.ja.md) |
| 構成 | パッケージ配置と検証境界 | [EN](architecture.md) | [JA](architecture.ja.md) |

## 検証・受け入れ

| 文書 | 役割 | EN | JA |
|---|---|---|---|
| 検証範囲 | 証拠と本番認定の区別、GPU 1台のfixture、残る検収ゲート | [EN](validation.md) | [JA](validation.ja.md) |
| 部品検証 | CUDA／indexer部品、Graph fixture、PP／EP／APCのfixture、履歴fixture | [EN](component-validation.md) | [JA](component-validation.ja.md) |
| ベンチマーク | TP=2ベンチの方法と全モデルのベンチ実行（P08、P11、P13〜P15、P17〜P19、P21、P22、保持） | [EN](benchmarks.md) | [JA](benchmarks.ja.md) |
| 画像入力 | 200KでのVision：設定、選定の経緯、実測、限界 | [EN](vision.md) | [JA](vision.ja.md) |
| FreedomBench | 政治的文脈の評価：必須試験と予備実測 | [EN](freedombench.md) | [JA](freedombench.ja.md) |
| ハーネス | ZCode／Claude Codeの接続方針と受け入れ試験一覧 | [EN](harnesses.md) | [JA](harnesses.ja.md) |
| ライセンス整理 | 対象別の商用利用・改造・再配布の可否 | [EN](licensing.md) | [JA](licensing.ja.md) |

## 最適化

| 文書 | 役割 | EN | JA |
|---|---|---|---|
| 推論最適化の全体像 | 各施策が効く段階、採用した構成、用途別の構成 | [EN](optimization-overview.md) | [JA](optimization-overview.ja.md) |
| 施策台帳 | 施策ID P01〜P22／E01〜E03、採否、再評価条件、比較記録の共通項目 | [EN](optimization-catalog.md) | [JA](optimization-catalog.ja.md) |
| 性能調査 | 計測手順：launch・同期、タスクbatch、EP、TP対PP | [EN](performance-investigation.md) | [JA](performance-investigation.ja.md) |
| 投機的デコーディング | MTPのメタデータview、k=1・k=3の実測 | [EN](speculative-decoding.md) | [JA](speculative-decoding.ja.md) |
| LPA | 後段Prefill近似：仕組み、使用範囲、再現、証拠 | [EN](lpa.md) | [JA](lpa.ja.md) |
| APC優先LPAの設計 | prefix cachingとLPAを併用する共有cacheの契約（P22） | [EN](apc-lpa-design.md) | [JA](apc-lpa-design.ja.md) |
| 候補順序 | 参照imageでのsparse候補の順序正規化 | [EN](candidate-order.md) | [JA](candidate-order.ja.md) |
| Indexer再利用 | CSA2の候補再利用・限定再採点の部品 | [EN](indexer-reuse.md) | [JA](indexer-reuse.ja.md) |

## 正典の所在

| 事実 | 正典 |
|---|---|
| モデルID、固定revision、base image digest、ローカル参照タグ、固定vLLM source commit | [config/runtime.lock.json](../config/runtime.lock.json) |
| 起動設定のスキーマと全キー | [examples/startup.example.toml](../examples/startup.example.toml)。説明は[起動設定](startup-configuration.ja.md) |
| MTPの投機設定例 | [examples/speculative.mtp1.json](../examples/speculative.mtp1.json)、[speculative.mtp3.json](../examples/speculative.mtp3.json) |
| FreedomBenchの設問・正答の固定 | [config/freedombench.lock.json](../config/freedombench.lock.json) |
| 施策ID、採否、再評価条件 | [施策台帳](optimization-catalog.ja.md) |
| 実測値とその条件 | [ベンチマーク](benchmarks.ja.md)、[画像入力](vision.ja.md)、[投機的デコーディング](speculative-decoding.ja.md)、[LPA](lpa.ja.md)、[部品検証](component-validation.ja.md)、[候補順序](candidate-order.ja.md)、[Indexer再利用](indexer-reuse.ja.md)、[NCCL検証](nccl-validation.ja.md)、[FreedomBench](freedombench.ja.md) |
| APC／LPAの共有状態契約（N・H・T・R・B） | [APC優先LPAの設計](apc-lpa-design.ja.md) |
| クライアント認証、allocator、レール、切替・復旧の契約 | [起動契約](launch-safety.ja.md) |
| checkpoint・MTP view・projector・image・stateの保管場所 | [運用手順](operations.ja.md#資材の保管場所とパス) |
| どちらのランチャーか：`startup`（実験用reference）と `service`（ゲート付き候補） | [運用手順](operations.ja.md#二つのランチャー) |
| ホストカーネルの要件、`7.0.0-1019-nvidia` のRoCE失敗と `kho=off` の回避策 | [運用手順](operations.ja.md#ホストカーネルと複数ノードroce) |
| 対象別のライセンス許諾と義務 | [ライセンス整理](licensing.ja.md)、[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) |
| ハーネス受け入れ試験と実施状態 | [ハーネス](harnesses.ja.md) |
| ZCodeの権限モード、モデル上限と圧縮予算の規則、既存ファイルガードのhook | [ハーネス](harnesses.ja.md#zcodeの権限モードモデル上限既存ファイルガード)、スクリプトは [examples/zcode-hooks/](../examples/zcode-hooks/) |
| 検証範囲と残る検収ゲート | [検証範囲](validation.ja.md) |
| 生応答、trace、全反復、失敗、実機固有の値 | 非公開の `records/<run-id>/`。配布せず、公開文書には検証した要約を置く |
| 非公開の実装計画とStage状態 | `docs/*PLAN*.md`。Git追跡外・公開対象外で、`docs/IMPLEMENTATION_PLAN.md` がローカルの索引 |
| サイト設定と取得状態 | `state/`。Git追跡外 |

## 約束事

- 変更履歴は[Changelog](../CHANGELOG.md)とGitに置き、各文書に「何を直したか」を溜めない。
- 実測値は正典の文書に一度だけ、image・source・負荷条件とともに書く。他の文書はリンクで参照する。
- 版数は `pyproject.toml` が所有し、版ごとの内容は[Changelog](../CHANGELOG.md)に書く。`python tools/check_publication.py` は素のsemantic versionを必須とし、`records/`・計画書・リポジトリ外へのリンクを拒否する。
- 本リポジトリ外の関連研究（Euryaleの投機draft研究など）は配布物に含まれないため、READMEでリンクなしに説明する。他の文書では言及に留める。
