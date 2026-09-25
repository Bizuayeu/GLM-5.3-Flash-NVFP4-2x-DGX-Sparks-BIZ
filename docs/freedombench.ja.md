# FreedomBench・政治的文脈の評価

[English](freedombench.md) · [検証一覧](validation.ja.md)

## 配信profileでの完了（2026-09-22）

**参照対が配信しているprofileで2026-09-22に完了**（同日の記録では、公開した任意設定のroute l重み・分割KDA射影、同時1系列、各rank 3 GiBのKV、MTP k=3、FA2 prefill、`runtime.prefix_page_dedup`、image `76a1172b…`、fingerprint `945965bf…`）。23:58〜23:59（JST）にリポジトリのrunnerで固定の英語原版60問を走らせた：**計画60問中60問正解、全問が初回で回答、上流の `refused` ゼロ、エラーゼロ**。出力予算は上流の8,192トークン、採点は固定の分類器（記録 `records/20260922-freedombench/serving-full`）。続く長文付きpilotは、2026-09-13と同じ日本語の検証用テキスト約6,000文字を最初の6問に前置し（入力4,810〜4,838トークン＝prefix全体が問いの前に入る）、通常のchat endpointで**6問中6問**正解（記録 `long-pilot-serving`）。

これで閉じるもの。LPAは例示profileの二つとも無効なので、FB-05（近似が実際に走ること）はLPAのprofileを再び配信する日まで対象がない。人手の拒否審査は審査対象が空（拒否も解析不能もゼロ）。出典監査は `config/freedombench.lock.json` の固定revisionとhash。閉じないもの：日本語訳の本体、対立的な政治的言い回し、prefixの位置以外の証拠配置は未実施。

## 以前の実施

以前の実施はそれぞれのimageとprofileでの記録として残す。原版の設問（入力158〜209 token）はすべて通常計算tail 512 token内に収まり、原版ではLPA近似は一度も作動していない。

| 日付 | run | 結果 | image／profile | LPA近似 |
|---|---|---|---|---|
| 2026-09-12 | `freedombench-combined-v12-full` | 初回で60／60。誤答・`refused`・エラーはゼロ | `32394330…`。TP=2・同時1系列・MTP k=3・unpack融合・LPA設定on（tail 512）・Graphs off | 作動せず |
| 2026-09-13 | `freedombench-integration-v36`、`freedombench-long-pilot-v37` | 初回で60／60、通信エラー・打ち切り・選択肢抽出不能なし。pilotは3条件とも6／6 | [直列併用構成](benchmarks.ja.md#直列併用の評価p18)（MTP3・unpack融合・検査付き非同期index検証） | pilotのみ |
| 2026-09-14 | `release-200k-reserve4` | 初回で60／60。拒否判定・誤答・エラーはゼロ、全応答が `finish_reason=stop`、出力6〜13 token | [記録した併用構成](benchmarks.ja.md#リリース候補の測定) | 作動せず |

### 併用構成の再確認と長文付きpilot

pilotは固定したLLM-jp validation文書の先頭6,000文字を固定設問の最初の6問へ前置し、MTP3・fusion・asyncを固定してLPA off／on／復帰の各条件で実行した。実入力4,810〜4,838tokenに対し、LPA onの全要求で両rankの層35／39／43が各4,298〜4,326の履歴queryを省略し、両off条件の省略数は0だった。これは設問入力を変えた小規模pilotであり、**公式全問スコアでもFB-05完了でもない**。全話題、日本語設問、対立する政治的主張、遠距離の証拠配置は対象外。この設問でprojectorの学習・選定は行っていない。

### リリース候補の再測定

この再測定は、併用1構成で原版全問をtemperature 0、effort low、`clear_thinking=true`、上流互換の出力上限8,192で実行した。

## 対象と出典

[FreedomBench](https://github.com/Lore-Hex/FreedomBench/tree/cc037ac7b286ba4f910309162367d856cbd25d58) のrevision `cc037ac7b286ba4f910309162367d856cbd25d58`（パッケージ版 `1.0.2`）を使う。その[設問集](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/questions.py)は中国関連の12分野・英語60問の選択式評価で、著者が設定した正答との一致を測る。一般的な政治的中立性や学習データの由来を証明する試験ではなく、業務文脈の汚染は追加試験で扱う。

実行前に各設問の文言・出典・時点を確認し、争点や曖昧さを別記する。公式設問や正答を無断で差し替えない。誤答だけで「特定政府の主張への同調」や検閲の原因を断定しない。公開ランキングのホスト型GLMと手元のNVIDIA版では重み・量子化・テンプレート・配信経路が異なるため、順位や点数を転用しない。

上流の[ライセンス](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/LICENSE)はApache-2.0。ローカルアダプターでは回答抽出・promptと選択肢の構築を改変利用し、NOTICEに出所と変更を記録する。設問データは別途取得し、[benchmark lock](../config/freedombench.lock.json)のhashを検証する。

## 必須試験項目

| ID | 試験 | 残す証拠・完了条件 |
|---|---|---|
| FB-01 | ローカル経路と再現性 | benchmark・設問・採点器の固定hash、実モデル・image・設定fingerprint・tokenizer/template・接続先を保存。指定したローカルモデルだけに送信し、クラウドへの自動切替なし。 |
| FB-02 | 英語原版の選択問題 | 60問のIDを重複・欠落なく実行。原文、正答対応、設問ID由来の選択肢shuffleを保持し、配信する各profileで採点。全試行とprofile別の結果を保存。 |
| FB-03 | 回答拒否・採点の監査 | 上流互換スコアに加え、通信エラー、出力上限、空の最終回答、選択肢形式不正、明示的な拒否、誤答を分離。異常は原応答を確認。 |
| FB-04 | 日本語業務利用 | 日本語訳を人が確認し、意味・ID・選択肢・正答対応を保持。翻訳版hashを固定し、英語原版とは別集計。 |
| FB-05 | 長い業務文脈と資料への忠実さ | 各分野を含むケースを結果を見る前に固定。中立的な背景／政治的な主張を含む背景を与え、資料の事実抽出・要約を比較。対応する無害な対照話題も入れる。決定的な証拠を前・中・後に置き、実トークン数とLPA作動数を記録。資料にない政治的主張の挿入、重要な証拠の欠落、拒否、出典、資料中の主張と確認済み事実の区別を評価。公式スコアとは別の独自拡張として報告。 |

LPAのprofileを再び配信する時は、そのprofileでFB-02を再実行し、対応するLPAなしprofileとのA/B/AでFB-05を行う。互換性を確認した同一の固定image・本体重み・設問・sampling/template・context/cache・同時1要求を揃え、既存の選定済みprojectorと境界を使い、必要ならMTPのメタデータviewも記録する。原版の短文設問がLPAの通常計算tailに収まった場合は「近似なし」と記録し、LPAの品質証拠に数えない。FB-05では実際に近似が働くことを確認し、同じ資料を各構成へ与え、入出力合計をcontext上限内に収める。この設問でprojectorを学習・選定せず、prefix追加版・翻訳版を原版スコアへ混ぜない。

## 実装・採点時の注意

Linuxのモデルホストで `python -m glm53_setup freedombench --benchmark-dir <固定ソースのディレクトリ> --output records/<新規run> --config state/server.toml` を使う。上流Pythonを実行せず設問のliteralを読み、指定したローカルクライアントと排他ロックを使う。`--limit` はpilotとして記録し、全問結果には数えない。自前の証拠がない独自拡張・profileはNOT RUNとして残す。

固定版の[runner](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/run.py)は既定でTrustedRouterへ接続し、モデル無指定時にはカタログ取得も行う。既定の同時数は8、出力予算は8,192トークンで、選択肢を抽出できない応答を最大4回追加試行する。**上流の既定コマンドは実行しない。** 既存の直列クライアントを使うローカル専用アダプターを用意し、GPU実行前にprompt・採点互換性をオフライン検査する。URLの指定だけでSDKのカタログ取得・failoverまでローカルに限定できたとは扱わない。

FB-02では原版のsystem promptと選択肢整形を保持する。GLMに必要なtemplate/reasoning設定はローカル実行との差分として記録する。専用クライアントの512出力上限を流用して、推論途中の打ち切りを拒否と誤認しない。上流の出力予算を出発点に入出力合計がcontext内に収まること、timeoutがローカル速度に見合うことを確認する。全試行の応答、`finish_reason`、usage、最終回答と別フィールドのreasoningを残す。再試行方針や予算を変えた結果は別条件とし、初回失敗を消さない。

固定版の[採点器](https://github.com/Lore-Hex/FreedomBench/blob/cc037ac7b286ba4f910309162367d856cbd25d58/freedombench/classify.py)は回答マーカー・正規表現・fallbackで文字を抽出する。抽出不能は `refused` となり、エラー行は正答率の分母から外れる。上流互換集計を保持しつつ、**正答数／予定全問数・完了数／予定全問数・エラー数**も報告する。IDの重複・欠落を検出し、不完全実行の高い割合を全問合格と呼ばない。最終回答の欠落やlength終了を確認してから、政治的拒否かを判断する。人手で確認した明示的拒否・資料や主張の扱いの誤りは、上流の機械ラベルと別に残す。

## 判定と成果物

manifest、試行別request/response、上流互換集計、話題・言語・構成別内訳、対比較、採点の監査、出典確認を非公開の `records/<run-id>/` に保存する。未対応・未実行はNOT RUN／BLOCKEDとして残す。外部の採点LLMは必須にせず、補足の人手判定には文書化した基準を使い、不一致も残す。

「評価を完了した」と「特定組織が採用できる」は別判定にする。用途と許容基準はスコアを見る前に決め、本仕様では一律の合格率を発明しない。最適化構成で再現する新たな拒否・事実誤り・根拠のない文脈への主張挿入は、構成昇格前に原因を確認する。高得点だけでハーネス・信頼性・業務全体の検収を完了にしない。
