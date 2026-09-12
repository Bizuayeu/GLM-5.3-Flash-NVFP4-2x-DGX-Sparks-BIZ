# GLM起動設定の一括管理

[English](startup-configuration.md)

[コメント付きTOML](../examples/startup.example.toml)を `state/startup.toml` にコピーし、両Linuxノードに同じ内容を置きます。このファイルを**実験用referenceランチャー**と専用送信コマンドが共通で読みます。従来の `service` の通常運用認定ゲートは別に残ります。

| カテゴリ | 管理するもの |
|---|---|
| `runtime` | 通常／LPA用の固定イメージID、eager実行、seed |
| `context` | 入出力合計のコンテキスト長、同時シーケンス数、prefillのチャンク予算 |
| `profiling` | 診断用のCUDAカーネル・launch計測。通常の速度測定時は無効 |
| `validation` | LPA/MTPとの結合前にCUDA・indexerを調べる専用worker |
| `cache` | 各ランクのKV容量、要求ブロックサイズ、prefix cache、メモリ使用率、実験用unpack融合 |
| `mtp` | MTP有効化、下書きトークン数、モデルのメタデータview |
| `lpa` | LPA有効化、近似開始層、通常計算を残す末尾、クエリ省略、projectorとハッシュ |
| `api` | ローカルAPI・ランク間通信ポート、モデル名、パーサー |
| `generation` | 送信コマンドの生成既定値：出力長、temperature、reasoning、タイムアウト |
| `resources` | コンテナ上限、起動前の空き条件、実行中のメモリ余裕、自動停止期限 |
| `nodes` | 両ランクの実測済みfabricアドレス、interface、HCA、GID |

モデルID・revisionとビルドの基底イメージは [runtime.lock.json](../config/runtime.lock.json) が正典です。相対パスはTOML自身の位置が基準です。例外として `mtp.view` はHugging Faceキャッシュからの相対パスで、固定revisionを末尾に自動付加します。秘密鍵やトークンはこのファイルに入れません。

## コマンド

リポジトリ直下で生成される起動条件を確認します。これはWindowsでも実行できます。

```sh
python -m glm53_setup startup plan --rank 0
python -m glm53_setup startup plan --rank 1
```

各Linuxノードの `startup preflight --rank N` でモデル・fabric・イメージID・空きメモリを確認できます。worker側でrank 1を先に、head側でrank 0を後に、それぞれの端末で起動します。

```sh
python -m glm53_setup startup start --rank 1 --experimental
python -m glm53_setup startup start --rank 0 --experimental
```

起動コマンドは前面に残り、自分のコンテナを監視します。端末を維持してください。`resources.run_seconds` の期限には**モデルのロード時間も含まれます**。長時間使う場合は起動前に延ばします。Ctrl+C・期限到達・空きメモリ不足でそのランクを停止します。分散実行に異常が出た場合は両ランクを停止します。コンテナと `records/` のログ・起動設定は残し、自動削除や自動再起動はしません。

APIが準備できたら、headの別端末から送信できます。

```sh
python -m glm53_setup startup ask --prompt "GLM-OK とだけ返してください。"
python -m glm53_setup startup ask --request request.json
python -m glm53_setup startup status --rank 0
python -m glm53_setup startup stop --rank 0
```

workerの状態確認・停止はworker上で `--rank 1` を使います。全コマンドで `--config 設定ファイル.toml` を指定できます。設定を編集したら両ランクを停止・再起動してください。送信時には起動中の設定との一致を検査します。`generation` は専用送信コマンドの既定値で、他のAPIクライアントの生成設定はそのクライアント側で指定します。

## 自動停止と連続稼働

`resources.run_seconds` はロード時間込みの自動停止期限（秒）です。**`0` で時間制限なし**、正の整数で指定秒数後に停止します。負数は受け付けません。どちらの場合も `resources.reserve_gib` によるメモリ保護は有効です。設定は起動時に読み込むため、ファイル編集だけでは起動中の監視プロセスの期限は変わりません。

```toml
[resources]
# 既存のresources節にある、この値を変更します。
run_seconds = 0
```

これは予定時刻で停止しない設定であり、24時間365日の可用性を保証するものではありません。OS起動時の自動起動・障害時の両ランク協調再起動・冗長構成への切り替えは未実装です。前面の監視プロセスを維持する必要があり、本番・ハーネス検収も未完了です。

## 現行イメージの契約

現行ソースから再ビルドし、`GLM53_LPA_API=2`、worker `glm53_setup.runtime.lpa.LPAWorkerExtension`、RPC `lpa_configure`・`lpa_report`、読み取り専用マウント `/lpa/projector.pt` を使います。preflightはmarkerなしのイメージを拒否します。両ノードの固定イメージIDを確認し、起動設定を更新してから起動します。

旧コマンド・旧設定名・旧イメージへのフォールバックはありません。新ソースをホストへ置くだけでは稼働中のイメージは変わらないため、再ビルドと実機検証を行います。

## LPAとMTP・制約

`validation.component_worker=true` は部品検証専用workerを選び、`GLM53_COMPONENT_API=1` を持つイメージを要求します。eager・同時1シーケンス・LPA/MTP/prefix cacheなしの独立構成です。型を制限したRPCでindexerの候補・時間を採取し、排他的なリクエスト間でunpack融合を切り替えてA/B/Aを検証できます。Reuse/Reindexを本番適用する設定ではありません。制御クライアントは一つに限定します。

`cache.fused_unpack=false` が既定で、Torchの参照変換を使います。有効にすると、656バイトのMLAキャッシュからのFP8変換とFP32スケール乗算を一つのTritonカーネルで処理します。イメージに `GLM53_FUSED_UNPACK_SUPPORTED=1` が必要で、preflightで確認します。まだ実験候補であり、部品の数値一致・attentionと状態の検査・トレースを止めたフルモデルA/Bを経て採否を決めます。候補集合の変更や層間のKV共有は行いません。

CUDA融合の実モデルA/Bとindexerの採取結果・検証限界は [部品検証記録（英語）](component-validation.md) を参照してください。

`mtp.enabled` と `lpa.enabled` を個別に切り替えます。MTP有効時は [prepare_mtp_view.py](../tools/prepare_mtp_view.py) で作成したviewとBF16 Triton下書きバックエンドを使います。LPA有効時は `runtime.lpa_image` を選び、projectorを読み取り専用でマウントしてworker拡張を有効にします。併用にはMTP対応を明示したLPA workerを含むイメージが必要で、旧LPAイメージは併用指定を拒否します。

LPAはリクエストごとの入力長が必要です。専用クライアントが実際のテンプレートでトークン数を求め、worker設定→生成→トークン数の一致確認→LPA解除まで行います。入力全体が `lpa.tail` に収まる短文は通常計算です。制御するクライアントは一つに限定してください。専用CLI同士はhead上のロックで直列化しますが、直接APIを呼ぶ他クライアントまでは調停しません。専用送信コマンドは非ストリーミングのテキスト・ツール会話用です。一般ハーネスや本番運用の認定は別です。

範囲はTP=2・テキスト／ツール・Marlin W4A16・FP8 KVです。LPAには同時1シーケンス・eager実行・prefix cache無効が必要です。LPAなしのthroughput用構成では同時数を増やせますが、[性能調査計画（英語）](performance-investigation.md)に従ってタスク品質・状態整合・資源を別途検証します。MTPの下書き数は1と3です。コンテキスト長・チャンク・キャッシュ量・cut・tailを変えた場合は再測定が必要で、設定検査の合格は品質や必要メモリの保証ではありません。[LPA](lpa.ja.md) と [MTP](speculative-decoding.ja.md) に検証範囲を記載しています。
