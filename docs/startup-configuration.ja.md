# GLM起動設定の一括管理

[English](startup-configuration.md)

[コメント付きTOML](../examples/startup.example.toml)を `state/startup.toml` にコピーし、両Linuxノードに同じ内容を置きます。このファイルを**実験用referenceランチャー**と専用送信コマンドが共通で読みます。従来の `service` の通常運用認定ゲートは別に残ります。

| カテゴリ | 管理するもの |
|---|---|
| `runtime` | 固定イメージID、eager／decode Graph実行、独立EP／PPと層境界、seed、画像入力の切替 |
| `context` | 入出力合計のコンテキスト長、同時シーケンス数、prefillのチャンク予算 |
| `profiling` | 診断用のCUDAカーネル・launch計測。通常の速度測定時は無効 |
| `validation` | CUDA・indexer用、またはexpert実配置用の独立観測worker |
| `cache` | 各ランクのKV容量、要求ブロックサイズ、prefix cache、メモリ使用率、実験用unpack融合 |
| `mtp` | MTP有効化、下書きトークン数、モデルのメタデータview |
| `lpa` | LPA有効化、近似開始層、通常計算を残す末尾、損益分岐の閾値、クエリ省略、projectorとハッシュ |
| `api` | ローカルAPI・ランク間通信ポート、モデル名、パーサー |
| `generation` | 送信コマンドの生成既定値：出力長、temperature、reasoning、タイムアウト |
| `resources` | コンテナ上限、起動前の空き条件、実行中のメモリ余裕、自動停止期限 |
| `nodes` | 両ランクの実測済みfabricアドレス、interface、HCA、GID |

モデルID・revisionとビルドの基底イメージは [runtime.lock.json](../config/runtime.lock.json) が正典です。相対パスはTOML自身の位置が基準です。例外として `mtp.view` はHugging Faceキャッシュからの相対パスで、固定revisionを末尾に自動付加します。秘密鍵やトークンはこのファイルに入れません。

## 配布用の既定設定

配布用TOMLは、[200Kでの画像入力](vision.ja.md)を含む直列の最適化構成を既定にします。これは設定の選定であり、本番・ハーネス検収の完了を意味しません。既存の `state/startup.toml` は自動更新されません。

| 項目 | 既定値 |
|---|---|
| 実行 | TP=2、eager、1系列、204,800 token、chunk512 |
| 入力 | テキスト・ツール呼び出し・画像（`runtime.vision = true`）。動画は拒否 |
| キャッシュ | FP8、各rank 2.5 GiB、APC有効、checkpoint保持 `dense`、unpack融合有効、画像前処理キャッシュ0.1 GiB |
| 投機・近似 | MTP k=3。LPA無効（有効時はcut32／tail512／B128、未使用MLA query省略） |
| 検査・並列 | 非同期index検査、EP無効、PP分割なし |
| 生成 | temperature=0、max_tokens=512、reasoning_effort=low、clear_thinking=true |
| 資源 | コンテナ112 GiB、起動前空き108 GiB、実行中余裕2.5 GiB |

テキスト専用の代替は `runtime.vision = false`、`max_model_len = 262144`、`kv_cache_memory_bytes = 3221225472` です。その[256K確認](benchmarks.ja.md#256kでの実入力確認)は保護余裕4 GiBで実施しており、2.5 GiBでは未検証です。
| 実行期限 | `run_seconds=0`：時間による自動停止なし。メモリ監視は継続 |

**導入時はimage ID、両機の接続情報、MTP view、LPA projectorとhashを準備してください。** imageとprojectorのゼロ値は差し替え必須の仮値で、準備不足を理由に機能を黙って無効化しません。資材の配置は[運用手順](operations.ja.md#資材の保管場所とパス)が正典です。MTP／LPAは個別に無効化でき、基準比較ではAPC・保持・融合・非同期検査も明示的に戻します。

期限は起動時に固定されます。`run_seconds` の変更を稼働中の監視へ反映するには、[両rankの切替手順](launch-safety.ja.md#全レール検査と両rankの切替)で再起動します。設定ファイルの変更だけでは既存の期限は消えません。コンテキストを拡大するときは、以下の容量条件と実要求を別に検証します。

## コマンド

認証クライアント、allocatorの未指定／空文字、全HCA検査、両rankの停止前検査と切替は[起動契約と運用検証](launch-safety.ja.md)を参照してください。P10／P19／P22／E03の追加範囲であり、複数レール実通信などの未検収を機能実装と区別します。

**P22のGPU状態隔離・校正・最終併用・held-out参照評価を確認済みです。** LPAとprefix cachingを両方有効にする場合は `GLM53_APC_LPA_API=1` のimageが必要です。schedulerが全状態を揃えて復元したprefixと `lpa.break_even_tokens` から適用を決めます。テンプレートは[P22の校正](benchmarks.ja.md#apc優先lpaの損益分岐計測p22)に基づく保守的な閾値128を使います。最初の近似以降は、通常計算する末尾・decodeを含めて共有登録を止めます。このモードの `startup ask` は判断をサーバーへ任せます。テンプレートは `lpa.enabled = false` を既定とし、バッチ入力に限って用途ごとに有効化します（近似した要求は共有cacheに何も登録しないため）。有効にしている間、要求に `"vllm_xargs": {"glm53_lpa_mode": "off"}` を指定すると通常計算し、通常状態の共有cacheを育てられます。`api.prompt_tokens_details = true`（任意キー、未指定は無効）は `--enable-prompt-tokens-details` を付け、`usage.prompt_tokens_details.cached_tokens` で復元prefix長を返します。無いとvLLMは `null` を返し、cacheが当たっていてもハーネスの表示は0のままです。APCなしの通常の `startup ask` はH=0として同じ閾値を使います。[実装契約](apc-lpa-design.ja.md)を参照し、更新するTOMLには新しい閾値キーを明示してください。

`runtime.expert_parallel=false` が既定です。有効にすると両rankへ `--enable-expert-parallel` を追加し、TP=2／DP=1、精度、固定KV予算を維持します。`GLM53_EXPERT_PARALLEL_API=1` を持つイメージが必要ですが、このmarkerは設定対応を表し、EPの検収済み証明ではありません。初期範囲はeager・1／2系列・MTP/LPA/fusion/APCなしです。既存TOMLにも新しいキーを明示し、欠落時の暗黙fallbackは設けません。使用前に[EPの独立評価手順](performance-investigation.md#expert-parallel-p21)を参照してください。

`runtime.vision` は任意キーで、未指定はfalse、テンプレートは `true` です。`false` は `--language-model-only` を残し、視覚塔を読み込まずテキスト・ツール専用で動かします。`true` は両rankからこのフラグを外し、`--limit-mm-per-prompt '{"video": 0}'` を付けます。**`vision = true` でも動画入力は無効で、送ると拒否されます。受け付けるのは画像だけです。** 理由は起動時のメモリです。vLLMは最大の入力1件を一度エンコードしてメモリを見積もり、このチェックポイントの動画の上限（30,000 token＝120,000パッチに制限済み）は画像1枚（最大8,000 token）よりはるかに大きいためです。1 promptあたりの画像枚数はvLLMの既定のままです（チャットハーネスは過去の画像を毎ターン送り直すため）。Vision有効時は `cache.mm_processor_cache_gb`（任意キー、未指定は0.1）が `--mm-processor-cache-gb` を決めます。これは前処理済み画像のキャッシュで、vLLMはAPIプロセスとエンジンプロセスの両方に持ち、どちらもrank 0にだけ載ります。vLLMの既定4 GiBのままだと、headのホストRAMを最大8 GiB使い得ます。上限より大きい画像はキャッシュせずに処理し（警告のみ）、拒否はしません。視覚塔はBF16で量子化の対象外です（両方の除外リストに `model.visual*` があり、固定vLLMも量子化設定なしで組み立てる）。実測、headのメモリ余裕、未解決の事項は[200Kでの画像入力](vision.ja.md)にあります。検証fixtureはこのキーに関係なくテキスト専用で読み込みます。

`runtime.pipeline_parallel_size=1` はTP=2を維持し、2にすると同じ2台でTP=1／PP=2を選びます。`GLM53_PIPELINE_API=1` を持つイメージが必要です。`pipeline_split_layer` は前段stageの層数で、既定候補24なら24／21層に分け、この固定モデルでは各stageに21 MoE層ずつを置けます。容量を保証する値ではありません。両stageにMLAが必要なため、境界の許容範囲は4〜43です。初期範囲は1系列・eager・EP/MTP/LPA/fusion/APCなし。[小層の検証結果](component-validation.ja.md#8層ppの観測)は、全モデル速度・長文の数値同値・本番運用の認定ではありません。TOML更新時は両キーを明示してください。

`validation.expert_worker=true` は、実際のexpert配置・kernel・parameter情報を返す型付きRPC `expert_info` を有効にします。独立したeager TP2の基準／EP条件、最大2系列が対象で、他の観測worker・MTP/LPA/APC・PPとは併用しません。層のhash観測は明示的な `pipeline_observe` RPCで初めて開始するため、性能測定中はそのhookを入れません。実験用のローカル制御経路であり、業務利用の認定ではありません。

`runtime.index_checks` は `auto`／`sync`／`async`（配布既定） を選びます。autoはeagerで同期検査、Graphで非同期検査を使い、従来の動作を維持します。asyncを明示すると、独立評価したeagerの非同期検査を選べます（`GLM53_ASYNC_INDEX_CHECK_API=1` が必要）。範囲検査は常に実施します。asyncで不正indexを検出するとCUDA contextが使えなくなる場合があるため、両rankを再起動します。Graphではsyncを拒否します。MTP/LPA/fusionの直列併用はP18／P22で範囲を限定して確認済みです。

コンテキスト長や同時数を変更する前に、[KV容量とRAMの条件](#kv容量とramの条件)も確認してください。

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

## KV容量とRAMの条件

**最大長の要求をB本同時に保持するなら、入出力合計の上限Cに対してB×C token分を収容できる容量の確認が必要です。** `max_model_len`は入力と生成の合計上限、`max_num_seqs`は同時実行の上限です。この二つを設定するだけで、最大長×同時数のKVが確保・検収されるわけではありません。

このランチャーの`cache.kv_cache_memory_bytes`は、**各rankで要求間共有する固定KV poolのバイト予算**です。1 GiBを指定したまま同時数を1→2にしても、各rankのKV予算は1 GiBのままです。要求1本あたり1 GiBでも、2台の予算を自由に合算した一つのpoolでもありません。バイト指定時は`gpu_memory_utilization`によるKV容量の自動推定を使わないため、この比率をRAM全体の保護上限として扱いません。[vLLMの設定仕様](https://docs.vllm.ai/en/latest/configuration/engine_args/#kv-cache-memory-bytes)

実行中に必要なcacheは、保持中の各要求の入力＋生成済みtokenに従ってpool内のblockを消費します。最大長を同時に保証したい場合は、出力予算も含む最大条件で検証します。GLMは疎MLA・IndexPool・系列ごとのKDA状態を併用するため、一般的なdense attentionの単純なbytes/token式をそのまま使わず、**固定runtimeのcache spec・block整列・各groupの容量と状態slot数**で見積もります。MTP等の追加状態も別途含めます。

各ノードで、重み＋KV/cache状態＋activation・indexer等の一時領域＋MTP/Graph等の追加領域＋CPU/OS・他負荷＋運用余裕が、利用可能な統合RAMに収まる必要があります。KV poolを固定しても、context・chunk・同時数に依存する別の割当が増えることはあります。コンテナ上限と`reserve_gib`は保護手段であり、容量適合や無停止の保証ではありません。以前の 256K テキスト専用 profile は実測 121 GiB のホストでこの保護に接していました。available は 4.5 GiB 前後で推移し、保護余裕 4 では監視停止（`stop-reason: memory-reserve`）が 2 回発生し（2 回目は 16,859 token の近似要求の最中）、その後 3 へ下げました。現在の画像入力構成では、配信中の head の空きは約 4.0〜4.2 GiB で（[実測](vision.ja.md#メモリ最終構成)）、テンプレートは 2.5 です。値は小数でも指定できます。変える前に、この保護が捉えられる範囲を見積もってください。監視は `MemAvailable` を 2 秒ごとに読み、実測では割り込んだ標本からコンテナ終了まで約 9 秒かかりました。実行中のコンパイルで測った 0.15 GiB/秒の下降なら、reserve から約 1.6 GiB 下まで沈み得ます。参照ホストではカーネル・コンテナとも OOM kill の記録はなく、ホスト側のページは 16 GiB の swap が受けるため、reserve を割った先の危険はワーカー内の GPU 確保の失敗です。その場合は監視停止ではなく、エンジンが突然終了します。ドライバの `NV_ERR_NO_MEMORY` カーネルメッセージは空き 4 GiB 以上、主にロード中に出るもので、底の目印にはなりません。

KVが不足すれば起動が拒否される場合があり、実行時は待ちやpreemption・再計算により性能が落ちることがあります。固定KV poolが勝手に必要量まで拡張されるわけではありません。KV以外の割当やRAM予算が不足すればOOMやガード停止も起こり得ます。[vLLMのpreemption説明](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption)

起動時のcache容量・最大並列度は計算上の目安として保存し、**意図する入出力長×同時数の実要求、preemption回数、両rankの空きメモリ最小値、OOM・ガード停止**を確認してから対応範囲を表明します。現行preflightの合格は、最大長×同時数の収容試験の代わりにはなりません。実測範囲は[標準batchingの独立評価](benchmarks.ja.md#標準batchingの独立評価)を参照してください。

## 現行イメージの契約

現行ソースから再ビルドし、`GLM53_LPA_API=2`、worker `glm53_setup.runtime.lpa.LPAWorkerExtension`、RPC `lpa_configure`・`lpa_report`、読み取り専用マウント `/lpa/projector.pt` を使います。preflightはmarkerなしのイメージを拒否します。両ノードの固定イメージIDを確認し、起動設定を更新してから起動します。

旧コマンド・旧設定名・旧イメージへのフォールバックはありません。新ソースをホストへ置くだけでは稼働中のイメージは変わらないため、再ビルドと実機検証を行います。

## LPAとMTP・制約

`runtime.enforce_eager=true` が既定です。`false` は実験用のdecode Graph経路を明示的に選び、`CompilationMode.NONE`・`FULL_DECODE_ONLY`・capture size `[1]` を渡します。prefillはcompileせず、イメージには `GLM53_DECODE_GRAPH_API=1` が必要です。単体検証の範囲は同時1シーケンス・LPA/MTP/APCなしで、融合は独立設定です。これは全モデルの受入完了を意味しません。併用と複数系列は後段の検収まで起動設定で拒否します。

Graph経路では内部候補indexの範囲検査をGPU上で非同期に行います。不正indexを黙って許容せずdevice assertにしますが、通常のPython例外と異なりCUDA contextが使用不能になり得るため、障害時は両rankを停止して再初期化します。Graph用のメモリ保持・起動時capture時間も比較対象です。eagerの検査方式も `runtime.index_checks` に従い、配布既定はasyncです。

`validation.component_worker=true` は部品検証専用workerを選び、`GLM53_COMPONENT_API=1` を持つイメージを要求します。eager・同時1シーケンス・LPA/MTP/prefix cacheなしの独立構成です。型を制限したRPCでindexerの候補・時間を採取し、排他的なリクエスト間でunpack融合を切り替えてA/B/Aを検証できます。Reuse/Reindexを本番適用する設定ではありません。制御クライアントは一つに限定します。

`cache.fused_unpack=true` が配布既定です。falseならTorchの参照変換を使い、trueなら656バイトのMLAキャッシュからのFP8変換とFP32スケール乗算を一つのTritonカーネルで処理します。イメージに `GLM53_FUSED_UNPACK_SUPPORTED=1` が必要で、preflightで確認します。部品一致と全モデルA/B／直列併用の実測範囲で使用します。広範な品質・本番検収とは区別します。候補集合の変更や層間のKV共有は行いません。

CUDA融合の実モデルA/Bとindexerの採取結果・検証限界は [部品検証記録](component-validation.ja.md) を参照してください。

`mtp.enabled` と `lpa.enabled` を個別に切り替えます。MTP有効時は [prepare_mtp_view.py](../tools/prepare_mtp_view.py) で作成したviewとBF16 Triton下書きバックエンドを使います。LPA有効時は `runtime.lpa_image` を選び、projectorを読み取り専用でマウントしてworker拡張を有効にします。併用にはMTP対応を明示したLPA workerを含むイメージが必要で、旧LPAイメージは併用指定を拒否します。

LPAはリクエストごとの入力長が必要です。専用クライアントが実際のテンプレートでトークン数を求め、worker設定→生成→トークン数の一致確認→LPA解除まで行います。入力全体が `lpa.tail` に収まる短文は通常計算です。制御するクライアントは一つに限定してください。専用CLI同士はhead上のロックで直列化しますが、直接APIを呼ぶ他クライアントまでは調停しません。専用送信コマンドは非ストリーミングのテキスト・ツール会話用です。一般ハーネスや本番運用の認定は別です。

範囲はTP=2・テキスト／ツール・Marlin W4A16・FP8 KVです。LPAには同時1シーケンス・eager実行が必要で、prefix cacheは上記のP22経路を使います。LPAなしのthroughput用構成では同時数を増やせますが、[性能調査計画](performance-investigation.ja.md)に従ってタスク品質・状態整合・資源を別途検証します。MTPの下書き数は1と3です。コンテキスト長・チャンク・キャッシュ量・cut・tailを変えた場合は再測定が必要で、設定検査の合格は品質や必要メモリの保証ではありません。[LPA](lpa.ja.md) と [MTP](speculative-decoding.ja.md) に検証範囲を記載しています。
