# 起動契約と運用検証

[English](launch-safety.md)

P10（メモリ）、P19／P22（APC）、E03（運用）を拡張する項目です。高速化施策として重複計上しません。実装とCPU契約の確認に加え、実コンテナ・全モデルの回帰結果を採用前に別途記録します。

## モデルAPIクライアント

共通送信処理は非空の `API_KEY` を優先し、それが空／未設定なら非空の `VLLM_API_KEY` を使います。両方とも空／未設定ならAuthorizationを送りません。`server ask`、それを使うベンチ、profiler制御、componentの準備確認が対象です。宛先originはモデルAPIとして明示し、同一originを含めredirectは拒否します。ダウンロード経路には適用しません。キーを設定やfingerprintへ入れず、HTTP例外にはヘッダーや応答本文を保存しません。401／403は失敗として記録し、成功した測定から欠測として除きません。

固定vLLMの認証middlewareが保護するのは `/v1`・`/v2`・`/inference`・`/cohere` です。`/health`・`/metrics`・`/tokenize`・`/collective_rpc`・`/reset_prefix_cache`・profiler制御は保護しません。Bearer送信だけでサーバー側の保護範囲は変わりません。APIはloopback限定を維持します。今回追加するのはクライアント認証対応であり、公開サーバー用の認証層ではありません。 `api.dev_endpoints = true` は、本来devモードで動かないprofileにもdev経路（cache reset・collective RPC・sleep）を載せます。これらも同様に無認証です（[サーバー設定](server-configuration.ja.md#コマンド)）。

## allocatorと共通起動設定

任意の `runtime.cuda_allocator_conf` を `PYTORCH_CUDA_ALLOC_CONF` へ渡します。省略はimage／runtime既定を維持し、文字列は明示的な空文字を含めそのまま渡します。確認した基準imageにallocator環境設定はありません。hidden-state KV connectorの互換性を検収したことにはなりません。

環境変数の上書きは起動元で一度だけ解決します。

```sh
python -m glm53_setup server freeze --config state/server.toml --output state/launch.json
python -m glm53_setup server plan --config state/server.toml --launch state/launch.json --rank 0
```

環境変数は空文字でも「存在」すればTOMLより優先します。同じ凍結JSONを両rankへ配布し、start／preflightの `--launch` へ渡します。rank側の環境変数は再解決しません。ローカルallocator環境変数がある直接起動では、凍結済みmanifestを必須とします。manifestは解決済みprofileとlockに結び付くfingerprintを含み、APIキーは含みません。`freeze` は既存manifestを上書きしません。

## 全レール検査と両rankの切替

各nodeの主レールは従来の `hca`・`interface`・`local_ip`・`gid_index`（port 1）です。任意の `additional_rails` に同じ項目と `port` を持つレコードを追加します。全レールで共通GID index、port／NIC／IPの重複排除、Ethernet portとlinkの稼働、IPv4対応RoCE v2 GID、当該NICへのIP割当を確認します。カンマ区切りのdevice文字列は受けず、構造化した設定を使います。NCCLには全HCA／portを完全一致指定し、socket bootstrapは主NICを使います。設定検査の成功と複数レール実通信の検収は別です。

単一レールでも `=hca:1` とport 1を明示します。portを省略すると、そのHCAの全portが対象となり、検査した範囲を超えるためです。[NVIDIAのNCCL HCA指定仕様](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)を参照してください。

```sh
python -m glm53_setup cluster switch --config state/server.toml \
  --hosts spark-head spark-peer --checkout /srv/glm53/source \
  --remote-config /srv/glm53/state/server.toml \
  --output records/switch-run
```

両hostには同じ監査済みcheckout・image・資材を用意します。停止前のsource識別検査は不一致を拒否するため、切替の前に両方のcheckoutを更新します。稼働中の対は起動時のprofileパスを `state/startup-rank<N>.json` に記録しており、切替失敗後の復旧はその記録パスから旧profileを再起動します。profileファイルを改名する場合（1.1.0で `state/server.toml` へ移動）は移動ではなく複製し、新しい対の準備完了を確認してから旧ファイルを削除してください。`--remote-config` はprojector相対パスのLinux側基準、共通の凍結manifestは設定値を指定します。新しい対が complete になると、両rankは `--config` の本文をそのパスへ書き込みます。rank上でファイルを読むコマンド（`server agreement`・`server mojibake`・ベンチ）が、稼働中のprofileを見るようにするためです。各rankは、自分が起動されたprofileと一致する本文だけを受け付け、内容の違う旧ファイルは `<名前>.bak-<UTC時刻>-<旧fingerprintの先頭8桁、または unparsed>` として隣に残し、renameひとつで置き換えます。切替が失敗した時と旧対へ復旧した時は何も書かず、書き込みの失敗はjournalの `config` に記録するだけで対を巻き戻しません。`--no-send-config` を付けるとリモートのファイルに触れません。`cluster resume` は、`--config` を渡され、かつ新しい対がreadyだった時だけ書き込みます。必要に応じて `--ssh-config` を指定します。停止前に両rankの資材・fabricと共通source／image／model／profile／allocatorを検査し、停止直前にも再照合します。重みの照合はindex hash・shardサイズ・ローカルfile識別であり、元の重み完全性検査の代わりではありません。ロード用の空きメモリ検査は停止後に行います。[他のGPUコンテナの検査](operations.ja.md#フルモデルの起動検査)は両方の時点で行います。停止前はラベルを持つ稼働中の旧い対を除外し、停止後は他のGPUコンテナが動いていれば起動しません。

停止前の不合格では稼働中コンテナを維持します。停止後の不合格では今回予約した起動分だけを停止し、記録済みの旧profileで復旧を試みます。停止確認が取れない場合は競合する復旧起動を避けます。結果に失敗・cleanup・復旧を分けて残します。停止を伴う切替であり、原子的な無停止切替ではありません。稼働rankに設定パスの記録がない場合は復旧条件が揃わないため停止前に拒否します。他用途のコンテナは停止しません。ランチャー外のGPUコンテナがあれば停止前の検査が不合格になり、何も停止しません。2026-09-17には、旧い対が稼働したままこの検査つきで基準の対を切り替え（停止前の検査は両rankで合格）、3台目のホストでは実行中の部品試験コンテナが他のGPUコンテナとして報告されました。

稼働中の対と新しい対に、同じimage要件は課しません。切替は稼働中の対を復旧先として検査し、失敗後の再起動も復旧先として予約します。復旧先は起動したときのまま戻せる必要があるので、自分のimageのMoE順markerを保ちます。新規の起動にはmarker 2が必要です（[起動検査](operations.ja.md#フルモデルの起動検査)）。復旧先の印は凍結したlaunchの中ではなく横に載せるので、復旧した対のfingerprintは以前と同じです。復旧先に何を許したかは、保存される `recovery_assets` の `warnings` に出ます。

読み取り専用のSSH確認は通信失敗時に最大3回まで再試行します。rank側が再送を無害にできる二つの変更操作も同じです。`stop` は冪等で、`start` は既に走り出した試行に対しては二つ目の監視プロセスを立てずに `replayed` を返します（終了済み・取消済みの試行は従来どおり拒否します）。切替が新しいrankを起動する時点では古いrankを止め終えているので、その段で接続が一度切れるだけで、これまでは復旧に回っていました。起動枠の予約、profile本文の書き込み、warmupの段は自動再送しません。準備確認の通信が戻らない場合は `readiness-unconfirmed` と記録し、今回の監視プロセスによるメモリ・期限ガードを維持します。同じ `--output`・`--hosts`・`--checkout`・必要なら `--ssh-config` で `cluster resume` を実行すると、再起動せず所有識別・資材・準備状態を照合します。resumeがreadyと確かめた対は、完了した切替と同じものを受け取ります。warmupの段と、`--config` を渡した場合は両rankへのprofile本文の書き込みです。切替と同じく、どちらかが失敗しても記録に残すだけで、対は動かし続けます。復旧した旧い対には、どちらも行いません。rankの終了や準備期限の超過を確認した場合はcleanup／復旧へ進みます。失敗理由はコマンド本文や秘密値を含まない構造化した情報として残します。

旧profileの復旧中も同様に扱い、`recovery-readiness-unconfirmed` では復旧中の両rankを保持して `cluster resume` で再確認します。復旧確認が成功しても新候補の失敗は残し、`recovered=true` を別に記録します。復旧・cleanupの失敗にも構造化した理由を残します。

所有する2台での実機検査では、projector hashを故意に不一致にしても稼働中の両rankが維持されました（`pre-stop-failure-v69`）。続いて新profileの起動前空き条件を999 GiBにした試験では、静的検査の後に旧rankを停止し、新しい起動はメモリ条件で失敗、その後に旧profileの両rankがAPI準備完了まで復帰しました（`rollback-fault-v72`、`recovered=true`）。先行する`v70`の復旧確認失敗も残し、復旧側の通信確認を修正する根拠にしています。制御した起動・復旧試験であり、長時間の可用性保証ではありません。単一レールの実検査とallocatorの3状態伝達も両hostで通過しました。複数レール実通信と外部KV connectorは未検収です。

### 切替の後のdecode検査

新しい起動は仮定せずに確かめます。4層fixtureでは起動が二つの数値の状態に分かれ、対では2026-09-22に、kernelの表が不変のまま6起動中1起動が別の状態で計算しました（[1.9.0での測定](benchmarks.ja.md#新imageの6起動5回は同じcompletion1回は違うcompletion)）。切替のたびに、rank 0で `tools/decode_check.py` を課題ごとに `TOKENS_OUT` 付きで走らせ、次の切替が消す前に両rankのcontainer logを保存します：

```sh
for kind in prose count code; do
  PROMPT_KIND=$kind SAMPLES=3 TOKENS_OUT=records/<run>/tokens-$kind.json \
    python3 tools/decode_check.py > records/<run>/decode-$kind.jsonl
done
docker logs <rank0のcontainer> 2>&1 | gzip > records/<run>/logs-rank0.txt.gz   # rank 1も同様に、そのhostで
```

decode検査は他の要求が走っていない時に取ります。同時2系列のprofileでは、他の要求とstepを共有する要求は別のcompletionになり、回ごとにも変わります（[1.10.2での測定](benchmarks.ja.md#1102での測定)）。起動の中では3標本が一致すること（`distinct_completions` が1）。起動を跨いでは `completion_sha256` を同じprofileの前の起動と比べます。違ったら記録を残す：`tools/decode_divergence.py` が二つの `tokens-*.json` の最初に分岐したtokenを出し（本文の後ろでの一回の同点割れか、早くからの系統的なずれか）、二つのlogが起動ごとの唯一の証拠です。速さと採択長がそのprofileのいつもの幅の中なら、違いは同点であって故障ではありません。

profileが `validation.memory_probe = true` を持つなら、切替の後、decode検査の前に重みのdigestも取ります：`python3 tools/weight_digest.py --output records/<run>/weights.json --reference records/<前の起動>/weights.json` が両rankのloadされた全parameterとbufferをfingerprintし、前の起動と違うtensorを名指しします（終了状態1）。別の数値状態の起動でdigestが同一なら同じbitから違う計算をした、違うならloadが違い、記録がどこかを言います。続けて `python3 tools/kernel_hashes.py --output records/<run>/kernels.json --reference records/<前の起動>/kernels.json` が、indexerの計算を固定入力で両配信workerの中で走らせ、両rankを互いと前の起動と比べます（差があれば終了状態1、keyを表示）。別の状態の起動が違って計算する箇所がtraceなしで名指しされます。

## APCの履歴検証

固定runtimeはキー省略時に **0** を使い、意味上必要なcheckpoint／replay境界／共有prefixの分岐点を保持します。dense保持とは異なります。任意の `cache.prefix_cache_retention_interval` で、標準機能を明示できます。実scheduler blockと同じ正の間隔なら、その境界ごとにKDA checkpointを保持します。Full attentionのdense保持は変わらず、`KpoolTailManager` はAPCへ登録しない要求専用の1block循環領域を維持します。全group一括削減ではなく、checkpointを残す設定です。正の値が実scheduler blockに整列しなければ、固定runtimeが拒否します。配布用TOMLは、履歴・保持圧力・A/B/Aと直列併用の実測を踏まえ `dense` を明示します。[配布既定](server-configuration.ja.md#配布用の既定設定)とruntimeの省略時挙動を区別してください。

`"dense"` は固定CLIの `None` に対応し、MTP切替でblock幅が変わっても値を書き直さず全checkpointを保持します。数値の間隔は比較実験用に残します。今回の整列されたKDA配置では、denseとKDA block幅と同じ間隔は同じ標準のdense maskになります。最終併用は別途検査します。

追加試験は追記、10／50／90%位置の編集・分岐、別会話への交互再訪、eviction圧力、実block／pool／MTP境界、LPA後の通常要求による再訪です。実token共通prefix・共同復元H・再計算・時間・メモリ・MTP／LPA作動を保存し、通常APCからP22併用の順に確認します。保持変更は独立したA/B/Aで判断します。CPU状態契約と従来のP22結果だけで、この追加試験を検収済みとはしません。

`apc-history` は排他的な直列APC／LPAサーバーで、通常primingとexact／auto／restored要求を使って全モデルの機能試験を行います。既定の1巡は機能検査であり、性能採用には使いません。稼働中サーバーの実block幅を `--block-tokens` に指定し、固定コーパスhashと新しい出力先を渡します。読むのはvalidation分割のみです。SSEの最初の出力とchunk間隔を分け、MTP時のchunk間隔を個別tokenのITLとしません。全モデルの前段では `apc-lpa-fixture --history` でGPU共有状態を確認します。[小層の結果](component-validation.ja.md#履歴fixtureの追加)を参照してください。

保持A/B/Aには `apc-history --timing-only --case-ids edit-50 --repeats 5` を使えます。通常計算・1出力tokenで、warmup1回を除き5回測り、毎回cache resetと通常primingを揃えます。これは時間比較の部分集合であり、機能試験全体の完了とはしません。3条件で入力token列・固定モデルruntime・KV予算を揃え、保持値と共同block整列をworkerの実値と照合します。小層の候補は `apc-lpa-fixture --retention-interval N --history` で先に検査します。

Mia PR #130／#136／#172／#175は要件の参考です。Mia実装のコピー・機械的書き直しは行わず、新しいAGPL依存は追加していません。このリポジトリと固定Apache-2.0 vLLMを使います。EXL3／DFlash、adaptive-k、4台推論、配線／IP変更、connector実装は対象外です。
