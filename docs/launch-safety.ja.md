# 起動契約と運用検証

[English](launch-safety.md)

P10（メモリ）、P19／P22（APC）、E03（運用）を拡張する項目です。高速化施策として重複計上しません。実装とCPU契約の確認に加え、実コンテナ・全モデルの回帰結果を採用前に別途記録します。通常の `service` の検収ゲートは維持します。

## モデルAPIクライアント

共通送信処理は非空の `API_KEY` を優先し、それが空／未設定なら非空の `VLLM_API_KEY` を使います。両方とも空／未設定ならAuthorizationを送りません。`startup ask`、それを使うベンチ、profiler制御、componentの準備確認が対象です。宛先originはモデルAPIとして明示し、同一originを含めredirectは拒否します。ダウンロード経路には適用しません。キーを設定やfingerprintへ入れず、HTTP例外にはヘッダーや応答本文を保存しません。401／403は失敗として記録し、成功した測定から欠測として除きません。

固定vLLMの認証middlewareが保護するのは `/v1`・`/v2`・`/inference`・`/cohere` です。`/health`・`/metrics`・`/tokenize`・`/collective_rpc`・`/reset_prefix_cache`・profiler制御は保護しません。Bearer送信だけでサーバー側の保護範囲は変わりません。APIはloopback限定を維持します。今回追加するのはクライアント認証対応であり、公開サーバー用の認証層ではありません。

## allocatorと共通起動設定

任意の `runtime.cuda_allocator_conf` を `PYTORCH_CUDA_ALLOC_CONF` へ渡します。省略はimage／runtime既定を維持し、文字列は明示的な空文字を含めそのまま渡します。確認した基準imageにallocator環境設定はありません。hidden-state KV connectorの互換性を検収したことにはなりません。

環境変数の上書きは起動元で一度だけ解決します。

```sh
python -m glm53_setup startup freeze --config state/startup.toml --output state/launch.json
python -m glm53_setup startup plan --config state/startup.toml --launch state/launch.json --rank 0
```

環境変数は空文字でも「存在」すればTOMLより優先します。同じ凍結JSONを両rankへ配布し、start／preflightの `--launch` へ渡します。rank側の環境変数は再解決しません。ローカルallocator環境変数がある直接起動では、凍結済みmanifestを必須とします。manifestは解決済みprofileとlockに結び付くfingerprintを含み、APIキーは含みません。`freeze` は既存manifestを上書きしません。

## 全レール検査と両rankの切替

各nodeの主レールは従来の `hca`・`interface`・`local_ip`・`gid_index`（port 1）です。任意の `additional_rails` に同じ項目と `port` を持つレコードを追加します。全レールで共通GID index、port／NIC／IPの重複排除、Ethernet portとlinkの稼働、IPv4対応RoCE v2 GID、当該NICへのIP割当を確認します。カンマ区切りのdevice文字列は受けず、構造化した設定を使います。NCCLには全HCA／portを完全一致指定し、socket bootstrapは主NICを使います。設定検査の成功と複数レール実通信の検収は別です。

単一レールでも `=hca:1` とport 1を明示します。portを省略すると、そのHCAの全portが対象となり、検査した範囲を超えるためです。[NVIDIAのNCCL HCA指定仕様](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)を参照してください。

```sh
python -m glm53_setup cluster switch --config state/startup.toml \
  --hosts spark-head spark-peer --checkout /srv/glm53/source \
  --remote-config /srv/glm53/state/startup.toml \
  --output records/switch-run --experimental
```

両hostには同じ監査済みcheckout・image・資材を用意します。`--remote-config` はprojector相対パスのLinux側基準、共通の凍結manifestは設定値を指定します。必要に応じて `--ssh-config` を指定します。停止前に両rankの資材・fabricと共通source／image／model／profile／allocatorを検査し、停止直前にも再照合します。重みの照合はindex hash・shardサイズ・ローカルfile識別であり、元の重み完全性検査の代わりではありません。ロード用の空きメモリ検査は停止後に行います。

停止前の不合格では稼働中コンテナを維持します。停止後の不合格では今回予約した起動分だけを停止し、記録済みの旧profileで復旧を試みます。停止確認が取れない場合は競合する復旧起動を避けます。結果に失敗・cleanup・復旧を分けて残します。停止を伴う切替であり、原子的な無停止切替ではありません。稼働rankに設定パスの記録がない場合は復旧条件が揃わないため停止前に拒否します。他用途のコンテナは停止しません。

読み取り専用のSSH確認は通信失敗時に最大3回まで再試行します。起動・停止・起動枠予約は自動再送しません。準備確認の通信が戻らない場合は `readiness-unconfirmed` と記録し、今回の監視プロセスによるメモリ・期限ガードを維持します。同じ `--output`・`--hosts`・`--checkout`・必要なら `--ssh-config` で `cluster resume` を実行すると、再起動せず所有識別・資材・準備状態を照合します。rankの終了や準備期限の超過を確認した場合はcleanup／復旧へ進みます。失敗理由はコマンド本文や秘密値を含まない構造化した情報として残します。

## APCの履歴検証

実測するまでは固定runtimeの実際の保持既定を維持します。このrevisionの `prefix_cache_retention_interval` は **0** が既定で、意味上必要なcheckpoint／replay境界／共有prefixの分岐点を保持します。dense保持とは異なります。GLMのKDA・MLA・indexer・MTP全groupへ新しい間引き設定やSWA専用則を導入せず、全group一括削減が安全だと仮定しません。

追加試験は追記、10／50／90%位置の編集・分岐、別会話への交互再訪、eviction圧力、実block／pool／MTP境界、LPA後の通常要求による再訪です。実token共通prefix・共同復元H・再計算・時間・メモリ・MTP／LPA作動を保存し、通常APCからP22併用の順に確認します。保持変更は独立したA/B/Aで判断します。CPU状態契約と従来のP22結果だけで、この追加試験を検収済みとはしません。

`apc-history` は排他的な直列APC／LPAサーバーで、通常primingとexact／auto／restored要求を使って全モデルの機能試験を行います。既定の1巡は機能検査であり、性能採用には使いません。稼働中サーバーの実block幅を `--block-tokens` に指定し、固定コーパスhashと新しい出力先を渡します。読むのはvalidation分割のみです。SSEの最初の出力とchunk間隔を分け、MTP時のchunk間隔を個別tokenのITLとしません。全モデルの前段では `apc-lpa-fixture --history` でGPU共有状態を確認します。[小層の結果](component-validation.md#additional-history-fixtures)を参照してください。

Mia PR #130／#136／#172／#175は要件の参考です。Mia実装のコピー・機械的書き直しは行わず、新しいAGPL依存は追加していません。このリポジトリと固定Apache-2.0 vLLMを使います。EXL3／DFlash、adaptive-k、4台推論、配線／IP変更、connector実装は対象外です。
