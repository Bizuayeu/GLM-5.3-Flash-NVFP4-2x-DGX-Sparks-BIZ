# 2台でのNCCL通信検証

[English／実行コマンド](nccl-validation.md) · [QSFP準備](qsfp-network.ja.md) · [検証範囲](validation.md)

[同梱の診断ツール](../tools/nccl_probe.py)は、2 rankで異なる値を持つテンソルを通信し、FP32・BF16のAllReduceを1 KiB／1 MiB／16 MiB／256 MiBで検査します。加えてFP32のAllGather・ReduceScatter・Broadcastも確認します。モデル重みはロードせず、フルモデルの合格証跡は生成しません。

## 実行の順序

1. 両機で固定IPv4・HCA・RoCEv2 GID・peer宛経路を確認する。Dockerだけでなくホストの転送プロセスも確認し、帯域測定に他の負荷が重ならない時間を選ぶ。他作業を勝手に止めない。
2. 同じソースと固定ベースイメージを使い、新しいコンテナ名・出力先を用意する。
3. [英語版の実行例](nccl-validation.md#run-on-both-hosts)のrank・interface・HCA・GIDを各機の実測値へ置き換える。HEAD_IPは両方ともrank 0のアドレス。rank 0の29653番ポートが未使用か確認する。
4. rank 1、続けてrank 0を起動する。待ち合わせは90秒、試験全体には外部から5分の期限を設ける。超えたら今回の2コンテナだけを停止し、ログを保存する。
5. 両rankのJSON、transportログ、イメージID、終了状態、同時負荷をまとめて判定する。

Docker引数の`NCCL_IB_HCA==...`と`NCCL_SOCKET_IFNAME==...`は誤記ではありません。環境変数の値を`=名前`として渡し、前方一致ではなく対象deviceへ完全一致させます。[NCCL公式仕様](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

## 合格条件と数値の読み方

- 両プロセスが終了コード0、各JSONが11項目すべて合格、OOMなしであること。
- 両方のログで`Using network IB`、指定したHCA/RoCEとbootstrap interfaceを確認すること。TCPで待ち合わせできただけではcollective経路の証明にならない。
- NCCLの実ランタイム版はcommunicator初期化ログでも確認する。`torch.cuda.nccl.version()`と一致しない場合があるため、診断では後者を`torch_reported_nccl`として保存し、実際にマップされたライブラリも記録する。
- 帯域はwarmup後10回の平均。Python呼び出しと同期を含む。`payload_GB_per_s`は1 rank分のpayload bytesを秒で割った十進GB/sであり、合計回線帯域、公式nccl-testsの結果、モデル生成速度ではない。
- MTUと同時稼働ジョブを必ず記録する。本番向けの帯域合格閾値は、この診断だけでは設定しない。

NVIDIAの[Spark移植ガイド](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)では、統合メモリの制約から従来のGPUDirect RDMAとnvidia-peermem／DMA-BUF／GDRCopyは非対応とされています。`NET/IB`と`GDR 0`が併記されても、それだけでRoCE失敗とは判断しません。表示を変えるためだけにkernel moduleをロードしたり、GDRを強制したりしません。

初回のGB10 2台・MTU 1500では、NCCL実ランタイム2.30.7で全項目が合格しました。大きなAllReduceは約1.2 GB/sでしたが、別の転送が資源を共有していた可能性があります。`NCCL_NET_GDR_LEVEL=SYS`だけを追加した比較も合格したもののGDR有効化・帯域改善はなく、標準の実行例には採用していません。現在の適用範囲は[検証文書](validation.md)を参照してください。

最終試験はQSFP転送の終了後に同梱probeで再実施し、両rankとも11項目合格、256 MiB AllReduceは1.18〜1.21 GB/sでした。別モデルのディスクchecksumは稼働中だったため、ホスト全体が完全無負荷の測定とは呼びません。全試験コンテナは終了コード0・OOMなし。フルモデルTP=2は引き続き未検証です。
