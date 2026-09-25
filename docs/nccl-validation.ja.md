# 2台でのNCCL通信検証

[English](nccl-validation.md) · [QSFP準備](qsfp-network.ja.md) · [検証範囲](validation.ja.md)

[同梱の診断ツール](../tools/nccl_probe.py)は、2 rankで異なる値を持つテンソルを通信し、FP32・BF16のAllReduceを1 KiB／1 MiB／16 MiB／256 MiBで検査します。加えてFP32のAllGather・ReduceScatter・Broadcastも確認します。モデル重みはロードせず、フルモデルの合格証跡は生成しません。

## 実行の順序

1. 両機で固定IPv4・HCA・RoCEv2 GID・peer宛経路を確認する。Dockerだけでなくホストの転送プロセスも確認し、帯域測定に他の負荷が重ならない時間を選ぶ。他作業を勝手に止めない。
2. 同じソースと固定ベースイメージを使い、新しいコンテナ名・出力先を用意する。
3. 各機のLinux checkoutで、次の例示値をrank・interface・HCA・GIDの実測値へ置き換える（rank 1は自機のinterface／HCA／IPを使う）。`HEAD_IP`は両方ともrank 0のアドレス。rank 0の29653番ポートが未使用か確認する。

   ```sh
   RANK=0
   FABRIC_IF='REPLACE_WITH_OBSERVED_INTERFACE'
   HCA='REPLACE_WITH_OBSERVED_HCA'
   GID='REPLACE_WITH_OBSERVED_INDEX'
   HEAD_IP='10.53.0.1'
   RUN_ID='REPLACE_WITH_UNIQUE_RUN_ID'
   IMAGE=$(python -c 'from glm53_setup.config import load_lock; print(load_lock()["image"])')
   mkdir -p "records/$RUN_ID"
   ```

4. rank 1、続けてrank 0を速やかに起動する。待ち合わせは90秒、試験全体には外部から5分の期限を設ける。超えたら今回の2コンテナだけを両機で停止し、ログを保存する。

   ```sh
   docker run --name "glm53-nccl-$RUN_ID-rank$RANK" \
     --gpus all --network host --memory 16g --memory-swap 16g --shm-size 1g \
     --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
     --device /dev/infiniband:/dev/infiniband \
     -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
     -e "NCCL_IB_HCA==$HCA" -e "NCCL_IB_GID_INDEX=$GID" \
     -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
     -e NCCL_SOCKET_FAMILY=AF_INET -e "NCCL_SOCKET_IFNAME==$FABRIC_IF" \
     -e "GLOO_SOCKET_IFNAME=$FABRIC_IF" \
     -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH \
     -v "$PWD/tools/nccl_probe.py:/probe.py:ro" \
     -v "$PWD/records/$RUN_ID:/out" \
     --entrypoint python3 "$IMAGE" /probe.py \
     --rank "$RANK" --head "$HEAD_IP" --port 29653 \
     --output "/out/rank$RANK.json" >"records/$RUN_ID/nccl.log" 2>&1
   ```

5. 両rankのJSON、transportログ、イメージID、終了状態、同時負荷をまとめて判定する。

Docker引数の`NCCL_IB_HCA==...`と`NCCL_SOCKET_IFNAME==...`は誤記ではありません。環境変数の値を`=名前`として渡し、前方一致ではなく対象deviceへ完全一致させます。[NCCL公式仕様](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)を参照してください。分散用ポートは信頼できるfabric内に閉じます。

## 合格条件と数値の読み方

- 両プロセスが終了コード0、各JSONが11項目すべて合格、OOMなしであること。
- 両方のログで`Using network IB`、指定したHCA/RoCEとbootstrap interfaceを確認すること。TCPで待ち合わせできただけではcollective経路の証明にならない。
- NCCLの実ランタイム版はcommunicator初期化ログでも確認する。`torch.cuda.nccl.version()`と一致しない場合があるため、診断では後者を`torch_reported_nccl`として保存し、実際にマップされたライブラリも記録する。
- 帯域はwarmup後10回の平均。Python呼び出しと同期を含む。`payload_GB_per_s`は1 rank分のpayload bytesを秒で割った十進GB/sであり、合計回線帯域、公式nccl-testsの結果、モデル生成速度ではない。
- MTUと同時稼働ジョブを必ず記録する。本番向けの帯域合格閾値は、この診断だけでは設定しない。

NVIDIAの[Spark移植ガイド](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)では、統合メモリの制約から従来のGPUDirect RDMAとnvidia-peermem／DMA-BUF／GDRCopyは非対応とされています。`NET/IB`と`GDR 0`が併記されても、それだけでRoCE失敗とは判断しません。表示を変えるためだけにkernel moduleをロードしたり、GDRを強制したりしません。

参照の結果（GB10 2台、MTU 1500、NCCL実ランタイム2.30.7、fabricの転送終了後）：両rankとも11項目合格、全コンテナが終了コード0・OOMなし。256 MiB AllReduceは1.18〜1.21 GB/sで、別モデルのディスクchecksumが稼働中だったため無負荷ホストの測定ではありません。`NCCL_NET_GDR_LEVEL=SYS`を加えた比較も合格しましたが、GDRは有効にならず帯域も増えず、上の実行例には含めていません。

## チャネル数

NCCL 2.30.7に任せると、このpairでは64チャネルになります（2026-09-11と2026-09-17で同じ）。テンプレートは[`runtime.nccl_channels = 8`](server-configuration.ja.md)を設定します。その根拠となる測定は2026-09-17に、参照imageとランチャーと同じfabric環境変数で、`NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` だけを変えて行いました。

**集団通信単体。** 2 rankのBF16 AllReduceを、順序を逆にして2周測り、サイズごとに7バッチの中央値を取りました。メモリはcommunicatorの初期化から全サイズを流し終えるまでの `MemAvailable` の減少です。モデルが使う大きさは、hidden size 4096・BF16（1 tokenあたり8,192バイト）で決まります。MTP k=3のdecode検証は32 KiB、下書き1回は8 KiB、512 tokenのprefillチャンクは4 MiBです。

| チャネル | rankあたりのメモリ | 32 KiB | 4 MiB | 256 MiB |
|---:|---:|---:|---:|---:|
| 64（NCCLの選択） | 2.3 GiB | 0.022 ms | 0.41〜0.45 ms | 20.0 ms（MTU 9000）、26.1 ms（MTU 1500） |
| 32 | 1.5 GiB | 0.022 ms | 0.35〜0.39 ms | 19.7〜23.5 ms |
| 16 | 1.0 GiB | 0.022 ms | 0.34〜0.35 ms | 19.7〜20.6 ms |
| 8 | 0.85 GiB | 0.022 ms | 0.33〜0.36 ms | 19.4〜19.5 ms |
| 4 | 0.75 GiB | 0.022 ms | 0.35〜0.37 ms | 19.4 ms |

decodeの大きさのメッセージは変わりません。小さいメッセージでは、NCCLが元から少ないチャネルしか使わないためです。メモリはMTUに依存しません。これらの時間は、上のprobe（FP32、別のジョブがホストを使っていた状態）の値とは比べられません。

**実モデル。** 同じprofileを、チャネル数かMTUだけを変えて起動しました。すべて同じ再起動の後です。headの監視ダッシュボードは、MTU 9000の計測では止め、MTU 1500の計測では動かしたままでした（headでRSS 65 MiB・CPU約2%、peerではメトリクス取得のコマンド）。このためMTU同士の比較にはその負荷の差も含まれます。同じMTUでのチャネル数同士の比較には含まれません。prefillは新規の38,961 token promptを3回流した中央値です。最小空きメモリは、各rankの監視プロセスの標本から、起動・warmup・計測の全期間で読みました。

| MTU | チャネル | prefill tok/s | headの最小空き | peerの最小空き |
|---:|---:|---:|---:|---:|
| 1500 | 64 | 487.3 | 4.18 GiB | 6.48 GiB |
| 1500 | **8** | **492.0** | **6.93 GiB** | **9.43 GiB** |
| 9000 | 64 | 499.8 | 2.94 GiB | 4.76 GiB |
| 9000 | 8 | 503.2 | 5.81 GiB | 8.08 GiB |
| 9000 | 16 | 497.3 | 5.39 GiB | 7.57 GiB |

各エンジンはcommunicatorを2本開くので、8チャネルにするとrankごとに約3 GiB戻ります。prefillの差はどちらも1%以内です。decodeは、同じ設定の計測ごとのぶれ（約±15%）の方が設定の差より大きい値でした。

**MTU。** 9000（RoCEのactive MTUは4096）にすると、prefillは最大で2.3〜2.6%上がり（MTU 1500側にだけダッシュボードの負荷があったため上限値）、空きメモリはホストあたり1.1〜1.7 GiB減りました。モデルを動かしていないホストでも同じく1.4 GiBの差があり、NICの受信バッファが大きくなる分（4インタフェース×20キュー×1,024本）と合います。参照pairはMTU 1500で運用します。再起動前に取ったMTU 1500のprefill（446 tok/s）は、MTUではなくホストの状態で低く出ていたので、表から外しています。
