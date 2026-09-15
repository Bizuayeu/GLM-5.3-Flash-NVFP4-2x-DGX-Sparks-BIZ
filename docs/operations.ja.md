# 運用手順 — ベータ版

[English](operations.md)

**通常運用としてのTP=2デプロイは、まだ検収されていません。** 直列のフルモデル参照profileには[実験結果](validation.ja.md#フルモデルtp2の実験範囲)と[初期ベンチ](benchmarks.ja.md)がありますが、ガード付きの通常ランチャーは候補実装のままです。

## 二つのランチャー

checkoutには起動経路が二つあり、一方の証拠は他方の検収になりません。

| コマンド | 役割 | 状態 |
|---|---|---|
| `python -m glm53_setup startup …` | [起動設定TOML](startup-configuration.ja.md)で動く実験用referenceランチャーと直列クライアント。両rankの[切替・復旧手順](launch-safety.ja.md#全レール検査と両rankの切替)はこれを包む | 本リポジトリの全モデル実測はすべてこの経路で行った。実験用であり、通常運用の検収ではない |
| `python -m glm53_setup service …` | `state/site.json` と[フルモデルの起動ゲート](#フルモデルの起動ゲート)を持つ、ガード付きの通常運用候補 | `plan`・`preflight` は確認用。`start` はロックのbase imageを選択したままで、そのnative GB10 NoPE経路には動作阻害があり、まだ生成手順のない検収証跡を要求する。reference imageをビルドしても切り替わらない |

## 資材の保管場所とパス

本節がデプロイ時の保管パスの正典です。モデルID・revision・base imageのdigestは[runtime.lock.json](../config/runtime.lock.json)で固定し、本リポジトリは重みを配布しません。運用者固有のホスト名、home配下の絶対パス、認証情報は公開ソースの外に置いてください。

| 資材 | 各Linuxホストでの既定の場所 | 役割 |
|---|---|---|
| 本体checkpoint | `$HOME/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/<revision>/` | 固定したモデル・config・tokenizerのview。重みファイルは同階層の `blobs/` ディレクトリへリンクし、データ本体はそちらが持つ |
| MTPメタデータview（配布既定で必要） | `$HOME/.cache/huggingface/local-views/glm53-mtp-compatible/<revision>/` | 既存のtensorデータをリンクし、checkpoint同梱のBF16 MTPに合わせて量子化メタデータを調整する。元のsnapshotを編集せずに[viewを作成](speculative-decoding.ja.md#各linuxホストでの準備)する |
| LPA projector（`lpa.enabled = true` のとき必要。テンプレートは無効） | [起動設定TOML](startup-configuration.ja.md)の `[lpa].projector` で選ぶ非公開ファイル。そのTOMLからの相対パスまたは絶対パス | 別途学習した補助重みで、NVIDIAのsnapshotにもソース配布物にも含まれない。[LPAの手順](lpa.ja.md)で対応するprojectorを入手または学習し、両ホストでhashを確認する。通常の推論とbatchingには不要 |
| Dockerのbase／reference image | Dockerが管理する保管領域 | 固定したbaseをpullし、本ソースからreference imageをビルドする。ソースのcheckout、image、checkpointは別々の資材 |
| ローカル設定と取得状態 | `<checkout>/state/` | サイト固有の起動設定と `download-status.json`。後者は実際に取得した `snapshot` のパスを記録する |
| runtime／JIT cacheと証跡 | `<checkout>/state/tp2-runtime-cache/`、`<checkout>/records/` | 再生成できるruntimeデータと非公開の実行記録。モデル重みでも配布物の入力でもない。分散起動はTriton・TileLang・TorchInductorのcacheをruntime cacheへ向け、コンパイル済みkernelを再起動後も残す |

実験用の起動ランチャーは、ホスト既定のHugging Face cacheを読み、containerの `/hf` へ読み取り専用でmountします。選択したsnapshotまたはMTP viewは、そのmount内で解決します。モデルcache全体の `blobs`／`snapshots` の関係を保ってください。snapshotディレクトリだけを複製しても足りません。両ホストのディスクに完全なcheckpointが必要です。TP=2が分割するのはロード済みのtensorであり、ダウンロードしたファイルではありません。

ダウンローダーはHugging Faceのcache環境設定に従いますが、現行のランチャーは既定のcache rootを前提とします。このベータ版では、これらの資材を取得する際に `HF_HOME`／`HF_HUB_CACHE` を設定せず、文書化した既定の場所を使ってください。任意のcacheへのダウンロードが成功しても、ランチャーがそれを見つけてmountできることの証明にはなりません。

ダウンロードを始めずに、想定される場所と記録された場所を確認します。各Linuxホストのcheckoutで実行してください。

```sh
python -c 'from pathlib import Path; from glm53_setup.config import MODEL, REVISION; print(Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION)'
python -c 'import json; from glm53_setup.config import STATE; s = json.loads((STATE / "download-status.json").read_text()); print(s.get("status"), s.get("snapshot", "not recorded"))'
```

2つ目のコマンドは、このcheckoutで取得が登録済みであることを前提とします。表示されたパスも `status=complete` も、checksum検証の代わりにはなりません。起動設定は、両機で実際にビルドして確認したimageを指す必要があります。

## 一度取得して検証する

`config/runtime.lock.json` の固定revisionを使います。`download` はHugging Face cacheの既存ファイルを再利用し、同じcheckout内でのダウンロード重複を防ぎます。`verify-download` は公式のchecksum検証を実行し、メタデータのためにHugging Faceへ接続する場合があります。推論がオフラインであることと、checksum検証がオフラインでできることは別です。

2台目へは、モデルの `blobs` と `snapshots` のツリーを完全な形でまとめて移送します。snapshotは `blobs` へのリンクを含むため、snapshotだけを複製・mountするとリンクが切れることがあります。既存のcacheファイルは保持し、削除同期のオプションは使いません。

移送後、そのcheckoutで固定snapshotを登録・確認するために `download` を一度実行します。一致するcacheファイルは再利用し、不足分は取得される場合があります。続いて `--wait` なしで `verify-download` を実行します。ファイルサイズだけで移送成功と判断しません。

## 各ホストの準備

1. 空きメモリ、ディスク、GPU・ドライバー、稼働中のモデルプロセス、ホストの状態を確認する。GLMを起動する前に、別のモデルはそれぞれの文書化された手順に従って停止する。
2. 各ホストで `prepare-image` を実行する。固定したARM64 baseをpullし、実際のパッケージ版数、GPU計算、GLMの登録状況を記録する。baseのnative NoPE経路は、検収済みの提供経路ではない。
3. `build-reference` でreference imageを一度だけビルドする。base digestはロックから取る。複製する場合は、検証済みのローカル回線越しにDockerのimage save/loadを使い、実際のimage IDを比較する。
4. [GPU 1台の検証](validation.ja.md)を実施する。image、精度、sourceのhash、生成した記録をまとめて保存する。

## ホストカーネルと複数ノードRoCE

**更新を入れる前と、2台で動かす手順の前に、カーネルを確認してください。** 本リポジトリの実測は、MSI EdgeXpert（MS-C931）上の `6.17.0-1032-nvidia`、ドライバー 580.173.02、ConnectX-7 ファームウェア 28.45.4028 で行いました。カーネル `7.0.0-1019-nvidia` は、ここでは未検証です。

2026-09-15 時点の更新では、`linux-nvidia-hwe-24.04` 系のメタパッケージが `7.0.0-1019-nvidia` へ上がり、580 open ドライバーのモジュールもそのカーネル向けに入ります。`apt` の更新でも DGX Dashboard の更新でも入るため、新しく導入した機体も最初の更新の後はこのカーネルで起動します。

このカーネルの既定設定では、2台間の RoCE 越しの NCCL が `NCCL WARN Call to ibv_reg_mr_iova2 failed with error Cannot allocate memory` で失敗することがあります。報告では、モデルのロードは済み、vLLM のプロファイル中や TP 通信で失敗し、`ib_write_bw` などの RDMA 単体試験は正常に見えます。NVIDIA の[更新に関する告知](https://forums.developer.nvidia.com/t/dgx-spark-update-advisory/383254)（2026-09-13）は、複数ノード・RoCE 構成の利用者に対し、DGX Dashboard 経由を含めてこのカーネルへの更新を見送るよう求めており、修正版は示していません。単体ノードの処理に影響するかは確認されていません。

[NV-Kernels PR #590](https://github.com/NVIDIA/NV-Kernels/pull/590)（未マージ。投稿者による分析で、NVIDIA の見解ではない）は、原因を Kexec HandOver（KHO）と特定しています。`7.0.0-1019-nvidia` のビルドは `CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y` です（パッケージの config で確認。`6.17.0-1032-nvidia` は KHO を既定では有効にしない）。KHO は起動時に、後の kexec 用の scratch メモリを確保し、CMA のページブロックとして解放します。その報告では約 9.3 GiB（4,761 ページブロック）で、`CmaTotal` には計上されません。RDMA のメモリ登録はページを長期間固定し、固定するページは先に CMA の外へ移す必要があります。GPU がメモリを使い込んでいるとこの移動が失敗し、登録が `ENOMEM` を返します。

2台とも同じ対応にしてください。

| 選択 | 手順 | 補足 |
|---|---|---|
| `6.17.0-1032-nvidia` を使い続ける | 更新の前に `sudo apt-mark hold linux-nvidia-hwe-24.04 linux-image-nvidia-hwe-24.04 linux-headers-nvidia-hwe-24.04 linux-modules-nvidia-580-open-nvidia-hwe-24.04 linux-tools-nvidia-hwe-24.04`。既に 7.0 を入れた場合も旧カーネルは残るので、GRUB メニューの詳細オプションから起動する（コンソール接続が必要）。 | 本リポジトリで検証済みの状態。修正版カーネルが出たら hold を外す。 |
| `7.0.0-1019-nvidia` を KHO 無効で使う | `/etc/default/grub` の `GRUB_CMDLINE_LINUX_DEFAULT` に、既存の値を残したまま `kho=off` を足す。`sudo update-grub` の後に再起動する。`/proc/cmdline` に `kho=off` があり、`sudo ls /sys/kernel/debug/kho` が "No such file or directory" で失敗することを確認する。 | 告知スレッドに投稿された回避策。PR では KHO 無効で、2台間のメモリ登録・NCCL・TP2 の試験が通ったと報告されている。本リポジトリでは未検証。KHO は kexec による稼働中更新のための機能で、この構成では使わない。 |

どちらを選んでも、提供を始める前に [NCCL 検証](nccl-validation.ja.md)とフルモデルの起動をやり直してください。

同じエラーの別の報告もあります。Ubuntu 汎用の 7.0 カーネル・ドライバー 595.84 の MS-C931 機で、空きが約 118 GiB あり重みのロード前だったにもかかわらず失敗し、MSI のボードファームウェア更新（組み込みコントローラー、SoC ファームウェア、USB-C PD）で解決したとしています（[MiaAI-Lab issue #259](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/259)）。メモリに余裕があるのにこのエラーが出る場合は、メーカーのファームウェアも確認してください。

## ネットワークとサイト設定

明示的な実験用reference経路には[起動設定TOML](startup-configuration.ja.md)を使います。以下の `service` の説明は、別系統の候補ランチャーとその検収ゲートについてのものです。

物理接続と永続的なIPv4設定は、[QSFPのハンズオン手順](qsfp-network.ja.md)に従います。

`examples/site.example.json` を各ホストで個別に `state/site.json` へコピーします。例示値はすべて、そのホストで実測した値へ置き換えてください。

- rank 0または1、自機のfabric IPv4、headのfabric IPv4
- Ethernet interface、RDMAのHCA、そのinterfaceのRoCEv2 GID index
- 未使用のAPIポートとrendezvousポート

HCAとGIDの番号は、両ホストで一致している必要はありません。GIDが自機のIPv4とnet deviceに対応することを確認してください。MTU 9000は、両端と経路全体が対応する場合にだけ使います。フルモデルをロードする前に、実際のNCCL transportとcollectiveの正当性を検証します。SSHで接続できることはRDMAの試験ではありません。

```sh
python -m glm53_setup service plan
python -m glm53_setup service preflight
```

`plan` はcontainerを起動せずに引数を表示します。`preflight` は失敗を記録し、要件が欠けていれば非ゼロで終了します。あわせて、完了済みで内容の一致するダウンロード状態も要求します。

## フルモデルの起動ゲート

候補である `service start` の経路は、実イメージとモデルrevisionに結び付いた `tp2-kernel-validation` の合格結果を持つ `state/kernel-validation.json` の証跡を要求します。**このベータ版には、その証跡を生成する完成した手順がまだありません。** 証跡は実際の2 rank検収から得るものであり、手編集やGPU 1台のfixtureから作るものではありません。

候補ランチャーはロックのbase imageを選択したままで、そのnative NoPE経路には動作阻害があります。reference imageをビルドしても、提供用としてそれが選ばれるわけではありません。フルモデルの検収では、このruntime選択と証跡の結び付けを実装して検証する必要があります。[順序付きのセットアップゲート](../SETUP.ja.md#6-フルモデルの検証--現ベータ版の停止条件)を参照してください。

将来その検収が済んだ後は、rank 1をheadlessで先に起動し、workerがrendezvousを待つ状態になってからrank 0を起動します。APIはhead側のloopbackアドレスにbindするため、遠隔クライアントからはSSHトンネルを使います。内部のrendezvousにはfabric IPを使います。事業サービスとして公開するには、別途検討した認証・TLS・アクセス制御の層が必要です。本リポジトリは、それを提供すると主張しません。

## 復旧と記録

スクリプトは、失敗したcontainerや重みを削除せず、再起動用のwatchdogも導入しません。`service stop` が停止するのは、このデプロイの所有ラベルを持つcontainerだけです。同じrank名を作り直す前に、ログを保存し、停止したcontainerの名前を変更してください。分散実行で障害が起きた後は、両rankをまとめて再初期化します。

`state/` は現在の取得状態とサイト設定を保持し、`records/` はrunごとの証跡を保持します。休止中の取得は意図的な停止です。検証の待機は終了コード2で終わり、ダウンロードを再開しません。ローカル移送の実行中に、新しい取得を開始しないでください。

数値・backendの詳細な制約は[validation.ja.md](validation.ja.md)にあります。リリース準備では、未解決の失敗を隠さず残してください。
