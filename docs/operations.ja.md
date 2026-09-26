# 運用手順

[English](operations.md)

**通常運用は宣言した範囲で受け入れ済みです**：両profileの同時1系列（2026-09-22）と、公開した任意設定の同時2系列profileの同時2系列（[同時実行の範囲](validation.ja.md#同時実行の範囲)、2026-09-23）。その受け入れと、各項目の証拠の所在は[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)が記録です。範囲ごとの状態はそこと、READMEの状態表、[ハーネス受け入れ一覧](harnesses.ja.md#受け入れ試験一覧と実施状態)で示し、コマンド名では示しません。他の機体、それを超える同時数、動画入力はその範囲の外です。

## ランチャーは一つ

checkoutの起動経路は `python -m glm53_setup server …` の一本です。[起動設定TOML](server-configuration.ja.md)で動き、稼働中の対がある場合は両rankの[切替・復旧手順](launch-safety.ja.md#全レール検査と両rankの切替)がこれを包みます。本リポジトリの全モデル実測はすべてこの経路で行い、起動前に行う検査は下記の[起動検査](#フルモデルの起動検査)です。

## 資材の保管場所とパス

本節がデプロイ時の保管パスの正典です。モデルID・revision・base imageのdigestは[runtime.lock.json](../config/runtime.lock.json)で固定します。本体checkpointは上流から取得し、任意のLPA projectorは独立したGitHub Release添付物として配布します。運用者固有のホスト名、home配下の絶対パス、認証情報は公開ソースの外に置いてください。

| 資材 | 各Linuxホストでの既定の場所 | 役割 |
|---|---|---|
| 本体checkpoint | `$HOME/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/<revision>/` | 固定したモデル・config・tokenizerのview。重みファイルは同階層の `blobs/` ディレクトリへリンクし、データ本体はそちらが持つ |
| MTPメタデータview（配布既定で必要） | `$HOME/.cache/huggingface/local-views/glm53-mtp-compatible/<revision>/` | 既存のtensorデータをリンクし、checkpoint同梱のBF16 MTPに合わせて量子化メタデータを調整する。元のsnapshotを編集せずに[viewを作成](speculative-decoding.ja.md#各linuxホストでの準備)する |
| LPA projector（`lpa.enabled = true` のとき必要。テンプレートは無効） | `<checkout>/state/lpa/glm53-lpa-cut32-v1/projector.pt`。[起動設定TOML](server-configuration.ja.md)の`[lpa].projector`に、そのTOMLからの相対パスまたは絶対パスを指定 | NVIDIAのsnapshot・ソース配布物とは別のRelease添付物。両ホストで[取得・hash検証](lpa.ja.md#学習済みprojectorの取得)するか、対応するprojectorを学習する。[有効化](lpa.ja.md#起動profileでlpaを有効にする)はprofile編集と切替を伴う別手順。通常の推論とbatchingには不要 |
| Dockerのbase／reference image | Dockerが管理する保管領域 | 固定したbaseをpullし、本ソースからreference imageをビルドする。ソースのcheckout、image、checkpointは別々の資材 |
| ローカル設定と取得状態 | `<checkout>/state/` | サイト固有の起動設定と `download-status.json`。後者は実際に取得した `snapshot` のパスを記録する |
| runtime／JIT cacheと証跡 | `<checkout>/state/tp2-runtime-cache/`、`<checkout>/records/` | 再生成できるruntimeデータと非公開の実行記録。モデル重みでも配布物の入力でもない。分散起動はTriton・TileLang・TorchInductorのcacheとCUDA driverのJIT cache（`CUDA_CACHE_PATH=/root/.cache/nv`。指定しなければcontainer内の `~/.nv/ComputeCache`）をruntime cacheへ向け、コンパイル済みkernelを再起動後も残す。それでもコンパイルされるものは[warmup ladder](#監視停滞検知warmup)が記録する |

LPA添付物の展開後の構成は次のとおりです。`manifest.json`は[projector lock](../config/lpa-projector.lock.json)の写しです。ソースcheckoutのアーカイブに、このディレクトリは含まれません。

ソースアーカイブには`state/`と`records/`も意図的に含めません。serverランチャーを使う前に、新しいcheckoutから各ホストの永続領域へsymlinkを作成します。

```sh
ln -sT /srv/glm53/state /srv/glm53/source/state
ln -sT /srv/glm53/records /srv/glm53/source/records
readlink -f /srv/glm53/source/state /srv/glm53/source/records
```

絶対パスを使います。`-T` を付けると既存ディレクトリの中に `state/state` を作らず失敗で止まり、`readlink -f` は `/srv/glm53/state` と `/srv/glm53/records` を表示するはずです。`state/state` や `records/records` で終わるパスが出たら入れ子です。復旧用に旧checkoutを保持してください。認証情報や生の記録をソースアーカイブへ置きません。

```text
state/lpa/glm53-lpa-cut32-v1/
├── projector.pt
├── manifest.json
├── README.md
├── README.ja.md
├── LICENSE
├── NOTICE
├── TRAINING_DATA.md
├── TEACHER_MODEL_CARD.md
└── LICENSES/
    └── ZAI-GLM-MIT.txt
```

実験用の起動ランチャーは、ホスト既定のHugging Face cacheを読み、containerの `/hf` へ読み取り専用でmountします。選択したsnapshotまたはMTP viewは、そのmount内で解決します。モデルcache全体の `blobs`／`snapshots` の関係を保ってください。snapshotディレクトリだけを複製しても足りません。両ホストのディスクに完全なcheckpointが必要です。TP=2が分割するのはロード済みのtensorであり、ダウンロードしたファイルではありません。

ダウンローダーはHugging Faceのcache環境設定に従いますが、現行のランチャーは既定のcache rootを前提とします。本リリースでは、これらの資材を取得する際に `HF_HOME`／`HF_HUB_CACHE` を設定せず、文書化した既定の場所を使ってください。任意のcacheへのダウンロードが成功しても、ランチャーがそれを見つけてmountできることの証明にはなりません。

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

200 GBのハッシュ計算はページキャッシュを埋め、それはGPUと同じunified memoryを使います。Spark 2台の他レシピでは、GPUが遊んでいる状態のchecksum検証9回中2回で電源が落ちたと報告されています（tonyd2wild PR #19、コードは採用しない）。検証はモデルを読み込む前に行い、稼働中の対の横では行わないでください。稼働中のホストでの大きな読み出しも規模は小さいが同じ向きに効きます。9.7 GiBの参照imageをpeer rankから書き出したとき、MemAvailableは8.8 GiB以上を保ったまま、MemFreeは3.0 GiBから0.87 GiBまで下がりました。NVIDIAのcheckpointの後続revision `09b04e5e`（2026-09-11）と固定revisionの差は `README.md` だけです。

## 各ホストの準備

1. 空きメモリ、ディスク、GPU・ドライバー、稼働中のモデルプロセス、ホストの状態を確認する。GLMを起動する前に、別のモデルはそれぞれの文書化された手順に従って停止する。
2. 各ホストで `prepare-image` を実行する。固定したARM64 baseをpullし、実際のパッケージ版数、GPU計算、GLMの登録状況を記録する。baseのnative NoPE経路は、検収済みの提供経路ではない。
3. `build-reference` でreference imageを一度だけビルドする。base digestはロックから取る。複製する場合は、検証済みのローカル回線越しにDockerのimage save/loadを使い、実際のimage IDを比較する。
4. [GPU 1台の検証](validation.ja.md)を実施する。image、精度、sourceのhash、生成した記録をまとめて保存する。

## ホストカーネルと複数ノードRoCE

**更新を入れる前と、2台で動かす手順の前に、カーネルとドライバーを確認してください。** 本リポジトリの実測は、MSI EdgeXpert（MS-C931）上の `6.17.0-1032-nvidia`、ドライバー 580.173.02、ConnectX-7 ファームウェア 28.45.4028 で行いました。カーネル `7.0.0-1019-nvidia` とドライバー 580.178.04 は、ここでは未検証です。

2026-09-15 時点の更新では、`linux-nvidia-hwe-24.04` 系のメタパッケージが `7.0.0-1019-nvidia` へ上がり、580 open ドライバーのモジュールもそのカーネル向けに入ります。同じ更新で `nvidia-driver-580-open` も 580.173.02 から 580.178.04 へ上がります（参照機2台の `apt list --upgradable` で 2026-09-18 に確認）。`apt` の更新でも DGX Dashboard の更新でも入るため、新しく導入した機体も最初の更新の後はこのカーネルで起動します。

このカーネルの既定設定では、2台間の RoCE 越しの NCCL が `NCCL WARN Call to ibv_reg_mr_iova2 failed with error Cannot allocate memory` で失敗することがあります。報告では、モデルのロードは済み、vLLM のプロファイル中や TP 通信で失敗し、`ib_write_bw` などの RDMA 単体試験は正常に見えます。NVIDIA の[更新に関する告知](https://forums.developer.nvidia.com/t/dgx-spark-update-advisory/383254)（2026-09-13）は、複数ノード・RoCE 構成の利用者に対し、DGX Dashboard 経由を含めてこのカーネルへの更新を見送るよう求めており、修正版は示していません。単体ノードの処理に影響するかは確認されていません。

[NV-Kernels PR #590](https://github.com/NVIDIA/NV-Kernels/pull/590)（未マージ。投稿者による分析で、NVIDIA の見解ではない）は、原因を Kexec HandOver（KHO）と特定しています。`7.0.0-1019-nvidia` のビルドは `CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y` です（パッケージの config で確認。`6.17.0-1032-nvidia` は KHO を既定では有効にしない）。KHO は起動時に、後の kexec 用の scratch メモリを確保し、CMA のページブロックとして解放します。その報告では約 9.3 GiB（4,761 ページブロック）で、`CmaTotal` には計上されません。RDMA のメモリ登録はページを長期間固定し、固定するページは先に CMA の外へ移す必要があります。GPU がメモリを使い込んでいるとこの移動が失敗し、登録が `ENOMEM` を返します。

2台とも同じ対応にしてください。

| 選択 | 手順 | 補足 |
|---|---|---|
| `6.17.0-1032-nvidia` を使い続ける | 更新の前に `sudo apt-mark hold linux-nvidia-hwe-24.04 linux-image-nvidia-hwe-24.04 linux-headers-nvidia-hwe-24.04 linux-modules-nvidia-580-open-nvidia-hwe-24.04 linux-tools-nvidia-hwe-24.04 nvidia-driver-580-open`。設定後は `apt-mark showhold` で6つとも並ぶことを確かめる。既に 7.0 を入れた場合も旧カーネルは残るので、GRUB メニューの詳細オプションから起動する（コンソール接続が必要）。 | 本リポジトリで検証済みの状態。修正版カーネルが出たら hold を外す。 |
| `7.0.0-1019-nvidia` を KHO 無効で使う | `/etc/default/grub` の `GRUB_CMDLINE_LINUX_DEFAULT` に、既存の値を残したまま `kho=off` を足す。`sudo update-grub` の後に再起動する。`/proc/cmdline` に `kho=off` があり、`sudo ls /sys/kernel/debug/kho` が "No such file or directory" で失敗することを確認する。 | 告知スレッドに投稿された回避策。PR では KHO 無効で、2台間のメモリ登録・NCCL・TP2 の試験が通ったと報告されている。本リポジトリでは未検証。KHO は kexec による稼働中更新のための機能で、この構成では使わない。この行はドライバー 580.178.04 も受け入れることになり、`kho=off` は下記のホスト固着には効かない。 |

**同じ更新には、RoCE とは別の故障の報告があります。** Spark 2台の他レシピは、DGX OS 7.5.0→7.6.0 の更新（カーネル `7.0.0-1019-nvidia`、ドライバー 580.178.04、Docker 29.6.2）から約24時間のうちに、2台とも通常の配信負荷でホストごと固まったと報告しています（amasu、コミット `030d37e` の投稿草稿、コードは採用しない）。2日間に2つの配信スタックで5回以上、ping は返るが sshd が応答せず、復旧は電源の入れ直しだけでした。固まる前のカーネルログには `NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] ... returned from _memdescAlloc` が連続し（1回の起動で65回と148回）、そのとき MemAvailable は 9.4 GB・swap の空きは約73%で、OOM killer・Xid・panic はどれも記録されていません。同じ機体は 7.5.0・580.173.02 で数週間安定していたとあります。報告は、7.6.0 のリリースノートが Spark 向けに挙げるドライバーは 580.173.02 で、580.178.04 のサポート表に GB10 が無いことも指摘しています。ハードウェア診断の結果が未記入の草稿であり、示されているのは相関で、原因の確定ではありません。KHO は RDMA のメモリ登録の失敗を説明しますが、この固着は説明しません。ドライバーも hold の対象に入れ、更新後にこの症状が出たら `journalctl -b -1 -k | grep -c _memdescAlloc` で前回起動を確かめてください。

どちらを選んでも、提供を始める前に [NCCL 検証](nccl-validation.ja.md)とフルモデルの起動をやり直してください。

同じエラーの別の報告もあります。Ubuntu 汎用の 7.0 カーネル・ドライバー 595.84 の MS-C931 機で、空きが約 118 GiB あり重みのロード前だったにもかかわらず失敗し、MSI のボードファームウェア更新（組み込みコントローラー、SoC ファームウェア、USB-C PD）で解決したとしています（[MiaAI-Lab issue #259](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/259)）。メモリに余裕があるのにこのエラーが出る場合は、メーカーのファームウェアも確認してください。

**異常終了後は負荷時のクロックを確認する。** tonyd2wildはwatchdog reset後に約14 Wの電力制限が残り、負荷時611〜890 MHz、BF16行列演算の速度が正常機の約半分になったと報告しています。GPU reset・クロック／電力設定・再起動では戻らず、ACの抜き差しで復旧しました。GPU reset直後に再起動を挟まずCUDAを実行した場合もfaultが発生しています（speed-night報告、commit `9f5cc2c`。コードは採用しない）。基準の2台で再現した事実ではなく、外部の観測です。再起動後の低速化をモデル変更の効果と読む前に、同じ負荷でGPUクロックと電力を記録してください。アイドル時の電力だけではこの制限を診断できません。

## ネットワークとサイト設定

物理接続と永続的なIPv4設定は、[QSFPのハンズオン手順](qsfp-network.ja.md)に従います。

各ホストで実測した値を、[起動設定TOML](server-configuration.ja.md)の `[nodes]` 節に記録します。同じファイルを両ホストに置きます。

- 自機のfabric IPv4、headのfabric IPv4
- Ethernet interface、RDMAのHCA、そのinterfaceのRoCEv2 GID index
- `[api]` の未使用のAPIポートとrendezvousポート

HCAとGIDの番号は、両ホストで一致している必要はありません。GIDが自機のIPv4とnet deviceに対応することを確認してください。MTU 9000は、両端と経路全体が対応する場合にだけ使います。フルモデルをロードする前に、実際のNCCL transportとcollectiveの正当性を検証します。SSHで接続できることはRDMAの試験ではありません。

```sh
python -m glm53_setup server plan --rank 0
python -m glm53_setup server preflight --rank 0
```

`plan` は何も起動せずにcontainerコマンドを表示します。`preflight` は検査結果をJSONで表示し、一つでも失敗すれば非ゼロで終了します。完了済みで内容の一致するダウンロード状態が必要です。`start` は同じ検査を先に行い、合格したときだけ検査結果・設定・containerコマンドを `records/<timestamp>-server-r<N>/` に保存します。

## フルモデルの起動検査

`server preflight --rank N` は各ホストで、固定snapshotとMTP view、fabric設定、選択したimage IDと機能marker、LPA有効時のprojector checksum、他のコンテナがGPUを使っていないこと、空きメモリを検査します。`server start` も同じ検査を行い、失敗があれば起動しません。両rankの切替では、稼働中の対を止める前と新しい対を起動する前に、両rankでこの検査を繰り返します。

`server preflight` はimageの機能markerも検査します。有効な機能はそれぞれ、[imageの契約](server-configuration.ja.md#現行イメージの契約)に挙げたmarkerをimageの環境変数に見つけなければなりません。一つのmarkerには復旧のための例外があります。`runtime.canonical_moe_order = true` の場合、新規の起動には `GLM53_MOE_ORDER_API=2` が必要です。marker 1は、token整列のbuffer長を誤っていた以前のimageも持っているためです。marker 1のimageを指すprofileは、`server preflight`・`server start`・`cluster switch` の停止前検査で `moe_order_support` が不合格になります。参照imageを作り直し、`reference_image` を更新してください。すでにmarker 1のimageで稼働している対が取り残されることはありません。切替はその対を復旧先として検査し、新しい対が失敗した場合はmarker 1のまま再起動して、rankの `warnings` に `moe_order_marker_1_accepted_for_recovery` を記録します。稼働中のmarker 1の対のprofileに同じ読み取り検査を手で行う場合は、`server preflight` に `--recovery` を付けます。このflagは切替の復旧経路のためのもので、古いimageを新規に起動する手段ではありません。

三つのbuild時patchが上流vLLMの不具合を直しており、固定しているvLLMが上流の修正より先へ進んだら外します。どれもほかのbuild時patchと同じくsourceのSHA-256を照合し、どの検査もそのmarkerを要求しません。

- `GLM53_SLOT_MAPPING_GUARD=1`（1.7.0から作るimage。`glm53_setup/runtime/patch_slot_mapping.py`、vLLMのissue #53982）：slot対応付けのkernelが、block tableを行の中だけで読みます。これがないと、公開した任意設定は約25万tokenを超える要求で失敗します（[実測](benchmarks.ja.md#基準の2台の配信profile)）。このmarkerを要求する検査はないので、古いimageも起動できます。
- `GLM53_KPOOL_SEED_STRIDE=1`（1.13.0から作るimage。`glm53_setup/runtime/patch_kpool_seed.py`、vLLMのpull request #57477と同じ変更）：kpoolのprefill seed kernelが、indexerの生tailのblockをtail自身のstrideで番地付けします。これがないと、prefillのたびに要求自身のtail blockがseedされず、最後のtokenのkeyとgateが番号の小さいindexer blockへ書き込まれます。そのblockは別の要求のものであり得ます。壊れるのは長い文脈での疎なtop-kの選択で、prefix cacheから再利用する長いpromptで最も効きます。基準の2台では、patchを入れても測った出力は一つも変わりませんでした（[実測](benchmarks.ja.md#1130での測定)）。このmarkerを要求する検査はありません。
- `GLM53_KPOOL_RING=1`（現行のsourceから作るimage。未リリース。`glm53_setup/runtime/patch_kpool_ring.py`、vLLMのpull request #58454と同じ変更）：indexerの生tailのringを、1 pool分ではなく、MTPの深さkに対して `kpool * next_power_of_2(ceil((kpool + k) / kpool))` slotにします。MTPなしは4（従来どおり）、k = 1〜4は8、k = 5は16です。これがないと、kが2以上のとき、poolを完成させるdraftが棄却されると、その後ろのdraftが上書きした後のkeyからやり直すことになり、indexer cacheに誤った圧縮keyが入ります。壊れるのは、decodeで2,048 tokenを超えてpoolを作った後の疎なtop-kの選択です。kernelのfileへは `patch_kpool_seed` の後にだけ当たります（固定するhashはseed patchの出力）。上流はこれを部分的な修正とし、続く変更を予定しています。kernelの再現手順は[検証](validation.ja.md#kpool-tail-ringの再現)にあり、再現もフルモデルでの影響もまだ測っていません。このmarkerを要求する検査はないので、古いimageも `cluster switch` を通ります。

`exclusive_gpu` は、このランチャーの `glm53.experiment.startup` ラベルを持たない稼働中のコンテナがGPUを要求していると不合格になり、該当するコンテナを結果の `foreign_gpu_containers` に並べます。GPUの要求は `HostConfig.DeviceRequests` が空でないことで判定します。`--gpus` とCDI（`--device nvidia.com/gpu=...`）の要求はどちらもここに表れ、GPUのデバイスノードは `Devices` には表れません。このランチャーの対はfingerprintによらずラベルを持つので、旧い対が動いたままでも `cluster switch` の停止前の検査は止まりません。値が空のラベルは数えません。検査の途中で終了・削除されたコンテナは飛ばし、一覧に残っているのに調べられないコンテナがあれば、合格にせず例外で止めます。部品試験や別のモデルなど、他のGPU負荷は起動前に止めてください。無効化のオプションはありません。`server assets` もメモリ以外の検査として同じ判定を行います。着想はsfxnz PR #12（コードは採用しない）です。

preflightの合格は資材と設定の確認であり、品質や可用性の保証ではありません。通常運用の受け入れ項目と各項目の証拠の所在は[セットアップ手順](../SETUP.ja.md#6-フルモデルの検証)に、範囲別の現状はREADMEの状態表にあります。失敗した検査の緩和、attention候補の切り捨て、無断の精度変更で通過させないでください。

rank 1をheadlessで先に起動し、workerがrendezvousを待つ状態になってからrank 0を起動します。APIはhead側のloopbackアドレスにbindするため、遠隔クライアントからはSSHトンネルを使います。内部のrendezvousにはfabric IPを使います。事業サービスとして公開するには、別途検討した認証・TLS・アクセス制御の層が必要です。本リポジトリは、それを提供すると主張しません。

## 監視・停滞検知・warmup

各rankの前面の監視プロセスは2秒ごとに `MemAvailable` を読み、`resources.reserve_gib` を割ると自分のcontainerを停止します（`stop-reason: memory-reserve`）。rank 0では `resources.stall_seconds` が正のとき、同じ周期で `/metrics` も読みます。要求がrunningのまま、生成token計数・prompt token計数・KV使用率・running数のどれもその秒数動かなければ、`stop-reason: engine-stall` で停止し、止まったままの標本を記録します。engineが固まっても `/health` は200を返し続ける（V1のhealth checkはworkerを調べない）ので、生存の信号にはなりません。chunked prefillの間はKV使用率が動き、prompt token計数は最初の出力tokenで加算されるため、長いpromptは停滞になりません。テンプレートの600秒は `generation.timeout_seconds` と同じ値で、実測の最長の要求（256K・chunk 2048の参照要求で493秒）が収まります。`/metrics` に届かないときは証拠なしとして数えません。`resources.jsonl` の各行には `mem_free_gib` と `free_2mib_gib` も記録します。後者は `/proc/buddyinfo` から全zoneを合算した、2 MiB以上のbuddy blockに入っている空きです。tonyd2wildのGB10メモリの記録（コードは採用しない）によると、NVRMはページキャッシュを追い出さずにこの大きさのblockを確保するため、4 GiB以上空いていても `NV_ERR_NO_MEMORY` が出ることと合います。基準のheadでは配信中、MemAvailable 7.1 GiBのときMemFreeは1.1 GiB、2 MiB以上のblockは0.49 GiBでした。どちらも観測値です。rankを止めるのは `MemAvailable` だけで、読み取りに失敗した標本は停止させずに `memory_sample_error` として記録します。同じ記録では `vm.min_free_kbytes` を4 GiBに上げるとvLLMの起動時のメモリ検査が約6.2 GiB下がったとあるため、このキットでは配布時の既定値のままにします。KV 使用率を信号に含める理由は実測で裏付けられています。82,018 token の prefill 中、token 計数 2 つは 202.8 秒凍結したままでしたが、4 信号すべてが同時に凍結した最長は 8.3 秒でした。token だけを見る検知器は、運用する最長 prefill より上に閾値を置かざるを得ません。監視による停止はもう一方のrankを残すので、新しいpairを起動する前にそちらも止めます（`cluster switch` は不完全なpairを拒否します）。これらの記述はMia PR #70の現場記録を参考にしました。そこでの2件はどちらも、containerを強制終了し、短いCUDA probeでGPUを確かめ、再起動するだけで復旧し、電源断は要りませんでした。

`resources.jsonl` の各行には、containerのcgroupメモリと、その全プロセスの `VmRSS`／`RssAnon` の合計（`container_cgroup_gib`・`container_rss_gib`・`container_anon_gib`）も記録します。cgroup v2と `/proc` を権限なしで読みます。GB10ではGPUがホストのメモリを共有し、device側の確保はcgroupにもプロセスにも計上されないので、rankが `memory-reserve` で止まった時、記録そのものが二つの場合を切り分けます。cgroupとRSSが平らなまま `MemAvailable` が減るならdevice側の増加（decode Graph有効時の200K prefillで観測）、RSSが増えるならプロセス側の増加です。読めなかった観測は `container_memory_error` として記録し、rankの停止条件にはしません。

参照機はホストページ用に16 GiBのswapを持ちます。`vm.swappiness=0` は新しいページアウトを止めますが、既にswapに出たページは戻しません。長いprefill中に古いswapページへ触れたことがGB10のUVM livelockの引き金だったと同じ出典が報告しています。両containerが止まっている間に残りのswapを巡回します：`sudo swapoff -a && sudo swapon -a`。swapファイル自体は残します。swapを無くすと、確保の山でworkerがkillされました。2026-09-17に基準の対で、chunk 2048のまま `vm.swappiness` の60と0を比べました（0の前にswapを巡回）。60ではエンジンのプロセスにswapへ出たページはなく、headの0.38 GiB・peerの0.28 GiBは検索コンテナやデスクトップのシェルなど他のプロセスのものでした。0ではswapは空のままで、prefillの差は1.9%（再起動をまたぐばらつきの範囲内）、headの最小空きは0.45 GiB低く、peerは0.31 GiB高くなりました。この負荷では効果が見えなかったため、ホストは配布時の60のままにします。Spark 2台の他レシピは0を `/etc/sysctl.d` に永続化することを必須としています（tonyd2wild OPEN-PROBLEMS §4、コードは採用しない）。永続化する前に、自分の負荷で測ってください。

**ホストのデーモンは同じ統合メモリを奪い合います。** 監視は `MemAvailable` が余裕を割るとモデルを止めますが、原因がホスト上の別プロセスのこともあり、その場合はモデルだけが止まって原因は残ります。2026-09-16 には peer 側の rank が余裕 2.5 GiB に対し 2.49 GiB で停止しました。原因は、もう一方のホストで動く監視ダッシュボードが、メトリクスを 1 つ取るたびに peer へ新しい SSH ログインを張っていたことで、その頻度は毎秒 3.6 回でした。ログインごとに logind セッションと polkit の認可チェックが生じて `polkitd` が 6 日で 3.40 GiB まで太り、セッションが変わるたびに `wireplumber` が Bluetooth オーディオのプロファイルを登録し直して `bluetoothd` が「登録済み」と拒否し、この 2 つも太りました（0.69 GiB と 1.15 GiB）。さらにログインごとに `/etc/update-motd.d` の全スクリプトが走り、毎秒約 660 個のプロセスを生んでいました。ダッシュボードが動くホストは自分のメトリクスを直接読むので、0.03 GiB のままでした。ダッシュボードのホスト別名に OpenSSH の接続再利用（`ControlMaster auto`・`ControlPersist`）を入れると、peer へのログインは毎分 218 回から 0 回になり、ダッシュボードの値も変わらず取れました。長時間運転の前に、各ホストで `journalctl -u ssh --since -60s | grep -c Accepted` でログイン数を数え、`ps -eo user,rss,comm --sort=-rss | head` でデーモンの大きさを比べます。漏れるデーモンへ `MemoryMax` を入れるときは `Restart=on-failure` も併せて指定します。これらのunitは `Restart=no` で配布されており、上限に当たって落ちたきり戻らないためです。なお片肺はAPIからは見えません。生き残った rank が `/health` に 200 を返し続けるので、両ホストで `docker ps` を見ます。

**原因の特定と、その後に確かめたこと。** peer で Bluetooth を止めても接続の出入りは止まりませんでした（新しい D-Bus 接続は毎秒 7.3 回続き、接続元はディスプレイマネージャのグリーターセッション）。新しいプロセス・スレッドの ID を数えるとホストの差がはっきりし、10 秒で peer は 6,642、モデルの負荷がより重い head は 176 でした。新規プロセスのスナップショットではログインメッセージのスクリプトが 8 秒に 30 回起動しており、`journalctl -u ssh` には 60 秒で 218 回の受理ログインが head の fabric アドレスから記録され、その親はダッシュボードでした。接続を再利用させると、peer の 30 秒あたりの値は、新しい D-Bus 接続が約 220 から 1、`polkitd` の増加が約 450 KB から 0、新しいプロセス・スレッド ID が約 19,900 から 1,191 になりました。Bluetooth を再び有効にして電源を入れ直した後も、300 秒の間 `polkitd`・`bluetoothd`・`wireplumber` の大きさは変わらず、2026-09-17 のダッシュボードの再起動では、その後の 60 秒で peer へのログインは 0 回でした。事故の最中に peer の `polkitd` へ入れた上限（`MemoryMax=512M` と `Restart=on-failure`）は二重の備えとして残しており、上限に当たったときの再起動はまだ起きていません。

**待機中のpairではpeerの喪失を検知しません。** Mia Issue #193は、同じ機種でこの事故を逆向きにした事例を報告しています。TP=2で連続配信している最中にheadのホストがすべてのネットワークで応答しなくなり、物理的な再起動が必要になりました。containerはOOM killされておらず、直前にkernel・NCCL・containerのエラーは何も記録されず、pstoreも空でした。worker側では、kernelがRoCEリンクのダウンを、NCCLが再送上限超過の完了を記録しました。worker containerは動き続け、運用者が止めるまで100 GB超を抱えていました。報告者は固まった原因をモデルに帰しておらず、そのホストでは以前にも異常終了があったと書いています。この機体に持ち込める点は2つです。各監視プロセスは自分が守るホストの上で動くので、ホストが固まれば監視も一緒に止まります。reserveが防ぐのはモデルによる統合メモリの枯渇で、ホスト自体の固まりではありません。また、待機中の片肺はどちらの監視でも止まりません。メモリはreserveを割らず、停滞検知はrunningの要求を条件にしているためです。各rankが相手を確かめてpeer喪失で停止する停止理由は設計案で、実装していません。

`server warmup` はreadiness後に、通常のchat endpointへ要求のladderを流します。短文1往復、tool呼び出し、合成画像1枚（`runtime.vision` 有効時）、`generation.warmup_long_tokens` を指定した場合はその長さのprompt（配信中のtokenizerで長さを合わせる）です。`context.max_num_seqs` が1より大きいときは、段の後にburstを流します。2から `max_num_seqs` までの幅ごとに、その本数の短文を同時に送り、複数の系列が一緒にdecodeする形を作ります。段を1本ずつ流すだけでは届かない形です。1系列（配布既定）ではladderは変わりません。burstが公開した任意設定の同時2系列profileで配信中のコンパイルを減らすかは、まだ測っていません。これらは配信中にカーネルのコンパイルが観測された形です（[画像入力](vision.ja.md#限界と未解決の事項)）。固定の起動はvLLM自身のJIT warmupを無効にしており、そのコンパイルの山が一度headを保護余裕の下へ押し下げました。コンパイル済みカーネルはruntime cacheに残りますが、ladderは起動のたびに同じカーネルを報告します。固定のTritonは、プロセス内で初めてカーネルを使うとき、コンパイルしたかディスクのcacheから読み込んだかに関わらずpost-compile hookを呼び、jit monitorはそのhookで警告を出すためです（TileLang側の判定もプロセス内のcacheだけを見ます）。ladderが最初のユーザー要求より前に済ませているのは、このプロセスごとの読み込みです。記録（`records/<stamp>-warmup-r0/result.json`）には段ごとの秒数・prompt token・結果、burstごとの幅・秒数・要求別の結果（失敗した要求はそのburstとladderの判定を不合格にするが、ladderの残りは止めない）、jit monitorがladderの前と最中に報告したカーネル名、その後prefix cacheをリセットしたか（`api.dev_endpoints = true` のときだけ。それ以外ではwarmupのpromptは追い出されるまでcacheに残る）が入ります。`generation.warmup = true` なら、`cluster switch` は両rankのreadiness後にrank 0でladderを実行し、結果を `result.json` の `warmup` に残します。ladderの失敗は記録されるだけで、切替の失敗や巻き戻しにはなりません。`cluster resume` はreadinessを再観測するだけでladderは流さないので、必要なら後からhead上で `server warmup` を実行します。長文段は起動のたびにフルprefillを払います（chunk 2048では256Kで約490秒・200Kで380秒、512では200Kで約500秒・82Kで206秒の実測）。参照機では、毎回のladderが同じ10個のカーネルを報告します。その中には `BuildPrefillChunkMetadataKernel` と、一度はユーザーの要求を処理中にコンパイルされたTileLangの `mhc_pre_big_fuse_with_norm_tilelang` の形が含まれます。2026-09-17の5回の起動でruntime cache（Triton 1,840・TileLang 55ファイル）は1ファイルも増えず、短い段は1〜2秒でした。 `BuildPrefillChunkMetadataKernel` には、長い要求の途中でしか現れない形があります。indexerは1要求の問い合わせ長×圧縮後の系列長が `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`（固定imageで512）の予算を超えると問い合わせ側を分割し、2つ目以降の区間は開始位置が0でなくなって、Tritonの別の特殊化を要求します。圧縮比は `index_kpool` の4なので、分割が始まる入力長は 134,217,728 ÷ `max_num_batched_tokens` × 4 tokenです。2048では262,144で、判定は「以下」のため、配布既定の256Kいっぱいの要求でも分割は起きません。4096なら131,072、8192なら65,536から始まります。chunkを上げるか `max_model_len` を262,144より上げるときは、その長さ以上の `generation.warmup_long_tokens` を指定して、このコンパイルを起動時に済ませてください。値は稼働中のcontainerのsourceと設定から読みました（2026-09-18）。分割される長さの要求は流していません。機構の着想はMia PR #203（コードは採用しない）で、固定imageのvLLM自身のwarmup keyは3つの区分を既に列挙しており、その修正は要りません。

## 復旧と記録

スクリプトは、失敗したcontainerや重みを削除せず、再起動用のwatchdogも導入しません。`server stop` が停止するのは、このランチャーの所有ラベルを持つcontainerだけです。同じrank名を作り直す前に、ログを保存し、停止したcontainerの名前を変更してください。分散実行で障害が起きた後は、両rankをまとめて再初期化します。

`state/` は現在の取得状態とサイト設定を保持し、`records/` はrunごとの証跡を保持します。休止中の取得は意図的な停止です。検証の待機は終了コード2で終わり、ダウンロードを再開しません。ローカル移送の実行中に、新しい取得を開始しないでください。

数値・backendの詳細な制約は[validation.ja.md](validation.ja.md)にあります。リリース準備では、未解決の失敗を隠さず残してください。
