# 運用手順

[English](operations.md)

**通常運用としてのTP=2デプロイは、まだ受け入れていません。** 直列のフルモデル参照profileには[実験結果](validation.ja.md#フルモデルtp2の実験範囲)と[初期ベンチ](benchmarks.ja.md)があります。profileが実験段階か通常運用可能かは、記録した受け入れ状態（READMEの状態表と[ハーネス受け入れ一覧](harnesses.ja.md#受け入れ試験一覧と実施状態)）で示し、コマンド名では示しません。

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
| runtime／JIT cacheと証跡 | `<checkout>/state/tp2-runtime-cache/`、`<checkout>/records/` | 再生成できるruntimeデータと非公開の実行記録。モデル重みでも配布物の入力でもない。分散起動はTriton・TileLang・TorchInductorのcacheをruntime cacheへ向け、コンパイル済みkernelを再起動後も残す |

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

`server preflight --rank N` は各ホストで、固定snapshotとMTP view、fabric設定、選択したimage IDと機能marker、LPA有効時のprojector checksum、空きメモリを検査します。`server start` も同じ検査を行い、失敗があれば起動しません。両rankの切替では、稼働中の対を止める前と新しい対を起動する前に、両rankでこの検査を繰り返します。

preflightの合格は資材と設定の確認であり、品質や可用性の保証ではありません。通常運用として受け入れるまでに残る項目は[セットアップ手順](../SETUP.ja.md#6-フルモデルの検証)に、範囲別の現状はREADMEの状態表にあります。失敗した検査の緩和、attention候補の切り捨て、無断の精度変更で通過させないでください。

rank 1をheadlessで先に起動し、workerがrendezvousを待つ状態になってからrank 0を起動します。APIはhead側のloopbackアドレスにbindするため、遠隔クライアントからはSSHトンネルを使います。内部のrendezvousにはfabric IPを使います。事業サービスとして公開するには、別途検討した認証・TLS・アクセス制御の層が必要です。本リポジトリは、それを提供すると主張しません。

## 監視・停滞検知・warmup

各rankの前面の監視プロセスは2秒ごとに `MemAvailable` を読み、`resources.reserve_gib` を割ると自分のcontainerを停止します（`stop-reason: memory-reserve`）。rank 0では `resources.stall_seconds` が正のとき、同じ周期で `/metrics` も読みます。要求がrunningのまま、生成token計数・prompt token計数・KV使用率・running数のどれもその秒数動かなければ、`stop-reason: engine-stall` で停止し、止まったままの標本を記録します。engineが固まっても `/health` は200を返し続ける（V1のhealth checkはworkerを調べない）ので、生存の信号にはなりません。chunked prefillの間はKV使用率が動き、prompt token計数は最初の出力tokenで加算されるため、長いpromptは停滞になりません。テンプレートの600秒は `generation.timeout_seconds` と同じ値で、実測で最長の要求（200K、506秒）が収まります。`/metrics` に届かないときは証拠なしとして数えません。監視による停止はもう一方のrankを残すので、新しいpairを起動する前にそちらも止めます（`cluster switch` は不完全なpairを拒否します）。これらの記述はMia PR #70の現場記録を参考にしました。そこでの2件はどちらも、containerを強制終了し、短いCUDA probeでGPUを確かめ、再起動するだけで復旧し、電源断は要りませんでした。

参照機はホストページ用に16 GiBのswapを持ちます。`vm.swappiness=0` は新しいページアウトを止めますが、既にswapに出たページは戻しません。長いprefill中に古いswapページへ触れたことがGB10のUVM livelockの引き金だったと同じ出典が報告しています。両containerが止まっている間に残りのswapを巡回します：`sudo swapoff -a && sudo swapon -a`。swapファイル自体は残します。swapを無くすと、確保の山でworkerがkillされました。

`server warmup` はreadiness後に、通常のchat endpointへ要求のladderを流します。短文1往復、tool呼び出し、合成画像1枚（`runtime.vision` 有効時）、`generation.warmup_long_tokens` を指定した場合はその長さのprompt（配信中のtokenizerで長さを合わせる）です。これらは配信中にカーネルのコンパイルが観測された形です（[画像入力](vision.ja.md#限界と未解決の事項)）。固定の起動はvLLM自身のJIT warmupを無効にしており、そのコンパイルの山が一度headを保護余裕の下へ押し下げました。コンパイル済みカーネルはruntime cacheに残るため、profileごとの最初の起動以後は、ladderは主に「何もコンパイルされない」ことの確認になります。記録（`records/<stamp>-warmup-r0/result.json`）には段ごとの秒数・prompt token・結果、jit monitorがladderの前と最中に報告したカーネル名、その後prefix cacheをリセットしたか（`api.dev_endpoints = true` のときだけ。それ以外ではwarmupのpromptは追い出されるまでcacheに残る）が入ります。`generation.warmup = true` なら、`cluster switch` は両rankのreadiness後にrank 0でladderを実行し、結果を `result.json` の `warmup` に残します。ladderの失敗は記録されるだけで、切替の失敗や巻き戻しにはなりません。`cluster resume` はreadinessを再観測するだけでladderは流さないので、必要なら後からhead上で `server warmup` を実行します。長文段は起動のたびにフルprefillを払います（200Kで約500秒の実測）。

## 復旧と記録

スクリプトは、失敗したcontainerや重みを削除せず、再起動用のwatchdogも導入しません。`server stop` が停止するのは、このランチャーの所有ラベルを持つcontainerだけです。同じrank名を作り直す前に、ログを保存し、停止したcontainerの名前を変更してください。分散実行で障害が起きた後は、両rankをまとめて再初期化します。

`state/` は現在の取得状態とサイト設定を保持し、`records/` はrunごとの証跡を保持します。休止中の取得は意図的な停止です。検証の待機は終了コード2で終わり、ダウンロードを再開しません。ローカル移送の実行中に、新しい取得を開始しないでください。

数値・backendの詳細な制約は[validation.ja.md](validation.ja.md)にあります。リリース準備では、未解決の失敗を隠さず残してください。
