# 運用手順 — ベータ版

[English](operations.ja.md)

**通常運用としてのTP=2デプロイは、まだ検収されていません。** 直列のフルモデル参照profileには[実験結果](validation.ja.md#フルモデルtp2の実験範囲)と[初期ベンチ](benchmarks.ja.md)がありますが、ガード付きの通常ランチャーは候補実装のままです。

## 資材の保管場所とパス

本節がデプロイ時の保管パスの正典です。モデルID・revision・base imageのdigestは[runtime.lock.json](../config/runtime.lock.json)で固定し、本リポジトリは重みを配布しません。運用者固有のホスト名、home配下の絶対パス、認証情報は公開ソースの外に置いてください。

| 資材 | 各Linuxホストでの既定の場所 | 役割 |
|---|---|---|
| 本体checkpoint | `$HOME/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/<revision>/` | 固定したモデル・config・tokenizerのview。重みファイルは同階層の `blobs/` ディレクトリへリンクし、データ本体はそちらが持つ |
| MTPメタデータview（任意） | `$HOME/.cache/huggingface/local-views/glm53-mtp-compatible/<revision>/` | 既存のtensorデータをリンクし、checkpoint同梱のBF16 MTPに合わせて量子化メタデータを調整する。元のsnapshotを編集せずに[viewを作成](speculative-decoding.ja.md#各linuxホストでの準備)する |
| LPA projector（任意） | [起動設定TOML](startup-configuration.ja.md)の `[lpa].projector` で選ぶ非公開ファイル。そのTOMLからの相対パスまたは絶対パス | 別途学習した補助重みで、NVIDIAのsnapshotにもソース配布物にも含まれない。[LPAの手順](lpa.ja.md)で対応するprojectorを入手または学習し、両ホストでhashを確認する。通常の推論とbatchingには不要 |
| Dockerのbase／reference image | Dockerが管理する保管領域 | 固定したbaseをpullし、本ソースからreference imageをビルドする。ソースのcheckout、image、checkpointは別々の資材 |
| ローカル設定と取得状態 | `<checkout>/state/` | サイト固有の起動設定と `download-status.json`。後者は実際に取得した `snapshot` のパスを記録する |
| runtime／JIT cacheと証跡 | `<checkout>/state/tp2-runtime-cache/`、`<checkout>/records/` | 再生成できるruntimeデータと非公開の実行記録。モデル重みでも配布物の入力でもない |

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
