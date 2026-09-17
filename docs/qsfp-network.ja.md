# QSFP直結のハンズオン — NetworkManager

[English](qsfp-network.md) · [セットアップ全体](../SETUP.ja.md)

目的は、2台間の固定IPv4経路を作り、管理用Wi-Fiの既定経路を維持することです。**物理リンク、IP疎通、RoCE設定、NCCL実通信は別々に確認します。この手順の完了はTP=2合格ではありません。**

## 1. 作業前に記録する

Windowsでは、スタートメニューからPowerShellまたはWindows TerminalのPowerShellを開きます。既存のSSH設定を使う場合は、設定ファイルと登録済みホスト名を指定します。以下のパスと`node-a`は例示なので、実環境の値へ置き換えてください。SSHの秘密鍵を手順書へ貼り付ける必要はありません。

```powershell
ssh -F 'C:\path\to\ssh_config' node-a
```

設定ファイルを使わない環境では`ssh USER@MANAGEMENT_HOST`のユーザー名・管理用ホストを置き換えて接続します。初回のホスト鍵は別の信頼できる経路で照合し、不一致警告を無視しません。接続後に`hostname`で対象機を確認します。PowerShellに戻るには`exit`です。2台目は別のPowerShellタブから接続すると取り違えにくくなります。

各ホストへ管理用SSHで接続し、その端末を開いたままにします。以下はLinux側で実行します。Windows PowerShellへ直接貼り付けないでください。

```sh
hostname
nmcli --version
nmcli -f NAME,UUID,TYPE,DEVICE connection show
ip -brief link
ip -brief address
ip route show table all
ip rule show
rdma link show
ibdev2netdev
```

管理経路、既存の接続UUID、対象interface、既存の自動接続優先度を非公開レポートへ記録します。既存接続の設定は、対象UUIDを指定して`nmcli connection show uuid <UUID>`で確認できます。秘密値を表示するオプションは使いません。新しいサブネットが既存LAN・VPN・コンテナ・ポリシー経路と競合しないことを確認してください。

## 2. ケーブルを挿してinterfaceを特定する

メーカーが対応を確認したケーブルを使用し、ポートとコネクターの向きを確認して、無理に押し込まず最後まで挿します。機種の取扱説明書に従ってください。接続済みなら、検査のために抜き差しする必要はありません。

```sh
ip -brief link
rdma link show
ibdev2netdev
```

対象Ethernetの`LOWER_UP`、RDMA側の`ACTIVE`／`LINK_UP`を確認します。NVIDIAの構成では物理ポート1個に複数の論理interfaceが対応するため、ケーブル1本で複数interfaceが上がることがあります。両機で同名interfaceを選ぶ必要はありません。[NVIDIAのポート対応表](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)と実測を照合します。

本手順では、各機から1つずつEthernet interfaceを選びます。同じサブネットを、同時に上がった複数interfaceへ重複設定しません。もう一方も活用する構成は別途検証します。

## 3. 値を決める

以下は**例示**です。ホスト名や機種固有の値を公開資料へコピーせず、実作業値は非公開レポートに保存してください。

| 項目 | ノードA | ノードB |
|---|---|---|
| 接続名 | `glm53-qsfp` | `glm53-qsfp` |
| 固定IPv4 | `10.53.0.1/30` | `10.53.0.2/30` |
| Ethernet interface | 実測して指定 | 実測して指定 |
| HCA／GID | IPv4設定後に確認 | IPv4設定後に確認 |

この/30では`.0`がネットワーク、`.3`がbroadcastで、2台に`.1`と`.2`を割り当てます。gateway・DNSは設定しません。最初はMTUを変えず、実際の値を記録します。

`ipv4.never-default yes`は、この接続にIPv4の既定経路を持たせない設定です。`autoconnect`は条件がそろった際の自動接続を有効にし、優先度は同じdeviceの候補profile間で選択するための値です。通信の経路優先度とは異なり、すでにactiveなprofileを自動的に置換もしません。そのため初回は明示的に`connection up`を実行します。[NetworkManager仕様](https://networkmanager.pages.freedesktop.org/NetworkManager/NetworkManager/nm-settings-nmcli.html)

## 4. 片方ずつprofileを保存・有効化する

AIは確認・記録を担当し、管理者認証が必要なら人が自分のSSH端末で`sudo`を実行します。パスワードをAIやレポートへ渡しません。

sudoが求めるのは接続先Linuxアカウントのパスワードです。入力中に文字や伏字が表示されなくても正常です。コマンドの末尾の`\`はLinux shellの行継続なので、その後ろへ空白を足さずに貼り付けます。エラーが出たら、後続コマンドへ進む前に内容を確認します。

まずノードAで変数を**実測値へ置き換えて**設定します。

```sh
FABRIC_IF='REPLACE_WITH_OBSERVED_INTERFACE'
FABRIC_CIDR='10.53.0.1/30'
FABRIC_PEER='10.53.0.2'
FABRIC_PROFILE='glm53-qsfp'
nmcli -f NAME,UUID,DEVICE connection show
```

同名profileがないことを確認してから、1回だけ作成します。

```sh
sudo nmcli connection add type ethernet \
  con-name "$FABRIC_PROFILE" ifname "$FABRIC_IF" \
  ipv4.method manual ipv4.addresses "$FABRIC_CIDR" \
  ipv4.never-default yes ipv6.method disabled \
  connection.autoconnect yes connection.autoconnect-priority 100
```

作成成功後だけ実行します。

```sh
sudo nmcli connection up "$FABRIC_PROFILE"
nmcli -f GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS,IP4.GATEWAY device show "$FABRIC_IF"
ip route get "$FABRIC_PEER"
ip route get 1.1.1.1
```

自機アドレスとpeer宛の対象interface、外部宛の管理経路が正しいことをAIまたは作業者が確認します。`ip route get`は経路選択の照会であり、インターネット疎通試験ではありません。この段階では他方にIPv4がないので、peerへのpingはまだ通らなくても異常ではありません。

確認後、ノードBで同じ手順を行います。`FABRIC_IF`はBの実測値、`FABRIC_CIDR`は`10.53.0.2/30`、`FABRIC_PEER`は`10.53.0.1`です。

同名profileが存在する場合は`add`を繰り返しません。既存UUIDと設定を確認し、正しければそのUUIDを`up`します。修正が必要なら、そのUUIDを指定した`connection modify uuid <UUID> ...`で対象を限定します。設定内容と所有者が不明なら変更前に解決します。

## 5. 双方向の疎通と経路を検査する

両機で、それぞれの変数を保持したSSH端末から実行します。

```sh
ip -4 address show dev "$FABRIC_IF"
ip route get "$FABRIC_PEER"
ping -I "$FABRIC_IF" -c 4 -W 2 "$FABRIC_PEER"
ip route get 1.1.1.1
nmcli -f connection.id,connection.uuid,connection.interface-name,connection.autoconnect,connection.autoconnect-priority,ipv4.method,ipv4.addresses,ipv4.gateway,ipv4.never-default,ipv6.method connection show "$FABRIC_PROFILE"
ip -details link show dev "$FABRIC_IF"
```

通過条件は、正しい自機IP、peerへの直結経路、双方向4/4応答、外部宛の元の管理経路、保存された自動接続設定です。外部アクセスも必要なら、通常利用する既知のHTTPS先へのDNS解決と到達性を別途確認します。

MTUを変更する前に両端を測ります。例えば両端が1500なら、IPv4 ICMPのpayload 1472で非断片化疎通を確認できます。

```sh
ping -I "$FABRIC_IF" -M do -s 1472 -c 4 -W 2 "$FABRIC_PEER"
```

MTUが異なる場合は値を流用しません。9000へ上げるのは、両端・経路の対応と変更時間枠を確認した後の別工程です。参照pairではprefillが最大2〜3%上がる代わりにホストあたり空きメモリが1.1〜1.7 GiB減ったため、1500で運用しています（[測定](nccl-validation.ja.md#チャネル数)）。ping成功から速度やRDMA性能は推定しません。

## 6. RoCEの対応を記録する

`ibdev2netdev`で対象interfaceに対応するHCAを特定します。そのHCAの`/sys/class/infiniband/<HCA>/ports/1/`にある`gids/`、`gid_attrs/types/`、`gid_attrs/ndevs/`を同じindexで照合します。

選ぶのは、typeがRoCE v2、ndevが対象interface、GIDのIPv4-mapped部分が自機の固定IPv4であるindexです。番号はホストごとに実測し、よくある`3`を決め打ちしません。これを[サイト設定](operations.ja.md#ネットワークとサイト設定)へ引き継ぎます。RDMA linkのACTIVEやGIDの存在だけでは、NCCLがその経路で通信した証明にはなりません。

## 7. 再接続・復旧・完了判定

保存したprofileとautoconnect設定の確認を「再起動試験済み」と記録しません。実際の再起動は、他のジョブと管理接続の回復方法を確認したメンテナンス枠で行い、起動後に手順5・6を再確認します。このHowToのために自動再起動は行いません。

失敗時は、対象interface・profile・経路・carrierを保存して調べます。NetworkManager全体のrestart、Wi-Fi profile変更、firewallの全面無効化を復旧の初手にしません。切り戻す場合は管理用接続から新profileのUUIDを指定して自動接続を止め、停止したうえで、事前に保存した旧profileを有効化します。

```sh
sudo nmcli connection modify uuid <NEW_UUID> connection.autoconnect no
sudo nmcli connection down uuid <NEW_UUID>
sudo nmcli connection up uuid <PREVIOUS_UUID>
```

`<...>`は実UUIDへ置き換える説明用表記です。旧profileがなければ最後の行は実行しません。profileは削除せず証跡を残します。

- [ ] 対応ケーブルを接続し、両端の物理リンクを確認した。
- [ ] 管理経路と既存profileを記録し、サブネット競合を確認した。
- [ ] 各機に正しい固定IPv4・never-default・autoconnectを保存した。
- [ ] 双方向pingと経路、MTUを確認した。
- [ ] HCA・IPv4に対応するRoCEv2 GIDを記録した。
- [ ] 再起動後の自動接続を確認した、または未実施と明記した。
- [ ] NCCL実通信・帯域・TP=2推論は別の未完了項目として引き継いだ。

結果は非公開の`records/<run-id>/REPORT.md`へ、工程、実行者、時刻・タイムゾーン、結果、証跡パス、次の操作を記録します。
