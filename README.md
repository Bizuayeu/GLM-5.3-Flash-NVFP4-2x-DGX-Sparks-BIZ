# GLM-5.3-Flash — NVIDIA NVFP4

NVIDIA配布の公式NVFP4重みを取得・確認し、2台のGB10で動かすための独立リポジトリ。モデルIDと固定revisionは [download_model.py](download_model.py)、実行候補は [runtime.lock.json](runtime.lock.json) に保持する。**取得・準備・起動ガードを実装中。実重みTP=2推論は未検証。**

## 取得と検証

リポジトリのルートでPython環境を作る。重みは既定のHugging Faceキャッシュに置く。

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-hf.lock.txt
.venv/bin/python download_model.py --background
```

状態は `state/download-status.json`、ファイル台帳は `state/model-manifest.json`、ログは `state/download.log` に保存する。同じ取得処理を並行実行しない。

`complete` は全ファイルの存在・サイズ・重み索引を確認した状態。公式チェックサムは次のコマンドで別途検証する。

```sh
.venv/bin/hf cache verify nvidia/GLM-5.3-Flash-NVFP4 --revision 423acf37583782c51c142d145aef733d72943d93 --fail-on-missing-files --fail-on-extra-files --json
```

## TP=2の準備

```sh
python3 prepare_image.py --background
python3 -m unittest discover -s tests -v
```

公式ARM64イメージを固定digestで取得し、GPUの小規模計算とGLMアーキテクチャ登録を検査する。結果は`records/`に保存する。取得済みであってもGLMのロード・生成の合格とはしない。

[site.example.json](site.example.json)を`state/site.json`へコピーし、各ノードのrank、fabricのIPv4、NIC、HCA、RoCEv2 GIDを実物から設定する。例のGIDやIPを無確認で適用しない。

```sh
python3 service.py plan
python3 service.py preflight
python3 service.py start
python3 service.py status
python3 service.py stop
```

rank 1を先に開始し、rendezvous待機を確認してrank 0を開始する。APIはheadのloopbackのみで待ち受ける。起動には固定image・revisionの`state/kernel-validation.json`（`kind=tp2-kernel-validation`かつ`passed=true`）が必要。この記録は実kernel検証の結果であり、手で合格に書き換えない。QSFP未接続、モデル取得未完了、未検証image、メモリ不足の状態では起動しない。

現在はP0（32K、同時1件、MTP/graphs/APC/vision無効）のみを実装している。停止したコンテナも保存するため、再作成前にはログを保存して対象コンテナを別名へ退避する。自動削除・自動restartは行わない。

公式イメージのnative NoPE経路はGB10で未成立。全候補を保持する`Dockerfile.reference`とGPU照合テストを追加したが、常用イメージへの昇格前である。現在の`service.py start`を検証記録の手書きで通過させない。詳細は[検証範囲](docs/validation.md)と[第三者通知](THIRD_PARTY_NOTICES.md)を参照する。

現在の実施経過は[セットアップレポート](records/20260911-tp2-setup/WORKLOG.md)、構築計画は[ローカル計画書](docs/IMPLEMENTATION_PLAN.md)を参照する。どちらも非公開の作業資料で、単独checkoutには含まれない。

GPU 1台での小型モデル統合は、Marlin W4A16＋NoPE参照経路で検査を通過した。[検証範囲](docs/validation.md#単体の小型モデル統合)と[単体統合レポート](records/20260911-single-node-fixture/WORKLOG.md)を参照する。全モデルの常用構成への昇格は未実施。

## 構成と検証範囲

コードと固定取得依存はルート、説明は [docs/validation.md](docs/validation.md)、生成状態は `state/`。状態・ログ・重みはGitで追跡しない。

別ノードへキャッシュを複製するときは、`blobs` と `snapshots` の参照関係を保ち、複製先でもチェックサムを検査する。ファイル取得の成功は、GB10でのロード・生成・MTPの動作を保証しない。

配布仕様と重みの利用条件は [NVIDIAモデルカード](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) を参照する。
