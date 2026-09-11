# GLM-5.3-Flash — NVIDIA NVFP4

NVIDIA配布の公式NVFP4重みを取得・確認する独立リポジトリ。モデルIDと固定revisionは [download_model.py](download_model.py) に保持する。**推論サーバーとCLIは未実装。**

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

## 構成と検証範囲

コードと固定取得依存はルート、説明は [docs/validation.md](docs/validation.md)、生成状態は `state/`。状態・ログ・重みはGitで追跡しない。

別ノードへキャッシュを複製するときは、`blobs` と `snapshots` の参照関係を保ち、複製先でもチェックサムを検査する。ファイル取得の成功は、GB10でのロード・生成・MTPの動作を保証しない。

配布仕様と重みの利用条件は [NVIDIAモデルカード](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) を参照する。
