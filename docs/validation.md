# 取得と推論を分けて確認する

取得処理は固定revisionに対し、ファイル台帳、全ファイルのサイズ、重みindexが参照する全shardの存在を確認する。公式ハッシュとの照合はREADMEの `hf cache verify` で行う。

実行イメージの準備とP0起動ガードは実装中。取得済みの状態からGB10の複数ノードへ進める際は、ロード時のピークメモリ、量子化カーネル、KVキャッシュ、集合通信、MTPの実ロードと採用を順に確認する必要がある。実重みTP=2のロード・生成はまだ合格していない。

`python3 -m unittest discover -s tests -v`でCPU契約を検査する。ネットワーク設定、固定revision、検証記録、P0の引数、HFキャッシュmountが対象。GPU演算・NCCL・実モデル品質はこのテストの対象ではない。

`prepare_image.py`は公式imageの取得とGPU smoke/モデル登録検査、`inspect_runtime.py`は実config検査を行う。いずれの結果も、`service.py`が必要とする`tp2-kernel-validation`の代わりにはしない。

## NoPE互換性の検査

固定公式イメージではconfig解釈が通っても、GB10のnative sparse MLA呼出しはNoPE形状を拒否した。`probe_attention.py`でその差を再現する。

`Dockerfile.reference`は全候補を保持するeager参照経路を組み込む検証用イメージ。実際のFP8キャッシュ書き込みを使う`tests/gpu_reference_attention.py`で、独立FP64計算との比較、page境界・2048超の候補数・padding、tail削除の検出、実backendとslot変換経由の一致を確認する。候補の削除は行わない。FP8キャッシュをFP32へ展開する丸めと、最終出力のBF16丸めを分けて検査する。

この参照経路はhost同期とPyTorch演算を含み、CUDA graphsや常用性能を保証しない。attention単体の試験を、画像・MTP・全45層のロード・KDA全体・2rank集合通信の合格記録として流用してはいけない。常用profileへの昇格は、未了の統合検査の後に行う。

## 単体の小型モデル統合

`make_fixture.py`は固定checkpointの先頭4層と共有language重みを、別の検証用checkpointへ切り出す。hidden/head幅・experts数・tensor bytesを維持し、各出力tensorのSHA-256を元と照合する。KDA 3層とNoPE MLA 1層を含み、MTPとvisionを含まない。元snapshotは書き換えない。

`run_fixture.py`はこのmarker付きcheckpointだけを使い、`summarize_fixture.py`が生成結果の完全性・有限性、A→B→A再現、batch間・prefill/decode間の差を評価する。コマンドJSONを生成結果へ混ぜない。検査は固定イメージ内、GPU 1台、外部networkなし、メモリ上限付きで実施する。

確認結果は、**NoPE参照経路＋Marlin W4A16**で小型モデルのロード・生成・状態比較を通過。attention管理blockの8,704-token境界を越える8,705-token入力も確認した。一方、CUTLASS W4A4では生成できるが数値不変性の条件に未達があり、batch-invariantモードはSM120 sparse MLA非対応で起動を拒否した。元のW4A4構成とMarlin構成を同値とは扱わない。

この検証は全45層・TP=2の合格ではない。数値結果、精度の根拠、実行コマンド、失敗の原因はローカルの[単体統合レポート](../records/20260911-single-node-fixture/WORKLOG.md)を参照する（単独checkoutには含まれない）。

重み配布元の想定ハードウェアや量子化形式は、そのままGB10上の動作証明にはならない。公開する性能値は、条件を揃えた実測結果が得られてから記載する。
