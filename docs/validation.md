# 取得と推論を分けて確認する

取得処理は固定revisionに対し、ファイル台帳、全ファイルのサイズ、重みindexが参照する全shardの存在を確認する。公式ハッシュとの照合はREADMEの `hf cache verify` で行う。

実行イメージの準備とP0起動ガードは実装中。取得済みの状態からGB10の複数ノードへ進める際は、ロード時のピークメモリ、量子化カーネル、KVキャッシュ、集合通信、MTPの実ロードと採用を順に確認する必要がある。実重みTP=2のロード・生成はまだ合格していない。

`python3 -m unittest discover -s tests -v`でCPU契約を検査する。ネットワーク設定、固定revision、検証記録、P0の引数、HFキャッシュmountが対象。GPU演算・NCCL・実モデル品質はこのテストの対象ではない。

`prepare_image.py`は公式imageの取得とGPU smoke/モデル登録検査、`inspect_runtime.py`は実config検査を行う。いずれの結果も、`service.py`が必要とする`tp2-kernel-validation`の代わりにはしない。

## NoPE互換性の検査

固定公式イメージではconfig解釈が通っても、GB10のnative sparse MLA呼出しはNoPE形状を拒否した。`probe_attention.py`でその差を再現する。

`Dockerfile.reference`は全候補を保持するeager参照経路を組み込む検証用イメージ。実際のFP8キャッシュ書き込みを使う`tests/gpu_reference_attention.py`で、独立FP64計算との比較、page境界・2048超の候補数・padding、tail削除の検出、実backendとslot変換経由の一致を確認する。候補の削除は行わない。FP8キャッシュをFP32へ展開する丸めと、最終出力のBF16丸めを分けて検査する。

この参照経路はhost同期とPyTorch演算を含み、CUDA graphsや常用性能を保証しない。画像・MTP・実重みロード・KDA全体・2rank集合通信の合格記録として流用してはいけない。実行イメージへの採用と常用profileへの昇格は、未了の統合検査の後に行う。

重み配布元の想定ハードウェアや量子化形式は、そのままGB10上の動作証明にはならない。公開する性能値は、条件を揃えた実測結果が得られてから記載する。
