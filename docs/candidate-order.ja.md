# sparse候補の順序正規化

[English](candidate-order.md)

参照imageでは、GLMのsparse MLA候補を、物理cache indexへ変換する前に論理token順へ揃えます。投機的デコーディングに依存しない、vLLM／GLM runtime共通の修正です。通常生成・MTP・外部proposerが同じbackendの境界を利用します。

調査したtop-k kernelでは、同じscore・有効範囲・候補集合でも返却順序が変わり、有限精度のAttention出力に差が生じました。正規化は候補の選択内容・重複数・paddingを保存し、再採点・切捨て・重複除去を行いません。別の順序付きtensorを作り、共有indexer bufferとKVデータは変更しません。並べる基準は物理cache indexではありません。

## 導入と更新

既存の参照imageビルドで、この修正も自動適用します。

```sh
python -m glm53_setup build-reference
```

公式base imageだけには、本リポジトリのパッチは入っていません。Docker build時に固定vLLM sourceのハッシュを検査し、一致するものへ適用します。別sourceなら停止します。インストール済みvLLMの手編集や、ハッシュ不一致の回避は行わず、[導入手順のステップ4](../SETUP.ja.md#4-イメージ準備と参照実装の単体検証)に従ってください。

checkoutを更新しても、既存imageや稼働containerは自動更新されません。確認したcheckoutから再ビルドし、生成したimage IDを記録し、必要に応じてimageを移送して各ホストでIDを照合します。既存の管理された切替手順を使い、稼働中containerへ直接パッチを当てません。

新しい参照imageは `GLM53_CANONICAL_CANDIDATES=1` を設定します。利用する実imageを確認してください。

```sh
docker image inspect "$IMAGE" --format '{{.Id}}'
docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}'
```

`IMAGE` は、その導入で選択した参照imageです。ビルドログでsourceパッチの検査が完了し、環境に `GLM53_CANONICAL_CANDIDATES=1` があることを確認します。タグ名や環境変数だけを、記録したビルド／imageの同一性の代わりにはしません。

対照試験やロールバックでは、container環境を明示的に `GLM53_CANONICAL_CANDIDATES=0` とすると正規化を無効にできます。GLM backendではそれ以外の値を拒否します。変更は実行記録へ残してください。無効化を速度面から推奨するものではありません。

## 範囲と検証

実装は `glm53_setup.runtime.candidate_order` です。固定sourceへの参照パッチが、`FLASHINFER_MLA_SPARSE_SM120` のGLM経路へ組み込みます。vLLMの全top-k演算や全モデルbackendを一律に変更するものではありません。このruntimeコードにEuryaleパッケージの依存はありません。

CPU契約では全順列・重複ID・padding・空行・非連続tensor・整数の上限・入力非変更・論理順序を物理変換より前に揃えることを検証します。GPU部品検査、fixture実測、全モデル／TP=2の検収は区別します。他の層を含む完全な数値再現性や、採択境界でscoreが同点になった場合の候補集合の決定性まで保証する変更ではありません。現在の検収上限は[検証文書](validation.md)を参照してください。

### GB10での回帰検証と費用

2026-09-13〜14（Asia/Tokyo）、パッチ適用image `sha256:2051793f66cfe6e352104bfec38348bf2e75756bc06892d449675bc7759fcd43` でGPUの候補／物理変換の部品検査を通過しました。固定sourceへのDocker buildも成功しています。ローカルCPU suiteは172試験を実行し、固定runtime依存の13試験はskip、新しいtensor契約試験は実行して通過しました。

GB10 1台・4層・Marlin W4A16のfixtureで、C1／TP1／eager／APCなし、KV予算512 MiB、モデル長上限32,768を用いました。他の算術条件は試験ハーネス側で固定しています。射影・mHC・logitsは13行、NoPEは8 query単位、KDA三角solverは4 warpです。これらは比較のための条件で、この候補順序パッチが追加導入する機能ではありません。内部tensor採取と全state監査hookは無効にし、投機ケースの採択境界修復adapterと試験用proposer（制御ファイルの費用を含む）は残しました。

- 通常生成：入力3／129／513／2051／2052／16384・出力32に、入力129・出力128を追加。ONの全7条件で各6反復の出力が一致しました。OFFでは16K入力の反復差を再現し、失敗判定を負の対照として保存しました。
- 投機検証：最大容量12のengineで実効幅7／9／12、入力129／2051／16384、oracle／先頭棄却／中央棄却を比較。全27条件を各3回実行し、新しい通常対照との出力一致と、意図した最初の検証stepの採択・棄却を確認しました。試験用proposerを用いており、学習済みEuryaleや標準MTP draftの品質・速度の結果ではありません。

下表はwarmup 1回を除いた5反復の要求時間中央値です。prefillと生成を含みます。OFF／ONでtoken列が異なる条件や、OFFの16K反復差があるため、同じ入出力長の費用比較であり、同一出力の速度向上を証明するものではありません。

| 入力／出力token | OFF | ON |
|---|---:|---:|
| 3／32 | 480.52 ms | 479.82 ms |
| 129／32 | 509.93 ms | 513.07 ms |
| 513／32 | 603.01 ms | 606.46 ms |
| 2051／32 | 976.93 ms | 982.52 ms |
| 2052／32 | 975.74 ms | 982.08 ms |
| 16384／32 | 4443.69 ms | 4466.76 ms |
| 129／128 | 1925.63 ms | 1937.21 ms |

中央値の変化は-0.15%〜+0.65%でした。小さな負値を速度向上の根拠にはしません。採取済みID・2,176列での単体測定では、CUDA eventで測った1呼出し当たりの時間は1／8／10／13行で約0.043〜0.047 ms、512行で約0.292 msでした（warmup 10回後、100呼出し単位を5回測定）。Torchの一時的な追加確保は1行で60 KiB、512行で約21.3 MiBでした。候補tensorの作業領域であり、KV cacheをもう一つ確保するものではありません。

実装が構築済みimage内にあり、ハーネスが候補順序処理を注入していないことも確認しました。全jobはOOMなく終了しています。検収範囲は上記fixture条件に限ります。全モデルTP=2、標準MTP、batching／APC／LPA併用、長時間負荷、target品質の回帰は別工程です。既存の全モデル実測は記録したimageに対するもので、このパッチを含む再ビルド後は対応する回帰検証が必要です。
