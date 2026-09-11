# MTP投機的デコーディング — 実験用

[English](speculative-decoding.md) · [MTPなしの基準値](benchmarks.ja.md)

最初のTP=2 baselineは投機なしです。今回検証した候補は、取得済みNVIDIA checkpointに含まれるMTPによる1トークン先読み（k=1）です。別draftモデル、EXL3変換、DFlash2重みの取得は不要で、モデルライセンスは追加されません。既存の[成果物ごとのライセンス](licensing.ja.md)は引き続き適用されます。

## フラグを足すだけでは足りない理由

固定checkpointのlayer45にはMTPの889テンソルがあり、888個BF16・1個F32、計13.844 GiBです。標準MTP設定は`modelopt_fp4`を継承しますが、元の除外設定はMTP層を対象にしていません。この浮動小数点重みへ一律にNVFP4設定を当てることはできません。

今回の候補は、**別viewのメタデータだけ**に`*.layers.45.*`の量子化除外を追加します。元の重み・snapshot設定は変更しません。本体0〜44層は元の量子化を維持し、MTPのMoEはBF16のTriton、本体はMarlin W4A16で動かします。固定版のloaderはtargetのVllmConfigも利用するため、draft側のメタデータだけを別にしても十分ではありません。

## 各Linuxホストでの準備

元snapshotの公式checksum検証を先に完了します。同じHF cache全体をcontainer mountできる位置へ、新しいviewを作ります。リポジトリのルートから実行してください。

```sh
HF_ROOT="$HOME/.cache/huggingface"
REVISION=$(python -c 'from glm53_setup.config import REVISION; print(REVISION)')
python tools/prepare_mtp_view.py \
  --snapshot "$HF_ROOT/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/$REVISION" \
  --output "$HF_ROOT/local-views/glm53-mtp-compatible/$REVISION"
```

ツールはMTPのheaderを確認し、量子化済みMTPを誤ってBF16扱いすることを拒否します。header検査は本体checksumの代わりではありません。tensorデータは複製せず、リンクと変更メタデータだけを作ります。出所・hashのレポートを非公開で保存し、両台のview設定が一致することを確認します。既存viewは上書きしません。

検証済みTP=2参照イメージ・設定で、モデルパスに**view側**を指定し、次を追加します。

```sh
--speculative-config "$(cat examples/speculative.mtp1.json)"
```

[設定JSON](../examples/speculative.mtp1.json)はMTP・k=1・draft専用のTriton MoEを指定します。本体のMarlin、eager、APCなし、同時実行1、比較対象のcontext/KV設定を維持します。containerには元snapshotとviewを含むcache root全体を読み取り専用mountします。viewだけのmountではリンクが切れます。

通常ランチャーの検収ガードを回避するための手順ではありません。元へ戻す場合は原snapshotを指定し、投機設定を外します。viewと試験記録は保持します。

## 合否と性能比較

- 本体がNVFP4/Marlinのまま、MTPが非量子化の浮動小数点layerとしてロードされることを確認する。不足scaleを仮の値で埋めない。
- メモリを実測し、ホストの余裕を保つ。重みの算術では約6.92 GiB/rank追加だが、複製・一時領域・KVは別途必要。
- 基準と同じ公式ベンチを実行し、終了コードに加えて要求完了数・出力token数も独立検査する。warmupとclient同時数を明示する。
- case前後の`/metrics`原文を保存する。受理率は採用draft token増分／draft token増分、bonus込み平均受理長は`1 + 採用token増分 / draft回数増分`。この区間はwarmupや初期probeを含み得るため、測定要求のみの時間統計とは区別する。[vLLMの定義](https://docs.vllm.ai/en/v0.24.0/api/vllm/v1/spec_decode/metrics/)
- 最終回答、ツール、SSE、EOS・長さ上限、状態を確認する。greedy token/logprob差は診断として残し、推論文の逐語一致を要求しない。
- decodeだけでなく初動と全体throughputを見る。長い入力・短い出力では速くならない場合がある。kの増加はk=1合格後の別試験にする。

## k=1の実測結果

2026-09-12（Asia/Tokyo）、全モデルでbaselineと同じ5条件・21測定要求・1,344出力tokenが完了し、要求エラーはありませんでした。全要求が指定どおり64 tokenを出力しました。イメージ、本体演算、通信、負荷条件は[MTPなしの基準](benchmarks.ja.md)と合わせています。別の実行同士の初期比較であり、A/B/Aを繰り返した統計評価ではありません。client同時数2でもserverは`max_num_seqs=1`で待ち行列を作ります。

| 入力token | client同時数 | MTPのTTFT中央値（秒） | decode off → k=1（token/s） | 全体出力 off → k=1（token/s） | draft受理率 |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 0.267 | 14.29 → 24.14 | 13.71 → 22.15 | 92.4% |
| 2,048 | 1 | 6.522 | 14.15 → 22.37 | 6.06 → 6.85 | 91.0% |
| 8,192 | 1 | 26.257 | 13.96 → 20.55 | 2.21 → 2.18 | 70.5% |
| 32 | 2 | 3.116 | 14.27 → 23.54 | 13.71 → 21.38 | 85.7% |
| 2,048 | 2 | 15.820 | 14.22 → 21.49 | 6.08 → 6.80 | 84.0% |

decodeは`1000 / mean_tpot_ms`です。受理率は前述のcase単位counter区間から求め、該当するclient準備・warmupを含みます。時間測定要求だけの集計ではありません。採用／draft数は表の順に122/132、122/134、105/149、204/238、204/243で、bonus込み平均受理長は1.70〜1.92 token/stepでした。

runtimeのmodel memoryは各rank 95.17 GiBで、baselineより約6.97 GiB増えました。ホストの最小空きメモリは7.90/10.25 GiB、6 GiBの余裕を守る監視停止は発動せず、両rankともOOM killなしです。この回のAPI準備完了には約14.4分かかりました。重みロードは依然重く、要求ごとのTTFTとは別です。

基礎APIの11項目はすべて合格しました。最終回答の再現、日本語計算、SSE、自動ツール引数・戻り値、Messages、token countingを含みます。テキストは`stop`、ツール要求は`tool_calls`、Messagesは`end_turn`で終了しました。推論文は再実行で異なり、診断扱いを維持します。出力分布全体の同等性や広範な品質を証明したものではありません。

**判断:** 次のテキスト・ツール・ハーネス評価に使う、明示選択の実験profileとしてk=1を残します。短い入力のdecodeは約1.69倍ですが、8,192入力・64出力では全体throughputが約1.3%低下し、TTFTは24.387→26.257秒へ増えました。比較用、およびメモリ余裕が少ない場合やprefill中心の用途にはMTPなしも残します。k≥2、実際の同時実行2、graphs、APC、画像、両ハーネスは未検証です。

試験サーバーは両台とも停止し、メモリは回復しました。rank 0は終了コード0、rank 1はcontrollerのDocker stop猶予後に137で終了し、`OOMKilled=false`でした。分散構成の正常停止・復旧は未検収です。通常ランチャーへの昇格や、その検収証跡の発行は行っていません。
