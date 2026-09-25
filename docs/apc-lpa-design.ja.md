# APC優先・未処理部分へのLPA：実装・検証の契約

[English](apc-lpa-design.md) · [施策台帳](optimization-catalog.ja.md)

**状態：完了。** CPU契約、4層GPUの共有状態隔離、全モデルの校正・限定品質・運用検査、非同期MTPを含む併用、最終held-out参照評価を通過した（[証拠](#証拠)）。LPAは既定offで、FA2 prefillとは排他（1.6.0）。この経路は対応imageのmarkerを必要とし、手動RPCによるAPC併用は拒否する。履歴編集・分岐・保持：[fixture](component-validation.ja.md#履歴fixtureの追加)、[全モデル](benchmarks.ja.md#apcの履歴保持の基準検査)。[現在のLPA使用範囲](lpa.ja.md#使用範囲)と[施策台帳](optimization-catalog.ja.md)も参照。

## 目的と初版の方針

APCで復元できるprefixを先に再利用し、未処理部分が長いときだけLPAを使う。[vLLMのAPC説明](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)も、共通prefixのprefill省略を対象としている。初版の共有キャッシュには通常計算由来の状態だけを登録し、近似状態は当該要求の中だけで使う。近似状態の共有・専用namespaceによる再利用は今回の範囲外とする。

| 値 | 定義 |
|---|---|
| N | runtimeが確定した入力token数 |
| H | 必要なMLA・indexer/pool/tail・KDA状態を揃えて復元できるprefix長。文字列の共通長や、単一cache groupの最大hit長ではない |
| T | 通常計算する末尾長。入力が短いときはN以下に制限する |
| R | LPAを適用可能な未処理長 `max(0, N - T - H)` |
| B | 実測で決める損益分岐の判定値。普遍的な定数とは扱わない |

`R > B` のときだけ `[H, N-T)` を近似する。`[0,H)` は再計算せず、末尾 `[max(H,N-T),N)` とdecodeは通常計算する。Rが小さい場合は、APC＋残りの通常計算となる。H=0も同じ規則で扱い、総コンテキスト長による別の振り分けは作らない。

MTPでは、先読みを再計算するため、固定vLLMの[cache coordinator](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/v1/core/kv_cache_coordinator.py)が一致した末尾blockをhitから除外する。KDA checkpointもこの復帰境界に合わせる。したがって「1blockを通常計算したら1blockを復元できる」とは限らず、fixtureのprimingには追加の通常blockを含める。Hにはこの調整後に実際に返された値を使う。起動時に整列されるblock幅も、MTPなしの数値を流用しない。

## 共有キャッシュの契約

1. 最初に近似する位置Sを、GPU計算と共有登録の前に決める。初回の通常経路ではLPA採用時のS=H、近似しない要求には上限を設けない。
2. 共有できるのは、全状態が通常計算由来で、末尾がSを越えない完全なblockだけとする。各cache groupの圧縮比・block境界を維持し、上へ丸めない。
3. S以降のKV・KDA状態を要求の実行に使うことと、共有hash表へ登録することを分ける。要求内の割当・cache更新・decodeは維持する。
4. 通常計算する末尾も、それ以前の近似に依存するため共有しない。decodeで作った状態も同様とする。
5. chunk継続、preemption／再計算、cancel、終了処理で制限を失わない。一度近似の影響を受けた要求の上限を、黙って後方へ緩めない。
   初回に選んだ近似区間は通常のchunk進行で変えない。preemptionで元の共有prefixを失った場合は、その位置を通常計算で再構成し、元の近似開始位置を維持する。元が通常計算の要求は通常計算を維持する。
6. LPA要求が共有prefixの既存KV・KDA checkpointを破壊しないことを確認する。copy-on-writeや解放・再割当の既存契約を維持する。

初見の長文をLPAで処理しても、近似した全文は次回用の共有APCには育たない。共通system prompt・tool定義・資料を通常計算で温めるため、要求単位の明示的なLPA offを用意する。off要求は、実際に全て通常計算された状態だけを共有へ登録する。

## 固定runtimeへの接続点

対象vLLMは既存lockの `385dce36bcee42309924a5ece951a96db3dce7f2`。上流更新や別バージョンへのfallbackは行わず、patch対象のsource hashを照合する。

- schedulerのcache照合結果からHを得る。`shared_prefix_boundary`は、遅れているcache groupがまだ復元できない境界を含み得るため、Hの代用にしない。
- `KVCacheManager.allocate_slots()` は計算前に `coordinator.cache_blocks()` を呼ぶ。事後の応答処理だけでは抑止できない。この経路と明示的な `cache_blocks()` の両方で共有上限を適用する。
- 要求IDとN/H/T/R・採否・共有上限をschedulerからworkerへ渡す。workerの全体設定だけを外部RPCで切り替える方式に依存せず、要求の取り違えを検出する。
- 非同期MTPではCPU側の進行位置は楽観値で、GPUが却下した投機token分を補正する。生成済みtokenがあり、両位置がprompt終了以降にあるdecodeでは、GPUの補正位置を使う。prefillの位置一致、promptへの巻き戻り禁止、位置列の連続性、共有登録上限は維持する。診断の`speculative_position_corrections`に補正を観測したstep数を残す。
- LPA hookは絶対位置を使い、実際に計算する未処理位置だけを近似・計数する。capture／oracleはcache hitで欠けたprefixを「採取済み」と見なさない。
- APC優先モード中の手動RPCが、schedulerに伝わらない近似を有効化できないようにする。通常APCの対照が近似状態を保存する経路を残さない。

## 証拠

- 4層fixtureでの共有状態の隔離（MTP・融合・非同期検査を含む）：[部品検証](component-validation.ja.md#apc優先lpaのcache隔離p22)。
- 損益分岐の校正と閾値B：[ベンチマーク](benchmarks.ja.md#apc優先lpaの損益分岐計測p22)。
- 全モデル：[MTP・融合・非同期検査との併用](benchmarks.ja.md#apclpamtp融合非同期検査の併用p22)と[held-out参照](benchmarks.ja.md#同一入力を再利用する場合の差)。

fixture試験は対応image内、GPU 1台で、byte照合済みの4層fixtureと新規の出力先を指定して実行する。fixture専用の合成projectorを作るため、検査するのは隔離であり、品質や損益分岐ではない。

```sh
python -m glm53_setup apc-lpa-fixture --fixture /fixture --output /out/validation
```

校正は、LPAとAPCを有効にして`lpa.break_even_tokens=0`とした専用TP2サーバーで、両rankに揃えた計測用TOMLから実行する。`--cached-prefix-tokens`はそのprofileでの実際のjoint復元単位（ここでは4,352）で、MTPなどで境界が変わるprofileには固有の値が要る。

```sh
python -m glm53_setup apc-lpa-benchmark \
  --config state/apc-calibration.toml \
  --corpus records/corpus/documents.jsonl --corpus-sha256 '<verified-sha256>' \
  --output records/apc-calibration --cached-prefix-tokens 4352 \
  --eligible-tokens 128 512 1024 2048 4096 8192 --repeats 5
```

`--cold-only --eligible-tokens 0 1 4 16 32 64`で小さい残余長を追加できる。校正用のB=0は運用推奨値ではない。
