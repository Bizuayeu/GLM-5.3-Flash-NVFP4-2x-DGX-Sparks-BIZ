# APC優先・未処理部分へのLPA：実装・検証計画

[English](apc-lpa-design.md) · [施策台帳](optimization-catalog.ja.md)

**状態：CPU契約、4層GPUの共有状態隔離、全モデルの校正・限定品質・運用検査、非同期MTPを含む併用と最終held-out参照評価を完了。** 用途によりMTPなしのprefix再利用構成を選ぶ。対応imageのmarkerを必要とし、手動RPCによるAPC併用は引き続き拒否する。[校正・併用結果](benchmarks.ja.md#apc優先lpaの損益分岐計測p22)／[現在のLPA使用範囲](lpa.ja.md#使用範囲)／[施策台帳](optimization-catalog.ja.md)を参照。履歴編集・分岐・保持圧力の追加評価は、この検収と分けて実施する。

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

最初の接続検証は、テキスト・eager・TP2・1系列・ローカルcacheで行う。Graph、PP、外部KV connector、細粒度Mamba prefix cache、複数系列を同時に追加しない。MTP／unpack融合／非同期検査の併用は、基本契約の通過後に個別の組合せとして検証する。

## 実装と検収の順番

4層GPU試験ではN=18,432、復元H=8,704で、近似suffixを通過した後も共有prefixの全検査対象hashが一致した。後続通常要求のHは8,704のままで、通常計算後に初めて17,408まで共有範囲が伸びた。対照・近似の後の通常要求・復帰対照の16出力tokenは一致し、合成近似の分布差は別に検出できた。これはTP1・eager・MTPなしの状態隔離の結果であり、全モデルや併用の受入ではない。

共有キャッシュ隔離の部品試験は、対応image内で次のコマンドから実行する。`--fixture` は全tensorのバイト照合を終えた4層fixture、`--output` は新規の保存先を指定する。GPUを1台使い、32K context・1 GiB KV・1系列で実行するため、別途コンテナ上限とホスト余裕を監視する。

```sh
python -m glm53_setup apc-lpa-fixture --fixture /fixture --output /out/validation
```

この試験は通常状態との違いが出る合成projectorをfixture専用に作る。実cache lookupの復元境界、共有prefixのGPUバイトhash、実際のquery省略数、後続通常要求の出力を検査する。全モデルの品質や損益分岐の評価には使わない。startup warmupは固定ソース内の明示的な範囲に限定して除外し、実要求にはscheduler由来のpolicyを必須とする。workerのキャッシュ形式は、モデル読込時に正規化された`fp8_ds_mla`を検査する。

損益分岐は、LPA/APCを有効にして`lpa.break_even_tokens=0`とした専用TP2サーバーで、Linuxホストから計測する。通常運用のTOMLを直接変更せず、計測用TOMLを両rankに揃える。次は、実際のjoint復元単位が4,352 tokenの条件に対応する例である。MTPの先読み再計算などで境界が変わる構成には、そのまま流用しない。

```sh
python -m glm53_setup apc-lpa-benchmark \
  --config state/apc-calibration.toml \
  --corpus records/corpus/documents.jsonl --corpus-sha256 '<verified-sha256>' \
  --output records/apc-calibration --cached-prefix-tokens 4352 \
  --eligible-tokens 128 512 1024 2048 4096 8192 --repeats 5
```

各測定前に共有cacheをリセットし、H>0では通常計算でprefixを作り直す。実際のN/H/Rとquery省略数を両rankで検査し、時間の測定区間にreset・priming・診断RPCを含めない。通常／LPA／復帰対照を繰り返し、最初の一巡をwarmupとして除外する。入力には指定コーパスのvalidation分割だけを使う。小さい残余長の追加計測には`--cold-only --eligible-tokens 0 1 4 16 32 64`を指定できる。校正用のB=0をそのまま運用推奨値とはしない。

| 段階 | 実施内容・完了条件 |
|---|---|
| 現在の単独評価 | 32K容量・時間とAPC off/on/offの速度、実hit、長文抽出・再参照・tool・cancelを確定。生結果を保持する |
| CPUの契約 | N/H/T/Bの境界、H=0、ほぼ全hit、複数cache groupの整合、全登録経路の上限、要求ID切替、cancel／preemption後の上限維持を検査する |
| 部品と小層fixture | 通常計算でprefixを作成→Hを復元→suffixにLPA→通常要求で再参照。近似以降が共有表に無いこと、元のprefix状態が不変であること、実際の省略数を確認する |
| 汚染を検出する試験 | 近似状態を意図的に通常状態から区別できるfixtureを使い、後続LPA off要求に流入しないことを検査する。oracleの値がたまたま同じであることだけで合格にしない |
| 損益分岐の計測 | H=0／H>0それぞれでRを段階的に変え、補助器warmupを除外したAPC＋通常／APC＋LPA／復帰を比較。Bを選び、条件・ばらつき・未測定範囲を記録する |
| 全モデル | hitなし／部分hit／ほぼ全hit、pool・block・T・B境界、長文の証拠位置、別文書・projector設定・LPA offへの分離、tool、SSE、cancel、容量、両rank停止復旧を確認する |
| 最終併用 | 既に採用したMTP3・unpack融合・非同期検査との組合せを同じ固定資産で検証し、機能受入・性能採用・既定設定を分けて記録する |

各要求には、N、実復元H、R、LPA採否、最初の近似位置、共有上限、実際のquery省略数・cache登録範囲を残す。速度測定で重いtensor hashやprofilerを有効にしない。品質は逐語一致・タスク正答・状態診断・復帰対照を分け、未完了を合格へ繰り上げない。検証用4 GiB reserveと通常8 GiB reserveを区別する。

起動オプションは既存のカテゴリ別TOMLへまとめ、APC優先モードとB、共通資料を通常計算する要求指定を文書化する。対応imageのmarkerを必須とし、schedulerの登録上限を確定できない手動RPCとの併用は拒否する。無効な要求オプションと内部policyの持込みはinput processorでも検査し、schedulerへ投入する前に入力エラーにする。
