# 推論最適化の全体像

[English](optimization-overview.md) · [施策台帳](optimization-catalog.ja.md) · [文書一覧](README.ja.md)

どの施策が推論のどの段階に効き、何を採用し、用途によってどの構成を選ぶかを一枚で示す読み物です。施策IDと採否の正典は[施策台帳](optimization-catalog.ja.md)、数値の正典は各実測文書で、本書の数値は条件付きの代表値です。数値を引用する際はリンク先の表と条件を確認してください。

## 基準構成

基準は[施策台帳の基準点](optimization-catalog.ja.md#今回の基準点と文書の役割)と同じで、context 16K・KV各rank 1 GiBで測定しています（[初期の測定条件](benchmarks.ja.md#初期の測定条件)）。32Kまでの容量は別に確認しています（[32K sweep](benchmarks.ja.md#32kまでの独立コンテキスト評価p15)）。

## 段階別の位置づけ

```mermaid
flowchart LR
    R[要求] --> A[prefix復元<br/>P19 APC・checkpoint保持・P22 判定]
    A --> P[prefill<br/>P11 chunk・P02 LPA・P03 unpack融合]
    P --> D[decode<br/>P01 MTP k=3・P08 非同期index検査・P06 Graphs 保留]
    D --> O[出力]
    S[並列・throughput<br/>P13 2系列 受入・P21 EP 不採用・P17 PP 不採用] -.- P
    S -.- D
    B[attention backend・indexer<br/>P04 P05 不採用・P16 保留・候補順序の正規化] -.- P
    B -.- D
```

## 施策と現在地

「採否」は台帳の判断（採用／受入／保留／不採用）、「既定」は[起動設定TOML](startup-configuration.ja.md)の既定値です。機能受入・性能採用・既定値・併用検収は別々に判断されており、採用でも既定onとは限りません。用途別の有効化は[用途別の構成](#用途別の構成)を参照してください。

### prefix復元（繰り返す会話）

| 施策 | 仕組み | 採否 | 既定 | 代表値と条件 | 正典 |
|---|---|---|---|---|---|
| P19 APC | 通常計算由来の状態だけを共有cacheに登録し、同一prefixのprefillを省略する | 受入（実測した直列・長文prefix再利用の実験用途） | off（`cache.prefix_caching=false`） | 16,320入力の再要求で約79%短縮。2Kは改善なし、長文2件のcoldは1.1〜1.7%悪化 | [P19](benchmarks.ja.md#全モデルのprefix-caching独立評価p19) |
| checkpoint保持（`cache.prefix_cache_retention_interval`） | KDA checkpointをscheduler blockごとに保持し、途中編集・分岐後に復元できるHを伸ばす | 採用（通常priming済み・直列の途中編集用途）。実測したarmは標準の間隔4,352。`dense`は実測した整列配置で同じ標準KDA maskになるが、最終併用の検収は別 | off（キー未指定＝固定runtimeの0。候補値は`dense`） | 約16K入力の50%位置編集・1出力で約27.6%短縮（A/B/A各5回）。編集・分岐・追記・別会話再訪・evictionの履歴試験を通過 | [保持A/B/A](benchmarks.ja.md#apcの履歴保持の基準検査)／[契約](launch-safety.ja.md#apcの履歴検証) |
| P22 APC優先LPA | 復元したHの先を、残余R＝N−T−Hが閾値Bを超えるときだけ近似する。近似状態は要求内に閉じる | 完了（校正・限定品質・最終併用・held-out）。B=128は保守的な候補閾値で、普遍的な損益分岐定数ではない | off（APCとLPAの両方を有効にしたとき動作。`lpa.break_even_tokens=128`） | H=0／4,352の12条件すべてでLPAが両対照より速い | [設計](apc-lpa-design.ja.md)／[校正](benchmarks.ja.md#apc優先lpaの損益分岐計測p22) |

### prefill

| 施策 | 仕組み | 採否 | 既定 | 代表値と条件 | 正典 |
|---|---|---|---|---|---|
| P02 LPA | cut=32（0始まり）以降の層で過去tokenのMLPを省き、末尾512 tokenは通常計算。生成時は全層 | 実測あり（実験用。一般品質は別ゲート） | off（`lpa.enabled=false`） | 8,192入力・1出力で約21.6%短縮（単独）。長文照合6件・tool往復は合格 | [LPA](lpa.ja.md) |
| P03 unpack融合 | FP8 MLA cacheの復元（コピー・FP32変換・scale乗算）をTriton 1 kernelに | 実測あり（部品一致・全モデルA/B/A。受入済みのP18併用の範囲で使用） | off（`cache.fused_unpack=false`） | 8Kの1出力対照で約17%短縮。短文decodeはほぼ不変 | [部品実測](component-validation.ja.md) |
| P11 prefill chunk | schedulerのtoken予算を128／512／1024で比較 | 既定512を維持。1024はthroughput候補、128は不採用 | 512（`context.max_num_batched_tokens`） | 1024は2Kの全体出力を約4.6%（1クライアント）／5.7%（2クライアント）改善するが最長停止が延びる | [P11](benchmarks.ja.md#prefill-chunk-の独立評価p11) |

### decode

| 施策 | 仕組み | 採否 | 既定 | 代表値と条件 | 正典 |
|---|---|---|---|---|---|
| P01 MTP k=3 | checkpoint同梱のBF16 draftを別メタデータviewで読み、3 token先読み。外部draftモデルなし | 選定（次の実験ではk=3を優先。k=2／k≥4は未試験） | off（`mtp.enabled=false`、有効時は`num_speculative_tokens=3`） | 短文decodeはk=1→k=3で24.1 → 30.3 token/s、その前のoff→k=1で14.3 → 24.1（1系列、別実行の比較）。長文の全体throughputはほぼ不変、TTFTは微増、約7 GiB/rank追加 | [k=3](speculative-decoding.ja.md#k3の比較結果)／[k=1](speculative-decoding.ja.md#k1の実測結果) |
| P08 非同期index検査 | 範囲検査を省かずGPU assertへ移す。tokenあたりCPU同期は各rankで約22回、copyは両rank合計で約22回減り、GPU kernelは11回増える | 受入（独立opt-in） | auto（`runtime.index_checks`。eagerでは同期＝実質off） | 128出力で約0.7〜2.5%の小幅改善、短文ほど大きい | [P08](benchmarks.ja.md#cpu同期削減の独立評価p08) |
| P06 CUDA Graphs | decodeのみcapture／replay | 保留（全モデル未検収。小層fixtureで2K入力の生成が分岐） | off（`runtime.enforce_eager=true`） | — | [Graph fixture](component-validation.ja.md#decode-graphのfixture独立評価) |

### 並列・throughput

| 施策 | 仕組み | 採否 | 既定 | 代表値と条件 | 正典 |
|---|---|---|---|---|---|
| P13 標準batching | `max_num_seqs=2`で実batch重複を作る。LPAは1系列限定 | 受入（限定throughput用途） | 1系列（`context.max_num_seqs=1`） | 16K×2の容量も確認 | [P13](benchmarks.ja.md#標準batchingの独立評価) |
| P21 Expert Parallel | Expert層の分割だけをTPからEPへ | 不採用（実測したthroughput負荷） | off（`runtime.expert_parallel=false`） | 全4ケースで両off対照より遅い | [P21](benchmarks.ja.md#expert-parallel-の独立評価p21) |
| P17 TP2／PP2 | 同じ2台をTP1×PP2に | 不採用（実測した生成負荷） | TP2（`runtime.pipeline_parallel_size=1`） | prefillは改善、decodeは低下。profiler再開始の障害でPP容量・decode traceは未完了 | [P17](benchmarks.ja.md#tp2pp2の独立評価p17) |
| P14 同種タスクbatching | 投入順を同種でまとめる | 不採用（この負荷） | —（設定項目なし） | 改善は約0.7〜0.9% | [P14](benchmarks.ja.md#同種タスクの投入順比較p14) |

### attention backend・indexer

| 施策 | 仕組み | 採否 | 既定 | 正典 |
|---|---|---|---|---|
| P04 NoPE attention融合 | Pythonのqueryループと多段演算の置換 | 不採用（launch削減だけでは速くならず） | —（serving未接続） | [P04](component-validation.ja.md#nope-attentionの融合とquery-batchingp04) |
| P05 SM121 backend選定 | 既存kernelへの候補幅適合 | 不採用（候補幅2176非対応・数値未達） | —（参照attentionを維持） | [部品検査](component-validation.ja.md#padding付きnative-attentionの直接試験) |
| P16 CSA2 | 層間の候補再利用・限定再採点 | 保留（部品保持、serving適用なし） | —（未統合） | [CSA2](indexer-reuse.ja.md) |
| 候補順序の正規化 | sparse MLA候補を物理index変換前に論理token順へ揃え、top-kの順序揺れを除く | 台帳外：新規ビルドの参照imageで有効なruntime共通の修正。**上記の全モデル実測はこの変更より前**で、patched imageには4層GB10 fixtureの回帰しかなく、再ビルドしたruntimeには全モデルの回帰が別途要る | on（新規ビルドの参照image） | [候補順序](candidate-order.ja.md) |

### 運用（性能施策ではない）

認証クライアント・allocator伝達・全HCA（レール）検査・両rank切替と復旧は運用の契約であり、高速化ではありません。[施策台帳](optimization-catalog.ja.md)はP10／P19／P22／E03の拡張として扱い、範囲と切替・復旧の制御試験の証拠は[起動契約](launch-safety.ja.md)が所有します。

## 直列併用の実測

単体の改善率を足したり掛けたりせず、併用状態を実測しています。

- **P18** MTP3＋unpack融合＋非同期検査を固定し、LPA off／on／復帰を比較。LPA追加分は2K／8Kの1出力で約15%／19%、128出力で約9%／13%。24課題でLPAだけが落ちる回帰は0件。[P18](benchmarks.ja.md#直列併用の評価p18)
- **P22最終併用** 上記にAPC・LPA cut32／tail512／B128を加え、KV各rank 2 GiB。24課題は通常21／LPA24／復帰23、held-out 8文書は7／8／8。[P22併用](benchmarks.ja.md#apclpamtp融合非同期検査の併用p22)
- **保持を含む最終回帰** さらに`dense`保持を加えた最終imageで、2K／8K（H=0）と16K（H=4,608）の128出力を3回測定。保持A/B/Aとは反復数・比較対象が異なるため、性能採用の根拠ではなく回帰確認です。[最終回帰](benchmarks.ja.md#保持候補を含む最終併用の回帰)

## 用途別の構成

全機能有効が常に最速ではありません。同一入力を繰り返す条件では、MTP併用がMTPなしより128出力で約20.6%（8,192入力）／28.4%（16,320入力）遅くなり、MTPの復帰境界によって再利用できる入力が短くなっていました。KV予算・kernel設定も異なる構成全体の比較であり、MTP単独のコスト推定ではありません。[同一入力再利用の差](benchmarks.ja.md#同一入力を再利用する場合の差)

| 用途 | 構成 | 根拠 | 注意 |
|---|---|---|---|
| 生成重視・直列 | MTP k=3＋unpack融合＋非同期検査＋LPA cut32／tail512、APCは任意 | P18／P22最終併用 | 1系列・eager。企業検収・ハーネスは未了 |
| prefix再利用重視 | APC＋LPA（P22、B=128）＋`dense`保持、MTPなし | 同一入力再利用・途中編集のA/B/A | 通常primingで共有cacheを育てる。cold処理は小幅悪化 |
| throughput | 2系列、LPAなし、chunk 512（停止延長を許容するなら1024） | P13／P11 | LPAは1系列限定。4系列以上は未検収 |
| 基準・切り分け | 全てoff、eager、1系列 | 基準ベンチ | 常用検収未了のベータ版 |

いずれも[起動設定TOML](startup-configuration.ja.md)の`[mtp]`・`[lpa]`・`[cache]`・`[context]`・`[runtime]`で切り替え、対応imageのmarkerが必要です。

## 読み替えの禁止事項

- 別実験の改善率を足さない・掛けない。imageや入力・KV予算は実験ごとに異なる
- 部品や小層fixtureの合格を全モデルへ繰り上げない
- 機能受入／性能採用／既定on・off／併用検収を分けて読む
- 候補順序の正規化は全モデル実測のimageに含まれない。patched imageには4層fixtureの回帰しかなく、再ビルドしたruntimeには全モデルの回帰が要る
- FreedomBenchの満点、限定課題の正答、fixtureの一致は、一般品質や本番信頼性の証明ではない

## 次の候補

- P06 Graphs：LPAのeager制約を解く検証と全モデル受入
- Euryale：本リポジトリ外の投機draft研究（非公開・独立プロジェクト）。標準MTP k=3との同条件比較で品質・性能・メモリ・復旧のゲートを通した場合に限り、既定の投機経路を置き換える候補。全モデルの教師採取・学習は未着手
- P09 FP8／BF16 KVのA/B、P20 indexer workspace、128K以上のcontext、複数系列×LPA
