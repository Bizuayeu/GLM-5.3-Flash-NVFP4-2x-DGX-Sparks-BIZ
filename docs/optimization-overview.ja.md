# 推論最適化の全体像

[English](optimization-overview.md) · [施策台帳](optimization-catalog.ja.md) · [文書一覧](README.ja.md)

どの施策が推論のどの段階に効き、それぞれがいまどこにあり、用途によってどの構成を選ぶかを一枚で示す読み物です。施策IDと採否の正典は[施策台帳](optimization-catalog.ja.md)、数値の正典はリンク先の各実測文書です。

## 基準構成

基準は[施策台帳の基準点](optimization-catalog.ja.md#今回の基準点と文書の役割)と同じで、context 16K・KV各rank 1 GiBで測定しています（[初期の測定条件](benchmarks.ja.md#初期の測定条件)）。32Kまでの独立容量評価は[32K sweep](benchmarks.ja.md#32kまでの独立コンテキスト評価p15)、従来の200K併用結果は[旧profileの記録](benchmarks.ja.md#旧profileの記録)の[リリース候補の測定](benchmarks.ja.md#リリース候補の測定)を参照してください。256K・KV各3 GiBのテキスト専用構成は[256Kの実入力確認](benchmarks.ja.md#256kでの実入力確認)、現在の配布既定（画像入力あり）は[画像入力](vision.ja.md)を参照してください。

## 段階別の位置づけ

```mermaid
flowchart LR
    R[要求] --> A[prefix復元<br/>P19 APC・checkpoint保持・P22 判定・P25 page重複排除]
    A --> P[prefill<br/>P11 chunk・P05 FA2 prefill・P02 LPA・P03 unpack融合]
    P --> D[decode<br/>P01 MTP k=3・P08 非同期index検査・P23 再パックした重み・P29 読み手のspin]
    D --> O[出力]
    S[並列・throughput<br/>P13 公開した任意設定で2系列・P28 3台のTP=3・P21 EP 不採用・P17 PP 不採用] -.- P
    S -.- D
    B[attention backend・indexer<br/>P04 不採用・P16 中止・候補順序の正規化] -.- P
    B -.- D
```

## 施策と現在地

「状態」は台帳の判断を短く言ったもので、日付・理由・再評価条件は台帳の行にあります。「既定」は配布既定 `examples/server.example.toml` の値です（[起動設定](server-configuration.ja.md#配布用の既定設定)）。3ノードのテンプレート `examples/server.tp3.example.toml` も、TPの幅（P17・P28）を除き、以下の施策ではすべて同じ値です。公開した任意設定との差は[公開した任意設定と配布既定の差](server-configuration.ja.md#公開した任意設定と配布既定の差)にあります。機能受入・性能採用・既定値・併用検収は別々に判断されており、採用でも既定onとは限りません。用途別の有効化は[用途別の構成](#用途別の構成)を参照してください。

### prefix復元（繰り返す会話）

| 施策 | 仕組み | 状態 | 既定 | 正典 |
|---|---|---|---|---|
| P19 APC | 通常計算由来の状態だけを共有cacheに登録し、同一prefixのprefillを省略する | 実測した直列・長文prefix再利用の用途で受入（実験用）。cold要求は小幅に遅い | on（`cache.prefix_caching=true`） | [P19](benchmarks.ja.md#全モデルのprefix-caching独立評価p19)／[正しさの関門](validation.ja.md#prefix-cacheの正しさの関門) |
| checkpoint保持（`cache.prefix_cache_retention_interval`） | KDA checkpointをscheduler blockごとに保持し、途中編集・分岐後に復元できるHを伸ばす | 通常priming済み・直列の途中編集用途で採用。実測したarmは標準の間隔4,352。`dense`は実測した整列配置で同じ標準KDA maskになるが、最終併用の検収は別 | `dense`（キー省略時のruntime既定は0） | [保持A/B/A](benchmarks.ja.md#apcの履歴保持の基準検査)／[契約](launch-safety.ja.md#apcの履歴検証) |
| P22 APC優先LPA | 復元したHの先を、残余R＝N−T−Hが閾値Bを超えるときだけ近似する。近似状態は要求内に閉じる | 完了（校正・限定品質・最終併用・held-out）。B=128は保守的な候補閾値で、普遍的な損益分岐定数ではない | APCはon、LPAはoff（有効化時に `lpa.break_even_tokens=128` を適用） | [設計](apc-lpa-design.ja.md)／[校正](benchmarks.ja.md#apc優先lpaの損益分岐計測p22) |
| P25 page重複排除 | 既にcache済みのblockを持つhashで満杯のblockを登録せず、MTP下で再送した履歴が古い履歴を追い出さないようにする | 採用（1.9.0） | off。公開した任意設定ではon（`runtime.prefix_page_dedup`） | [1.9.0](benchmarks.ja.md#190での測定) |

### prefill

| 施策 | 仕組み | 状態 | 既定 | 正典 |
|---|---|---|---|---|
| P02 LPA | cut=32（0始まり）以降の層で過去tokenのMLPを省き、末尾512 tokenは通常計算。生成時は全層 | 実測あり（実験用。一般品質は別ゲート）。長文照合・tool往復は合格 | off。バッチ用opt-in（`lpa.enabled=true`）——近似要求は共有prefixを公開しないため。FA2 prefillと排他 | [LPA](lpa.ja.md) |
| P03 unpack融合 | FP8 MLA cacheの復元（コピー・FP32変換・scale乗算）をTriton 1 kernelに | 実測あり（部品一致・全モデルA/B/A。受入済みのP18併用の範囲で使用） | on（`cache.fused_unpack=true`） | [部品実測](component-validation.ja.md) |
| P11 prefill chunk | schedulerのtoken予算を、2系列と、1系列の200K profileで比較 | 既定2048、128は不採用。2系列ではchunkが長いほど最長停止が延びる | 2048（`context.max_num_batched_tokens`） | [P11](benchmarks.ja.md#prefill-chunk-の独立評価p11)／[200K](benchmarks.ja.md#200k画像profileでのchunk予算2026-09-17) |

### decode

| 施策 | 仕組み | 状態 | 既定 | 正典 |
|---|---|---|---|---|
| P01 MTP k=3 | checkpoint同梱のBF16 draftを別メタデータviewで読み、3 token先読み。外部draftモデルなし | 深さ1〜5を測り、両方のcheckpointでk=3を選定。採択の履歴による深さ、draftの確信度の関門、draft側の設定二つは測って不採用 | on（`mtp.enabled=true`、`num_speculative_tokens=3`） | [両方のcheckpointで深さ3](speculative-decoding.ja.md#両方のcheckpointで深さ32026-09-21)／[固定の深さの先](speculative-decoding.ja.md#固定の深さの先2026-09-21) |
| P08 非同期index検査 | 範囲検査を省かずGPU assertへ移し、hostとの同期・copyを減らす代わりにGPU kernelが少し増える | 受入（独立opt-in） | async（`runtime.index_checks`） | [P08](benchmarks.ja.md#cpu同期削減の独立評価p08) |
| P06 CUDA Graphs | decodeのみcapture／replay | 不採用（2026-09-21）：全モデルでeagerより1 stepあたり遅い。選択肢としては残し、後のruntimeで測り直す | off（`runtime.decode_graphs=false`） | [Graph fixture](component-validation.ja.md#decode-graphのfixture独立評価)／[全モデル](benchmarks.ja.md#全モデルでのdecode-graphs) |
| P23 再パックした重み | attention projectionと `lm_head` をW4A16 NVFP4に再パック（route l）し、`runtime.derived_checkpoint` で配信 | 公開した任意設定として採用。losslessではないので配布既定には入れない | off。公開した任意設定ではon | [台帳](optimization-catalog.ja.md#性能施策一覧)／[配信profile](benchmarks.ja.md#基準の2台の配信profileattentionと-lm_head-の再パック深さ3) |
| P29 共有メモリの読み手のspin | headのEngineCoreがworker 0の返答を読んだ後、zmqのpollで眠る前のspinを1秒から2 msに | すべてのテンプレートで採用（2026-10-02）、headの温度のため：TP=2の対でheadのCPUとSoCの温度が下がり、countingのdecodeは少し遅い。TP=3は延長での適用で未測定 | 0.002（`runtime.shm_spin_seconds`） | [1.25.0](benchmarks.ja.md#1250での測定) |

### 並列・throughput

| 施策 | 仕組み | 状態 | 既定 | 正典 |
|---|---|---|---|---|
| P13 標準batching | `max_num_seqs=2`で実batch重複を作る。LPAは1系列限定 | 公開した任意設定は2系列を配信（2026-09-23受入、1要求あたり約200K tokenまで）。どんな負荷でもcompletionが反復するのは1系列のときだけ（[同時実行の範囲](validation.ja.md#同時実行の範囲)） | 1系列（`context.max_num_seqs=1`）。公開した任意設定では2 | [P13](benchmarks.ja.md#標準batchingの独立評価) |
| P28 TP=3 | スイッチなしのQSFPリングで3台。head・expertの幅・語彙を読み込み時に0で詰め、3で割れるようにする | 実施。2026-10-01に通常運用として受入（[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)）：2台より長く、多くの要求を同時に持てる | 2台はTP=2。3台は `examples/server.tp3.example.toml` | [1.24.0](benchmarks.ja.md#1240での測定)／[3ノード](server-configuration.ja.md#3ノード) |
| P21 Expert Parallel | Expert層の分割だけをTPからEPへ | 不採用（実測したthroughput負荷） | off（`runtime.expert_parallel=false`） | [P21](benchmarks.ja.md#expert-parallel-の独立評価p21) |
| P17 TP2／PP2 | 同じ2台をTP1×PP2に | 不採用（実測した生成負荷） | TP2（`runtime.pipeline_parallel_size=1`） | [P17](benchmarks.ja.md#tp2pp2の独立評価p17) |
| P14 同種タスクbatching | 投入順を同種でまとめる | 不採用（この負荷） | —（設定項目なし） | [P14](benchmarks.ja.md#同種タスクの投入順比較p14) |

### attention backend・indexer

| 施策 | 仕組み | 状態 | 既定 | 正典 |
|---|---|---|---|---|
| P04 NoPE attention融合 | Pythonのqueryループと多段演算の置換 | 不採用（launch削減だけでは速くならず） | —（serving未接続） | [P04](component-validation.ja.md#nope-attentionの融合とquery-batchingp04) |
| P05 FA2 prefill（SM121 backend選定） | prefillの大きさのNoPE attentionを、BF16に展開した行でFlashInferのSM90 FA2 wrapperに通す。SM120の直接差し替えは不採用 | prefillに採用（1.6.0） | on（`runtime.fa2_attention`）。1系列のdecodeは参照経路、LPAと排他 | [1.6.0](benchmarks.ja.md#160でのprefillとdecode)／[SM90 FA2](component-validation.ja.md#sm90-fa2-mla-wrapperの試験)／[SM120の試験](component-validation.ja.md#padding付きnative-attentionの直接試験) |
| P16 CSA2 | 層間の候補再利用・限定再採点 | 第一の門で中止（2026-09-21）：indexerがprefillに占める割合が小さすぎる。部品は保持 | —（未統合） | [CSA2](indexer-reuse.ja.md) |
| 再現性と正しさの修正 | imageに：sparse MLA候補順序の正規化と、samplerの語彙への上限。三つのスイッチで：expert内のtoken順を一つに、indexerのtop-kの同点をpool indexで決める、Inductorの設定を計時なしで選ぶ | 台帳外：同一要求を反復させ、サンプルしたidを語彙内に保つ修正。上記の初期比較は変更前 | on（参照image。スイッチはすべてのテンプレートで） | [候補順序](candidate-order.ja.md)／[再現性のスイッチ](server-configuration.ja.md#再現性のスイッチ)／[イメージの契約](server-configuration.ja.md#現行イメージの契約) |

### 運用（性能施策ではない）

認証クライアント・allocator伝達・レール検査・両rank切替は運用の契約であり、高速化ではありません。[施策台帳](optimization-catalog.ja.md)が既存のIDの下に整理し、[起動契約](launch-safety.ja.md)が所有します。

## 直列併用の実測

併用状態は併用状態として実測しています。

- **P18** MTP3＋unpack融合＋非同期検査を固定し、LPA off／on／復帰を比較。LPA追加分は2K／8Kの1出力で約15%／19%、128出力で約9%／13%。24課題でLPAだけが落ちる回帰は0件。[P18](benchmarks.ja.md#直列併用の評価p18)
- **P22最終併用** 上記にAPC・LPA cut32／tail512／B128を加え、KV各rank 2 GiB。24課題は通常21／LPA24／復帰23、held-out 8文書は7／8／8。[P22併用](benchmarks.ja.md#apclpamtp融合非同期検査の併用p22)
- **保持を含む最終回帰** さらに`dense`保持を加えた最終imageで、2K／8K（H=0）と16K（H=4,608）の128出力を3回測定。保持A/B/Aとは反復数・比較対象が異なるため、性能採用の根拠ではなく回帰確認です。[最終回帰](benchmarks.ja.md#保持候補を含む最終併用の回帰)

## 用途別の構成

全機能有効が常に最速ではありません。同一入力を繰り返す条件では、MTP併用がMTPなしより128出力で遅くなり、MTPの復帰境界によって再利用できる入力が短くなっていました。KV予算・kernel設定も異なる構成全体の比較であり、MTP単独のコスト推定ではありません。[同一入力再利用の差](benchmarks.ja.md#同一入力を再利用する場合の差)

| 用途 | 構成 | 根拠 | 注意 |
|---|---|---|---|
| 生成重視・直列（コード。配布既定） | 固定のcheckpoint、MTP k=3、unpack融合、非同期検査、FA2 prefill、APC、再現性のスイッチ、読み手のspin、1系列。effortを指定しない要求は `high` | [配布用の既定設定](server-configuration.ja.md#配布用の既定設定) | 1系列で通常運用として受入（[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)）。固定の重みに対してlossless |
| 日本語散文 | 公開した任意設定（NVFP4 BIZ AXL） | [公開した任意設定と配布既定の差](server-configuration.ja.md#公開した任意設定と配布既定の差)／[配信profile](benchmarks.ja.md#基準の2台の配信profileattentionと-lm_head-の再パック深さ3) | losslessではない。その費用はREADMEの比較にある（[確認した範囲](../README.ja.md#確認した範囲)）。prefix cacheの関門は判定不能だった（[関門](validation.ja.md#prefix-cacheの正しさの関門)） |
| 長い入力のバッチprefill | MTP k=3＋unpack融合＋非同期検査＋LPA cut32／tail512、FA2 prefillはoff、APCは任意 | P18／P22最終併用 | LPAはバッチ用opt-inで近似、FA2 prefillと排他、共有prefixの再利用を失う |
| prefix再利用重視 | APC＋LPA（P22、B=128）＋`dense`保持、MTPなし | 同一入力再利用・途中編集のA/B/A | 通常primingで共有cacheを育てる。cold処理は小幅悪化 |
| throughput | 2系列：公開した任意設定のexample | [同時実行の範囲](validation.ja.md#同時実行の範囲) | 要求が重なると処理量が増える。completionは同時に走る要求に依る。LPAは1系列限定。3系列以上は3台が要る（次の行） |
| 全長の文脈、多くの同時要求 | 3台のTP=3（`examples/server.tp3.example.toml`）、どちらのcheckpointも | [P28](optimization-catalog.ja.md#性能施策一覧)／[同時実行の範囲](validation.ja.md#同時実行の範囲) | 通常運用として受入（[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)）。テンプレートは各rank 3 GiBのKV・1系列なので、実測した容量には大きな予算が要る（[3ノード](server-configuration.ja.md#3ノード)）。PP2・EP・LPAは2ノード限定のまま |
| 基準・切り分け | 全てoff、eager、1系列 | 基準ベンチ | 比較用。検収した配信profileではない |

いずれも[起動設定TOML](server-configuration.ja.md)の`[mtp]`・`[lpa]`・`[cache]`・`[context]`・`[runtime]`で切り替え、対応imageのmarkerが必要です。

## 性能と容量のQ&A

現在の構成を拡張するときの考え方を整理します。どのテンプレートも要求あたり入出力合計262,144 tokenが上限で、1Mは約100万tokenです。台数ごとに何を検収したかは[同時実行の範囲](validation.ja.md#同時実行の範囲)にあります。

### Q. KVキャッシュを各rankに3 GiB確保すれば、256Kコンテキストを継続して利用できますか？

**KV容量としては、モデル・cache精度・MTP/LPA・並列方式・同時1系列が同じなら、はい**（[256Kの実入力確認](benchmarks.ja.md#256kでの実入力確認)）。APCの履歴・分岐・保持、blockの整列、同時数を変えたら状態と割当を確かめ、KVの収容と無停止運用の保証は分けて判断してください（[KV容量とRAMの条件](server-configuration.ja.md#kv容量とramの条件)）。

### Q. KVキャッシュを各rankに12.5 GiB用意できれば、1Mコンテキストを利用できますか？

**固定の重みで2台なら、ランチャーがrankあたり3 GiBに抑えるのでできません。3台では1Mのpoolで動かしています**（[KV容量とRAMの条件](server-configuration.ja.md#kv容量とramの条件)、[1.24.0での測定](benchmarks.ja.md#1240での測定)）。

### Q. MTPを無効化し、保護用のメモリ余裕を減らせば、1M運用は可能でしょうか？

**3台では不要で、約1M tokenの要求はMTPを有効にしたまま走りました。2台の固定の重みは、MTPの有無にかかわらず3 GiBが上限です。** MTPを無効にすると[draftのメモリ](speculative-decoding.ja.md#k1の実測結果)が空きますが固定量です。保護余裕を減らしても余白が小さくなるだけで、RAMは増えません。

### Q. 1Mコンテキストでは、どの程度の待ち時間を見込む必要がありますか？

**prefix cacheの再利用がなければ最初のtokenまで20分弱。3台・公開した任意設定で1回測りました**（[1.24.0での測定](benchmarks.ja.md#1240での測定)）。約200Kの要求は、配信するどちらのprofileでも約3分です（[主要な測定値](../README.ja.md#確認した範囲)）。TTFTの保証ではなく、長い入力を毎回読み直す対話では実用上の制約になります。

### Q. KVキャッシュを各rankに5 GiBへ増やすと、複数同時実行やEP・PPが有効になる可能性はありますか？

**2台では固定の重みは5 GiBを拒否し、公開した任意設定はすでに6 GiBから2系列を配信しています。長い要求をさらに同時に持つには3台が要ります（P28）。** [EP](benchmarks.ja.md#expert-parallel-の独立評価p21)・[PP](benchmarks.ja.md#tp2pp2の独立評価p17)は測定した負荷で不採用で、KVの増量だけで高速化が決まるわけではありません。

## 読み替えの禁止事項

- 別実験の改善率を足さない・掛けない。imageや入力・KV予算は実験ごとに異なる
- 部品や小層fixtureの合格を全モデルへ繰り上げない
- 機能受入／性能採用／既定on・off／併用検収を分けて読む
- 初期比較のimageと、候補順序の正規化後の併用回帰・リリース候補測定を区別する
- FreedomBenchの満点、限定課題の正答、fixtureの一致は、一般品質や本番信頼性の証明ではない

## 次の候補

候補とその再評価条件は、[施策台帳](optimization-catalog.ja.md#性能施策一覧)で候補・先送りとした行です。次に何をするかとその条件はREADMEの[Next Action](../README.ja.md#next-action)にあります。新しい固定版vLLMへ移るときは、上の施策すべてを再検収します。
