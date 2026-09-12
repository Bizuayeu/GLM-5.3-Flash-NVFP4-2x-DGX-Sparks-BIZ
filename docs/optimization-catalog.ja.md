# 企業利用に向けた性能・品質施策台帳

[English](optimization-catalog.md) · [プロジェクトの目的](../README.ja.md#企業利用に向けた取り組み) · [性能調査の手順](performance-investigation.md)

次のGLMでも「何を狙い、何を測り、何を採用しなかったか」を比較するための台帳です。施策の一覧と再評価条件を本書に集約し、実測値・詳細な試験仕様・起動設定は下記の正典へつなぎます。全項目が実装済み、あるいは有効化可能という意味ではありません。

## 今回の基準点と文書の役割

**初期基準：2026-09-12（Asia/Tokyo）、文書commit `9997ccc`。下表は2026-09-13までの追加検証を反映します。** 進行中の結合試験や未コミットの試作は、成功結果へ繰り上げません。次版との比較には参照した文書commit・run IDも残します。

| 正典 | 所有する情報 |
|---|---|
| [README](../README.ja.md) | 企業利用に向けた目的と入口 |
| 本台帳（日英の対応版） | 施策ID、狙い、基準日時点の到達点、再評価条件 |
| [性能調査](performance-investigation.md) | launch・同期・タスクbatch・TP/PPの計測手順 |
| [起動設定](startup-configuration.ja.md)／[runtime lock](../config/runtime.lock.json) | 設定の意味、対応範囲、固定資産。候補値は検収済み上限ではない |
| [基準ベンチ](benchmarks.ja.md)／[MTP](speculative-decoding.ja.md)／[LPA](lpa.ja.md)／[CUDA・indexer実測](component-validation.md) | 各実験の数値、条件、結果の限界 |
| [FreedomBench](freedombench.ja.md)／[検証範囲](validation.md)／[ハーネス](harnesses.ja.md) | 品質・機能・企業業務への受け入れ仕様 |
| 非公開の `records/<run-id>/` | 生応答、trace、全反復、失敗、実機固有情報。公開文書には検証した要約を置く |

最初の基準はGB10×2、TP=2、Marlin W4A16、FP8 KV、候補を保持するNoPE参照attention、eager、同時1系列、MTP/LPA/fusion/APCなしです。**各後続実験でイメージ・入力・資源条件が同一とは限らず、別実験の改善率を足したり掛けたりしません。** 元のW4A4 recipeと、実測したW4A16の算術も区別します。

現在の整理は、MTPが主にdecode、LPAが長文prefill、現行CUDA融合が主にprefillへ効いた、というものです。CUDA融合はKV復元部分の第一段階であり、NoPE attention本体の融合やGraphs対応の完了を意味しません。CSA2は部品を保持して実モデル適用を保留しています。

**P06の追加観測（2026-09-12、上記基準点以後）：** 単体Graphの起動設定と候補イメージを実装。融合・LPA・MTPなしの小層fixtureでcapture/replayを確認しましたが、2K入力の生成tokenがeagerと分岐したため、数値受入と全モデルへの進行を保留しました。既定はeagerのままです。[観測・固定資産・再開条件](component-validation.md#independent-decode-graph-fixture)。下表の基準点の結果とは区別します。

**P13の追加観測：** 1→2→1系列の独立比較で、短文／2K入力・64出力tokenの同時2要求についてthroughput改善と限定タスクの一致を確認しました。別の容量試験では16,320入力＋64出力を2要求同時に完了しました。**性能・品質・容量の検収範囲は分け、設定上限を変えてもKVが自動増額されるとは扱いません。** [実測と容量条件](benchmarks.ja.md#標準batchingの独立評価)を併記します。

**P05の判断：不採用。** 候補幅2176のdecode非対応、数値検査未達、prefill限定試験の形状拒否を踏まえ、今回のbackend選定は終了します。候補を削って合わせる変更は行っていません。再評価は上流の対応変更時とし、他施策へ進みます。[部品検査](component-validation.md#direct-padded-native-attention-probe)。

## 性能施策一覧

「期待効果」は検証する仮説です。「実測あり」も、その記録の条件での観測を指し、本番・品質・併用構成の検収を兼ねません。IDは次版でも維持し、結果が悪い施策も理由とともに残します。

| 施策名 | 内容 | 期待される効果・見る指標 | 今回の到達点／検証先 |
|---|---|---|---|
| P01 標準MTP・先読み3token | checkpoint同梱のBF16 draftを使い、外部draftモデルを追加しない。off／k=1／k=3を比較 | 有効tokenあたりのtarget step削減、decode改善。採択長・draft/verify時間・追加メモリも測る | **実測に基づきk=3を選定。** 有効化はopt-in。他の深さ・複数系列は未検収。[MTP](speculative-decoding.ja.md) |
| P02 LPA | 後段のattention入力を予測し、過去tokenのMLPと不要queryを省略。末尾は通常計算、生成時は全層実行 | 長文prefill・TTFT短縮。decode高速化は狙わず、近似された状態による品質差を測る | **実測あり・実験用。** 長文課題の限定検証。一般品質・結合構成は別ゲート。[LPA](lpa.ja.md) |
| P03 CUDA fusion：KV復元 | FP8 MLA cacheのコピー・FP32変換・scale乗算をTritonで融合 | KV復元のlaunch・中間tensor削減、主にprefill短縮 | **部品一致・全モデルA/B/A実測あり。** 8Kの約17%短縮は出力1tokenの対照。短文decodeはほぼ不変、既定off。[CUDA実測](component-validation.md) |
| P04 NoPE attention本体のカーネル化・融合 | Pythonのqueryループと多段演算を、候補集合・scale・因果maskを保って置換 | prefill／decodeのattention処理時間とlaunch数を削減 | **今回のquery chunk拡大・FP32融合案は不採用。** launchと一時メモリ削減だけでは高速化せず、serving未接続。別戦略の根拠が得られたら再評価。[部品実測](component-validation.md#nope-attention-fusion-and-query-batching-p04) |
| P05 SM121 attention backend選定 | FlashInfer/TRT系・Triton等の候補について、固定GLMのNoPE／sparse MLA形状・cache契約への対応を先に検査し、同一負荷でA/B | 安定した起動・warmup、対応kernelでの実効性能。意図しないfallbackを検出 | **不採用。** 数値条件未達・必要候補幅の非対応。上流対応変更時に再評価。[部品実測](component-validation.md#direct-padded-native-attention-probe) |
| P06 CUDA Graphs | hostへ戻る判定・動的処理・buffer寿命を整理し、対応shapeでcapture/replay | CPUのstep/launch overhead低減、特に定常decode改善 | **全モデル未検収。** LPAのeager制約を解く検証が必要。単体capture成功とサーバー全体対応は別。[性能調査](performance-investigation.md#kernel-launches-and-synchronization) |
| P07 launch・同期の計測 | 両rankのkernel／CUDA launch API／NCCL／host syncを別集計。prefill対照との差分でtokenあたりの数を推定 | 律速を特定し、削減前後を再現可能にする | **実測済み。** P03でlaunchは減ったがNCCL回数は不変。イベント時間の和を壁時計時間としない。[CUDA実測](component-validation.md) |
| P08 sync／copy／変換の追加削減 | P07で残る転送・dtype変換・host往復を特定。反復する重み変換を観測した場合だけload時の並べ替えも比較 | 余剰レイテンシ・UMAメモリ交通・一時bufferの削減 | **非同期index検査を独立opt-inとして受入。** 128出力で小幅改善、同期・copy各22回/token削減。GPU kernelは11回増加。既定auto、併用は未検収。[実測](benchmarks.ja.md#cpu同期削減の独立評価p08) |
| P09 FP8 KVの独立評価 | 重み精度を固定し、対応backendのFP8／BF16 KVを比較。量子化scale、pool容量、読み出し精度を検査 | cache容量削減による長文・同時数拡大。速度の方向と品質は実測で決める | **FP8経路は使用済み、dtype間A/Bは未了。** 現行起動系はFP8固定。BF16比較にはcache形式・runtime対応が必要。[起動設定](startup-configuration.ja.md) |
| P10 UMA・メモリ運用 | KV pin、memory utilization、host空き、コンテナ上限、swap・cache状態を再現条件として固定 | OOM・swap由来の遅延を抑え、測定再現性と収容限界を把握 | **ガード・設定あり、系統的最適化は未了。** `drop_caches`は必要なcold-load比較に限定し、定常推論の高速化手段と混同しない。[起動設定](startup-configuration.ja.md)／[基準ベンチ](benchmarks.ja.md) |
| P11 Prefill chunk × Kpool／indexer | chunk予算を変え、pool・tail・cache境界とその前後で通常／分割処理を比較 | 長文prefillと混合負荷のITLを改善し、状態・候補の取り扱いを維持 | **128は不採用、1024はthroughput候補。** 2K全体出力改善と最長ITL悪化を併記。既定512、1024の16K×2容量は確認済み。[実測](benchmarks.ja.md#prefill-chunk-の独立評価p11) |
| P12 同時ストリーム計測 | client×1／×2／×4とserverのactive seq数を別記し、TTFT・queue・個別ITL・aggregateを同じ表にする | 単発と並列のtradeoff、待ち行列、長文投入による停止を検出 | **待ち行列と実batchを区別して測定済み。** P13でactive2と複数decode行を確認。[実測](benchmarks.ja.md#標準batchingの独立評価) |
| P13 標準batching | LPAなしのseqs=2→4で実batch重複を確認。逐語決定性とタスク品質を分けて検収 | aggregate throughput向上と、個別レイテンシ・メモリの増減把握 | **限定throughput用途の2系列を受入。** 内容・tool確認と16K×2容量確認は別記。既定1系列、4系列・併用は未検収。[実測](benchmarks.ja.md#標準batchingの独立評価) |
| P14 同種タスクbatching | コード／翻訳／要約の同種groupと混在groupを、長さ・同時数・投入量を揃えて比較 | expert重複や投機採択が変わり、aggregateが改善する可能性 | **今回の用途では不採用。** 小規模な投入順比較の改善は約0.7〜0.9%。expert経路・課題完遂品質は未検収。標準batchingを維持。[実測](benchmarks.ja.md#同種タスクの投入順比較p14) |
| P15 コンテキスト長sweep | 入力長とKV予算を独立に変える。8K→32K→128K→262,144は検討する測定点であり、事前に容量と対応を確認 | TTFT・decode・memoryが悪化する地点と運用可能範囲を可視化 | **系統的sweep未了。** 設定可能な長さと実要求の検収を分ける。実測範囲から段階拡張。[起動設定](startup-configuration.ja.md)／[基準ベンチ](benchmarks.ja.md) |
| P16 CSA2：候補Reuse／Reindex | 層間候補の再利用・限定再採点・shared poolを比較。各層のKVと必須cache更新は保持 | indexerの削減可能な計算を減らす仮説。候補coverageと全体時間で判断 | **部品検証・観測済み、serving適用は保留。** 今回はop時間割合が小さく、8K相当のshared-pool部品も遅い。大きい文脈で全コスト・coverage・品質が成立した場合のみ再評価。[CSA2詳細](indexer-reuse.md)／[実測](component-validation.md#indexer-observation) |
| P17 TP=2／PP=2 | PPの中間tensor・mHC post/comb転送を実装・fixture検証後、同じ精度と負荷で比較 | 通信待ち削減の可能性と、stageの直列化・不均衡による損失を測る | **今回の生成負荷では不採用・既定TP2。** Prefillは改善、decodeは低下。速度A/B/A・限定品質・切断は完了。PPのprofiler再開始で障害が出たため、PP容量・decode traceは未完了。[実測](benchmarks.ja.md#tp2pp2の独立評価p17) |
| P18 MTP＋LPA＋fusion、必要ならGraphs | 単独と組合せを同じ資産・課題で比較。復帰対照を含め、実際のLPA作動・投機採択・captureを記録 | 改善の相互作用を測り、品質・メモリ・復旧の回帰を検出 | **結合検証中、確定結果なし。** 三者併用の効果を単体の改善率から推定しない。Graphsは対応後の別条件。[CUDA実測の残工程](component-validation.md#reproduction-and-remaining-integration) |
| P19 Prefix caching（APC） | 同じsystem/tools/履歴のcold／warmと異なるprefixを比較。hybrid state・境界・混線を検査 | 繰り返す会話のTTFT・再prefillを削減 | **未検収、現LPAとは非対応。** coldの高速化とcache hitの効果を分離。[LPA使用範囲](lpa.ja.md#使用範囲) |
| P20 Indexer workspace適正化 | 実shape・chunk・MTP深さ別の最大必要量を測り、過剰予約がある場合に限定して縮小 | host/KVの余裕を増やし、不要な割当を抑える | **条件付き候補、独立した効果の検収なし。** P16の候補削減とは別施策。上流で解消済みなら追加patch不要。[性能調査](performance-investigation.md) |

**追加施策（2026-09-13）：**

| 施策名 | 内容 | 期待される効果・見る指標 | 現在地／検証先 |
|---|---|---|---|
| P21 Expert Parallel | TP=2・DP=1・2系列・各rank固定KV予算を維持し、Expert層の分割だけをTPからEPへ変更する独立実験 | Expert計算効率と全体throughput。追加メモリ、通信、個別TTFT/ITL、品質も比較 | **今回の性能施策として不採用・既定off。** 全モデルA/B/Aの全4ケースで両off対照より遅い。限定品質・切断復帰・16K×2容量は通過。別負荷・併用は別検収。[実測](benchmarks.ja.md#expert-parallel-の独立評価p21) |

P12は測定軸、P13はscheduler設定、P14は投入順序の実験です。同じ改善を三つに数えません。また、MTPは単一系列でも複数の投機行をまとめて検証できるため、GEMM寄りの仕事を得る前提が必ずしも`max_num_seqs > 1`ではありません。

## 企業利用の品質・採用ゲート

性能調整は、資料に忠実な応答、ツールの正しさ、ライセンスの扱いやすさと併せて判断します。特定の思想へモデルを寄せることや、安全上適切な拒否を一律に取り除くことを目的にしません。

| 施策名 | 内容 | 期待される効果／示せる証拠 | 今回の到達点／正典 |
|---|---|---|---|
| E01 ライセンス・出所の選択 | モデル・独自コード・採用patch・container・harnessを分け、採用版と通知を固定。MIT/Apache中心の方針を維持 | 企業が利用・改造・配布範囲を確認しやすくする | 方針・台帳あり。完成イメージ全体やharnessまで同一licenseとは扱わない。[ライセンス整理](licensing.ja.md)／[通知](../THIRD_PARTY_NOTICES.md) |
| E02 FreedomBench・政治的文脈 | 英語原版、日本語、長い業務資料を分け、通常／MTP／LPA／併用を比較。拒否・誤答・形式・通信失敗を分離 | 対象設問での回答傾向、政治的主張の挿入や資料への不忠実、最適化による変化を可視化 | ローカル評価器と必須仕様あり。原版全問・全構成と独自拡張の検収は未了。[FreedomBench FB-01〜05](freedombench.ja.md) |
| E03 業務機能・運用の検収 | text、tool往復、SSE、cancel、要求間分離、ZCode／Claude Code、長時間負荷、両rank復旧を検査 | ベンチで速い構成が業務でも使えるかを判定し、戻せるprofileを残す | 基礎API等の限定実測あり。harness・本番信頼性は未検収。[検証範囲](validation.md)／[ハーネス](harnesses.ja.md)／[運用](operations.md) |

FreedomBenchは、著者の正答に対する一致を測る、対象範囲の限られた試験です。**高得点でも「思想の偏りが一切ない」とは証明できません。** 企業向けには対象・条件・結果・未検証範囲を示し、採用側が用途に即して判断できる証拠を提供します。日本語や長文の独自拡張を公式スコアへ混ぜず、LPAが作動しなかった短文結果を近似品質の証拠に数えません。融合やGraphsを加えた場合も、その実効設定を明記して再評価します。設問・採点法の詳細はFreedomBench文書だけで管理します。

## 次版でも使う比較の進め方

1. **新しい素の基準を作る。** model/revision、license、runtime、SM対応、量子化・cache形式を確認。旧patchを無条件で移植せず、上流が既に解決した項目は「不要」と判定する。
2. **部品から実モデルへ。** exact算術の部品検査と近似の品質検査を区別し、fixture状態、全モデルの同一負荷A/B/Aへ進む。trace採取と速度本測定は分離する。
3. **単体の採否を決め、最後に併用。** まず各施策を独立比較し、採用・保留・不採用まで判断する。依存する部品は基準側にも固定し、比較で変える施策は一つにする。採用候補が揃ってからP18で運用予定の組合せと必要なoff対照を測る。部品を追加するたびに全併用試験を繰り返さず、既存の併用結果は予備結果として保持する。P13/P14、長文拡張、PPを同時に変更しない。
4. **機能受入と性能採用を分ける。** Graphs等の基盤整備は、対応範囲で正しさ・資源上限・復旧性が確認できれば、速度改善が小さくても機能として受け入れられる。高速化としての採用には再現する改善が必要。既定の有効化と最終併用検収も別に判断する。逐語一致、内容・tool正答、拒否、未完了を分け、E02/E03を通す前に企業利用の検収済みとしない。閾値・課題・用途は結果を見る前に固定する。
5. **採用しなかった理由も残す。** 誤差内・遅い・資源不足・品質未達・上流で不要のいずれかと再評価条件を記録。旧版の結果は新しい値で上書きせず、Git履歴と新runで比較する。

### 機能受入と既定設定

単体の判断には、**機能受入（検証範囲付き）／性能採用／既定on・off／併用未検収・検収済み**を別々に記録します。「機能受入・性能差は誤差内・既定off・併用未検収」も有効な結論です。部品や小層fixtureの合格を全モデル対応へ繰り上げません。

P06では、Graph用のメモリ保持、起動時capture時間、対応shapeとbuffer寿命、異常時の復旧を確認します。Graphは副作用ゼロの設定ではありません。LPAのprefill経路とMTPの投機経路はP18で確認し、速度だけを理由に全経路をGraph化しません。PyTorchの[CUDA Graph制約とメモリ管理](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)を参照。

### 比較記録の共通項目

次版・次実験では、以下の行を実施レポートに使用します。新たな設定ファイル形式や自動採点器を定義するものではありません。

| 項目 | 残す内容 |
|---|---|
| 対象と判断 | 施策ID／旧版・新版／狙う用途・指標／事前の採否基準／採用・保留・不採用と理由。機能受入の範囲・性能採用・既定設定・併用検収を別記 |
| 固定資産 | model revision、source commit、両rank image、runtime/backend、精度、template/tokenizer、projector/view、license台帳への参照 |
| 実効設定 | profile fingerprint、TP/PP、active seq、chunk、KV dtype/容量、context、MTP/LPA/fusion/graph/APCの実効状態 |
| 負荷 | corpus/課題/設問hashとsplit、要求ID、実入力・出力token、effort/sampling、投入順序・同時数・実batch重複、他負荷 |
| 時間 | cold起動、warmup、TTFT、prefill対照、個別TPOT/ITL・queue/E2E、aggregate。反復数と分布を保存し、少数試験の中央値をSLAとしない |
| 内訳と資源 | rank別kernel/launch/NCCL/sync/copy、MTP採択・step数、host空き最小・container peak・swap・温度。重複区間を加算しない |
| 品質と完全性 | 予定／完了／欠落／エラー、拒否、打切り、内容・tool・資料忠実性、数値・状態診断、復帰対照。未検収項目も明記 |
| 再現と出典 | A/B/Aの全試行、profiler offの実時間、raw run ID、参照文書commit、公開要約へのリンク、復旧設定 |

入力長・tokenizerが変わる新版では、同じ文章による業務比較と、同じtoken予算による処理量比較を分けます。公開文書は生ログを複写せず、この台帳から結果の正典へリンクする形で更新します。
