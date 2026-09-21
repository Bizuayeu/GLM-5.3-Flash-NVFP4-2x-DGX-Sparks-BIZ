# 業務利用に向けた性能・品質施策台帳（BIZ）

[English](optimization-catalog.md) · [プロジェクトの目的](../README.ja.md#業務利用に向けた取り組みbiz) · [性能調査の手順](performance-investigation.ja.md)

次のGLMでも「何を狙い、何を測り、何を採用しなかったか」を比較するための台帳です。施策の一覧と再評価条件を本書に集約し、実測値・詳細な試験仕様・起動設定は下記の正典へつなぎます。全項目が実装済み、あるいは有効化可能という意味ではありません。

Mia更新を参考にした認証クライアント・allocator・全HCA検査・両rank切替・APC履歴保持は、P10／P19／P22／E03の拡張です。別の高速化施策として重複計上せず、[起動契約](launch-safety.ja.md)と[履歴保持の実測](benchmarks.ja.md#apcの履歴保持の基準検査)へ集約します。機能受入・性能採用・既定値・未検収の範囲を分けます。

## 今回の基準点と文書の役割

**初期基準：2026-09-12（Asia/Tokyo）、文書commit `9997ccc`。下表は2026-09-13までの追加検証を反映します。** 進行中の結合試験や未コミットの試作は、成功結果へ繰り上げません。次版との比較には参照した文書commit・run IDも残します。

本台帳（日英の対応版）が所有するのは、施策ID、狙い、基準日時点の到達点、再評価条件です。設定・実測値・受け入れ仕様・保管場所をどの文書が所有するかは[文書一覧](README.ja.md#正典の所在)に一度だけ書き（起動設定の候補値は検収済み上限ではありません）、[推論最適化の全体像](optimization-overview.ja.md)は本台帳を推論の段階と用途の軸で読み直したものです。非公開の `records/<run-id>/` には生応答、trace、全反復、失敗、実機固有情報を置き、公開文書には検証した要約を置きます。

最初の基準はGB10×2、TP=2、Marlin W4A16、FP8 KV、候補を保持するNoPE参照attention、eager、同時1系列、MTP/LPA/fusion/APCなしです。**各後続実験でイメージ・入力・資源条件が同一とは限らず、別実験の改善率を足したり掛けたりしません。** 元のW4A4 recipeと、実測したW4A16の算術も区別します。

現在の整理は、MTPが主にdecode、LPAが長文prefill、現行CUDA融合が主にprefillへ効いた、というものです。CUDA融合はKV復元部分の第一段階であり、NoPE attention本体の融合やGraphs対応の完了を意味しません。CSA2は部品を保持して実モデル適用を保留しています。

**P06の追加観測（2026-09-12、上記基準点以後）：** 単体Graphの起動設定と候補イメージを実装。融合・LPA・MTPなしの小層fixtureでcapture/replayを確認しましたが、2K入力の生成tokenがeagerと分岐したため、数値受入と全モデルへの進行を保留しました。既定はeagerのままです。[観測・固定資産・再開条件](component-validation.ja.md#decode-graphのfixture独立評価)。下表の基準点の結果とは区別します。

**P13の追加観測：** 1→2→1系列の独立比較で、短文／2K入力・64出力tokenの同時2要求についてthroughput改善と限定タスクの一致を確認しました。別の容量試験では16,320入力＋64出力を2要求同時に完了しました。**性能・品質・容量の検収範囲は分け、設定上限を変えてもKVが自動増額されるとは扱いません。** [実測と容量条件](benchmarks.ja.md#標準batchingの独立評価)を併記します。

**P05の判断：不採用。** 候補幅2176のdecode非対応、数値検査未達、prefill限定試験の形状拒否を踏まえ、今回のbackend選定は終了します。候補を削って合わせる変更は行っていません。再評価は上流の対応変更時とし、他施策へ進みます。[部品検査](component-validation.ja.md#padding付きnative-attentionの直接試験)。**2026-09-17に再開：** Spark 2台の他レシピとの比較を受け、行の種類別の再検討でもSM120経路は不採用のまま、SM90 FA2 wrapperは部品として合格しました。servingの切り替えは別の判断です（P05の行）。

**テンプレート既定の更新（2026-09-15）：** 配布用の起動テンプレートは `lpa.enabled = false` と `runtime.vision = true` を既定にしました（1.4.0までは204,800 token・KV各rank 2.5 GiB、1.5.0からは262,144 token・3 GiB）。P02／P22の採否は変えていません。近似した要求は共有prefix cacheに何も登録しないため、LPAはバッチ用のopt-inになり（[LPAの使用範囲](lpa.ja.md#使用範囲)）、画像入力構成は[画像入力](vision.ja.md)に記録しています。これらは既定値の判断であり、下記の基準日付きの状態欄を変えるものではありません。現在の値は[起動設定](server-configuration.ja.md#配布用の既定設定)が正典です。

## 性能施策一覧

「期待効果」は検証する仮説です。「実測あり」も、その記録の条件での観測を指し、本番・品質・併用構成の検収を兼ねません。IDは次版でも維持し、結果が悪い施策も理由とともに残します。

| 施策名 | 内容 | 期待される効果・見る指標 | 今回の到達点／検証先 |
|---|---|---|---|
| P01 標準MTP・先読み3token | checkpoint同梱のBF16 draftを使い、外部draftモデルを追加しない。off／k=1／k=3を比較 | 有効tokenあたりのtarget step削減、decode改善。採択長・draft/verify時間・追加メモリも測る | **実測に基づきk=3を選定。配布のcheckpointでも再量子化した複製でも同じ（2026-09-21）。** 起動テンプレートで有効。深さ1〜5を基準の2台で10入力で測り、要求の採択履歴による深さ、draftの確信度の関門、1段目のsparse top-kを使い回さない設定、rank内のdraft argmaxは同じ入力で測って不採用（関門のhost同期は2台では得と同じだけ費用になる）。複数系列は未検収。[MTP](speculative-decoding.ja.md#両方のcheckpointで深さ32026-09-21) |
| P02 LPA | 後段のattention入力を予測し、過去tokenのMLPと不要queryを省略。末尾は通常計算、生成時は全層実行 | 長文prefill・TTFT短縮。decode高速化は狙わず、近似された状態による品質差を測る | **実測あり・実験用。** 長文課題の限定検証。一般品質・結合構成は別ゲート。[LPA](lpa.ja.md) |
| P03 CUDA fusion：KV復元 | FP8 MLA cacheのコピー・FP32変換・scale乗算をTritonで融合 | KV復元のlaunch・中間tensor削減、主にprefill短縮 | **部品一致・全モデルA/B/A実測あり。** 8Kの約17%短縮は出力1tokenの対照。短文decodeはほぼ不変、起動テンプレートで有効。[CUDA実測](component-validation.ja.md) |
| P04 NoPE attention本体のカーネル化・融合 | Pythonのqueryループと多段演算を、候補集合・scale・因果maskを保って置換 | prefill／decodeのattention処理時間とlaunch数を削減 | **今回のquery chunk拡大・FP32融合案は不採用。** launchと一時メモリ削減だけでは高速化せず、serving未接続。別戦略の根拠が得られたら再評価。[部品実測](component-validation.ja.md#nope-attentionの融合とquery-batchingp04) |
| P05 SM121 attention backend選定 | FlashInfer/TRT系・Triton等の候補について、固定GLMのNoPE／sparse MLA形状・cache契約への対応を先に検査し、同一負荷でA/B | 安定した起動・warmup、対応kernelでの実効性能。意図しないfallbackを検出 | **1.6.0でprefillに採用（`runtime.fa2_attention`、テンプレートでon）。** prefillの大きさの呼び出しは、BF16に展開した行をSM90 FA2 wrapperに渡します。decodeのstepは参照計算のままで、LPAとは排他です。基準の2台で、38,962 tokenのprefillは572〜577 tok/sから、4回の起動で1,242〜1,272 tok/sになりました。同一要求の反復はbit一致のままで、教師強制のNLLは3桁目が両方向に動きました（[サーバー設定](server-configuration.ja.md#配布用の既定設定)）。以下は、そこに至るまでの部品の記録です。**SM120の直接差し替えは不採用。SM90 FA2 wrapperは部品として数値的に使える。** 空行を0にすればv15の差は消えるが、候補の少ない行は許容範囲を超えたままで、kernelの幅は2,048が上限。SM90 FA2 wrapperはGB10で全候補幅の数値検査にすべて合格し、512行で参照計算の約20倍速いが、KVはBF16だけ（FlashInfer 0.6.18はSM90以外でFP8のMLA KVを拒否）。servingの切り替えは、P03・P19・P22・候補順序・KV容量を再検収する別の判断。[直接試験と再検討](component-validation.ja.md#padding付きnative-attentionの直接試験)／[SM90 FA2](component-validation.ja.md#sm90-fa2-mla-wrapperの試験) |
| P06 CUDA Graphs | hostへ戻る判定・動的処理・buffer寿命を整理し、対応shapeでcapture/replay | CPUのstep/launch overhead低減、特に定常decode改善 | **全モデルのA/Bで不採用（2026-09-21）：基準の2台の配信profile（深さ4）で `runtime.decode_graphs = true` にすると、10入力すべてでeagerより1 stepあたり7〜9 ms遅く（平均109.3対101.1 ms）、completionは同一。選択肢はoffのまま残し、後のruntimeで測り直す。** それ以前：選択肢として残し、既定ではoff（1.6.0、2026-09-18のユーザー判断）。同一要求が反復する基準の上で測り直すと、MTP fixtureはeagerとの厳密な一致を通りました（8実行31組。2026-09-12の分岐はexpert内のtoken順でした）。全モデルは起動し、warmupの段・agreement・多バイトの検査を通った後、199,652 tokenの合言葉要求のprefill中に、headがメモリの保護余裕で自己停止しました（空き6.5→2.9 GiB。eagerは6.14 GiBを保ちました）。decodeは31.90 tok/sで、eagerの31.78と30.99の幅の中でした。**全モデル未検収。** LPAのeager制約を解く検証が必要。単体capture成功とサーバー全体対応は別。Spark 2台TP=2の他レシピの報告は+2〜5%（sfxnz PR #1）と変化なし（tonyd2wild PR #16）。コードは採用しない。[性能調査](performance-investigation.ja.md#kernel-launchと同期) |
| P07 launch・同期の計測 | 両rankのkernel／CUDA launch API／NCCL／host syncを別集計。prefill対照との差分でtokenあたりの数を推定 | 律速を特定し、削減前後を再現可能にする | **実測済み。** P03でlaunchは減ったがNCCL回数は不変。イベント時間の和を壁時計時間としない。[CUDA実測](component-validation.ja.md) |
| P08 sync／copy／変換の追加削減 | P07で残る転送・dtype変換・host往復を特定。反復する重み変換を観測した場合だけload時の並べ替えも比較 | 余剰レイテンシ・UMAメモリ交通・一時bufferの削減 | **非同期index検査を独立opt-inとして受入。** 128出力で小幅改善、同期・copy各22回/token削減。GPU kernelは11回増加。配布既定async、直列併用はP18で実測。[実測](benchmarks.ja.md#cpu同期削減の独立評価p08) |
| P09 FP8 KVの独立評価 | 重み精度を固定し、対応backendのFP8／BF16 KVを比較。量子化scale、pool容量、読み出し精度を検査 | cache容量削減による長文・同時数拡大。速度の方向と品質は実測で決める | **FP8経路は使用済み、dtype間A/Bは未了。** 現行起動系はFP8固定。BF16比較にはcache形式・runtime対応が必要。[起動設定](server-configuration.ja.md) |
| P10 UMA・メモリ運用 | KV pin、memory utilization、host空き、コンテナ上限、swap・cache状態を再現条件として固定 | OOM・swap由来の遅延を抑え、測定再現性と収容限界を把握 | **ガード・設定あり、系統的最適化は未了。** FA2のcold cache比較では、実行時のJITは翻訳単位3本（nvccは最大3個、参照ペアでも同じ）で、job数の制限で減らせるのは新しいcacheの最初の起動でのコンパイラ1個分でした。既定envを維持します（[測定と限界](component-validation.ja.md#fa2のcold-cache起動2026-09-20)）。NVIDIA driverのJIT cacheをコンテナの起動を跨いで残す案も同じfixtureで測り、不採用としました。このcacheだけを空にした起動は、全cacheがwarmの起動より遅くなりませんでした（readyまで189.8・172.3秒に対し203.9・197.0秒、順序はcold・warm・warm・cold）。最初の要求の時間とメモリの谷も同じです。全部coldの起動（278.3秒）との差、約80〜100秒は、すでに `/root/.cache` に残しているcacheによるものです。 `drop_caches`は必要なcold-load比較に限定し、定常推論の高速化手段と混同しない。監視記録にはMemAvailableと並べてMemFreeと2 MiB以上の連続空きを残す（1.4.0）。`vm.swappiness` 0と60の比較は200K profileで効果が見えず、ホストは60のまま。[起動設定](server-configuration.ja.md)／[基準ベンチ](benchmarks.ja.md)／[swap](operations.ja.md#監視停滞検知warmup) |
| P11 Prefill chunk × Kpool／indexer | chunk予算を変え、pool・tail・cache境界とその前後で通常／分割処理を比較 | 長文prefillと混合負荷のITLを改善し、状態・候補の取り扱いを維持 | **1.4.0から既定2048、128は不採用。** 同時1系列の200K画像profileで、1024と2048は39Kのprefillを14%・18%上げ、200Kの合言葉要求を410.8秒から361.3秒（2048）に縮めた。代わりにpeerの余白が最大0.7 GiB減る。2系列の1024では2Kの最長停止が1.56秒から2.79秒に延びていた。2048超は未測。`kv_cache_memory_bytes`固定時はmemory profilingを省略すると稼働imageのログで確認済みで、KV予算の指定はchunkを増やした際のactivation peakを検証しません。増やす前にreserveの余白を測り、実行中はホストメモリ監視で保護します。[2系列のP11](benchmarks.ja.md#prefill-chunk-の独立評価p11)／[200K profile](benchmarks.ja.md#200k画像profileでのchunk予算2026-09-17) |
| P12 同時ストリーム計測 | client×1／×2／×4とserverのactive seq数を別記し、TTFT・queue・個別ITL・aggregateを同じ表にする | 単発と並列のtradeoff、待ち行列、長文投入による停止を検出 | **待ち行列と実batchを区別して測定済み。** P13でactive2と複数decode行を確認。[実測](benchmarks.ja.md#標準batchingの独立評価) |
| P13 標準batching | LPAなしのseqs=2→4で実batch重複を確認。逐語決定性とタスク品質を分けて検収 | aggregate throughput向上と、個別レイテンシ・メモリの増減把握 | **限定throughput用途の2系列を受入。** 内容・tool確認と16K×2容量確認は別記。既定1系列、4系列・併用は未検収。他レシピでは25〜100Kの要求2本の同時処理が合計約4 tok/sまで落ちたと報告されている（tonyd2wild #14、コードは採用しない）。ここで受け入れた範囲の外。[実測](benchmarks.ja.md#標準batchingの独立評価) |
| P14 同種タスクbatching | コード／翻訳／要約の同種groupと混在groupを、長さ・同時数・投入量を揃えて比較 | expert重複や投機採択が変わり、aggregateが改善する可能性 | **今回の用途では不採用。** 小規模な投入順比較の改善は約0.7〜0.9%。expert経路・課題完遂品質は未検収。標準batchingを維持。[実測](benchmarks.ja.md#同種タスクの投入順比較p14) |
| P15 コンテキスト長sweep | 入力長とKV予算を独立に変える。8K→32K→128K→262,144は検討する測定点であり、事前に容量と対応を確認 | TTFT・decode・memoryが悪化する地点と運用可能範囲を可視化 | **KV各rank1 GiB固定で、1系列の32Kまでの容量・時間を実測。** 測定30要求をpreemptionなしで完了。長文課題品質と併用は別。[256Kの併用確認](benchmarks.ja.md#256kでの実入力確認)は異なるKV予算で実施。[sweep実測](benchmarks.ja.md#32kまでの独立コンテキスト評価p15) |
| P16 CSA2：候補Reuse／Reindex | 層間候補の再利用・限定再採点・shared poolを比較。各層のKVと必須cache更新は保持 | indexerの削減可能な計算を減らす仮説。候補coverageと全体時間で判断 | **コストの門で中止（2026-09-21）。** 4層fixtureでindexerはprefillの0.09%（2K）・0.31%（8K）・0.50%（32K token。indexer層は1つ、n^1.5程度で伸びる）。全モデルは同型のindexer層が11あるので、200Kのprefillでindexer全体が約4%、再利用で削れる採点の分は約2%。設計の第一の受け入れ門を下回るため再利用は作らない。部品は保持。[設計と結果](indexer-reuse.ja.md)／[以前の観測](component-validation.ja.md#indexerの観測) |
| P17 TP=2／PP=2 | PPの中間tensor・mHC post/comb転送を実装・fixture検証後、同じ精度と負荷で比較 | 通信待ち削減の可能性と、stageの直列化・不均衡による損失を測る | **今回の生成負荷では不採用・既定TP2。** Prefillは改善、decodeは低下。速度A/B/A・限定品質・切断は完了。PPのprofiler再開始で障害が出たため、PP容量・decode traceは未完了。[実測](benchmarks.ja.md#tp2pp2の独立評価p17) |
| P18 MTP＋LPA＋fusion、必要ならGraphs | 単独と組合せを同じ資産・課題で比較。復帰対照を含め、実際のLPA作動・投機採択・captureを記録 | 改善の相互作用を測り、品質・メモリ・復旧の回帰を検出 | **直列MTP3／LPA／fusion／async併用を実測範囲で受入。** 2K／8K速度対比較、24課題・tool、16K容量・近似作動中の切断復帰を完了。Graphs・batching併用・業務検収は別。[併用実測](benchmarks.ja.md#直列併用の評価p18) |
| P19 Prefix caching（APC） | 同じsystem/tools/履歴のcold／warmと異なるprefixを比較。hybrid state・境界・混線を検査 | 繰り返す会話のTTFT・再prefillを削減 | **実測した直列・長文prefix再利用用途で採用、起動テンプレートで有効。** 全モデルA/B/A、実hit、長文の分離・tool・切断を通過。cold処理は小幅悪化。LPA併用はP22で対応。[全モデル実測](benchmarks.ja.md#全モデルのprefix-caching独立評価p19) |
| P20 Indexer workspace適正化 | 実shape・chunk・MTP深さ別の最大必要量を測り、過剰予約がある場合に限定して縮小 | host/KVの余裕を増やし、不要な割当を抑える | **条件付き候補、独立した効果の検収なし。** P16の候補削減とは別施策。上流で解消済みなら追加patch不要。[性能調査](performance-investigation.ja.md) |

**追加施策（2026-09-13）：**

| 施策名 | 内容 | 期待される効果・見る指標 | 現在地／検証先 |
|---|---|---|---|
| P21 Expert Parallel | TP=2・DP=1・2系列・各rank固定KV予算を維持し、Expert層の分割だけをTPからEPへ変更する独立実験 | Expert計算効率と全体throughput。追加メモリ、通信、個別TTFT/ITL、品質も比較 | **今回の性能施策として不採用・既定off。** 全モデルA/B/Aの全4ケースで両off対照より遅い。限定品質・切断復帰・16K×2容量は通過。別負荷・併用は別検収。[実測](benchmarks.ja.md#expert-parallel-の独立評価p21) |
| P22 APC優先＋未処理部分のLPA | 全状態を揃えて復元したHと適用可能長max(0,N-T-H)で判断。通常計算由来だけを共有し、最初の近似以降は共有登録を抑止 | 既存prefixの再利用と長い残余prefillの短縮、要求間での近似状態の混入防止 | **単独校正、限定品質・運用、MTP／融合／async併用と最終held-outを完了。** prefix再利用重視ではMTPなしを選ぶ。[実装契約](apc-lpa-design.ja.md)／[構成別の結果](benchmarks.ja.md#同一入力を再利用する場合の差) |

**追加施策（2026-09-16）：**

| 施策 | 作業 | 期待する効果／指標 | 現在の状態／手順 |
|---|---|---|---|
| P23 NVFP4がBF16のまま残す射影のFP8 weight-only | checkpointの除外リストがBF16のまま残す重み——全層の `self_attn*`（両rank合計11.29 GiB）、`shared_experts*`（1.97 GiB）、`lm_head`（1.18 GiB。safetensorsヘッダから実測）——をload時に出力チャネルごとにFP8 e4m3へ量子化する。NVFP4のexpert・固定カーネル・MTP draftは変えない | 毎tokenで読まれる射影の重み転送量を減らす。短文・長文入力のdecode tok/s、TTFT、メモリを測る | **日本語散文向けの公開した任意設定として採用（2026-09-21）：attention projectionと `lm_head` をW4A16 NVFP4に再パック（route l）、`runtime.derived_checkpoint` で基準の2台がMTP k=3で配信、重みはHugging Face（[Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-NVFP4-attn-lmhead-W4A16)）。losslessではないのでテンプレートには入れない。** attentionの再パック（下のroute g）に `lm_head` を足すと、decodeのstepは全入力で12〜13 ms短くなり（BF16の `lm_head` はdraftの段ごとに読まれる）、教師強制NLLはroute gから日本語+0.27%・英語+0.02%・コード−0.09%・数学+1.53%、200Kの合言葉と261,461 tokenの3か所参照は正答（[配信profile](benchmarks.ja.md#基準の2台の配信profile)）。attention projectionだけのroute gは中間段階で、これに置き換わった。**それ以前、route g：** 同一要求が反復する基準の上で再開し、2026-09-19から21まで基準の2台で深さ4と組にして試験採用した。再パックしたcheckpoint（KDAとMLAのprojection、indexerの `wq_b` をW4A16 NVFP4に。`runtime.derived_checkpoint` で配信）を、expert内の順序の固定の後、prefillをFA2にした状態で、他を同じにしてoff／on／offで測り直しました。9標本のdecodeは、数え上げで32.87→42.39 tok/s（+29%）、散文で20.82→25.31（+22%）、コードで27.62→34.93（+27%）で、採択長は変わりません。無改変の2回の起動は1%以内で一致しました。重みはrankあたり4.0 GiB小さくなり、headの空きは最小9.52 GiBでした。教師強制のNLLは日本語で4.0%、コードで5.9%、数学で4.3%上がり、英語で1.1%下がりました。下にある最初の読み（decodeの利得なし）は、同一の実行の間で11%動く物差しで取ったものです。派生checkpointは作ったhostにしか無いので、テンプレートは固定のcheckpointのままです。MTPの深さ4は、この上で効きます（[投機デコード](speculative-decoding.ja.md#深さ152026-09-1920)）。再パックの対象にshared expertsを足す変種（2026-09-20、projectionだけの構成に対するoff／on／off）は採用しませんでした。重みはrankあたり0.68 GiB小さくなり、数え上げは+3.9%でしたが、draftの採択長が下がって、コードは−10.2%、短いpromptは−23%、散文は−2.3%でした。NLLは0.6〜5.4%動き、多くは下がる方向でした。以下が最初の読みです。**不採用。全モデルのA/B/Aを1回実施して閉じた（2026-09-18、ユーザー判断）。** メモリの利得にdecodeの利得が伴わず、品質はわずかに悪い方へ動いた。 参照機のMTP k=3で、詰め直したcheckpointは起動し、warmup ladder・多バイト検査・199,652 tokenの合言葉要求（363.3 s。無改変は361.3 sと354.2 s）を通過した。重みのロードはrankあたり4.0 GiB小さく（95.76→91.76 GiB）、headの最小空きは5.5〜5.9 GiBから10.0 GiBに増えた。decodeは26.96 tok/s（無改変24.91と26.21）で、無改変自身の幅（6標本で24.35〜34.44）の中。MTPの平均採択長が2.9から2.37に下がり、stepが速くなった分を打ち消した。4文の教師強制NLLは英語が無改変の幅の中、数理が約5%上（0.62、無改変0.58〜0.59）、参照実行とのargmax一致は0.909（無改変同士は0.946〜0.962）。tool利用・3か所参照・FreedomBenchは未実施。 方式をload時FP8から、attention側のlinear（KDA射影、MLAの `q_a`／`kv_a`／`q_b`／`kv_b`／`o_proj`、indexerの `wq_b`）をofflineでweight-onlyのW4A16 NVFP4へ詰め直す形に変えた。tenhksparkのroute gに倣うもので、expertと同じMarlinの算術になる。最初の段ではshared expertsと `lm_head` はBF16のまま。4層fixtureでは、詰め直した33テンソルの相対誤差は0.091〜0.095で外れ値がなく、MTP k=3は起動し、layer 3の候補集合は無改変fixtureに対してJaccard 0.950（無改変fixture同士の2回の起動では0.986）だった。8層fixtureでは2つ目のMLA層も0.948で、ずれはMLA層をまたいで積み上がらず、全語彙KLは層数にほぼ比例して約2倍になった（[再量子化検査](validation.ja.md#フルモデルtp2の実験範囲)）。A/B用の配信は `runtime.derived_checkpoint`。着想はMia PR #139（コードは採用しない。そのPRが対象にする先頭のdense MLP層は当方では既にNVFP4）。2026-09-20にはtonyd2wildのレシピが、別のstack（TP=4、DFlash2、abliterated重み）で独立に同じテンソル集合に到達した：attentionとdense MLPの射影を量子化し、`embed_tokens`・`lm_head`・indexer・`mlp.gate`・KDAのgateはBF16のまま。TP=4でタスク別に×1.14〜1.30を実測、TP=2は机上で×1.21、品質は未測定と著者自身が明記している（commit `9fea48c`、コードは採用しない。そのpatchと `W4A16_NVFP4` の選択は向こうのloaderの話で、本リポジトリは使わない）。門は品質：q/k射影のFP8化はsparse-MLAのtop-k候補選択を動かし得て、[候補順序の正規化](candidate-order.ja.md)で安定させた部分に触る。出典のPR自身が品質を暫定としている。既定を変える前に、同一imageのA/B/A（復元対照つき）・MTP採択パネル・FreedomBenchが要る。同PRのadaptive verification lengthは見送りのまま：eagerのMTP k=3には保つべきCUDA graphがない |

P12は測定軸、P13はscheduler設定、P14は投入順序の実験です。同じ改善を三つに数えません。また、MTPは単一系列でも複数の投機行をまとめて検証できるため、GEMM寄りの仕事を得る前提が必ずしも`max_num_seqs > 1`ではありません。

**追加施策（2026-09-17）：**

| 施策 | 作業 | 期待する効果／指標 | 現在の状態／手順 |
|---|---|---|---|
| P24 要求単位のprefix cache無登録 | 読み出しは残したままGPU prefix cacheへの**書き込み**だけを要求単位で止められるようにする。プールはhash済みblockを空きキューの後ろへ、未hashのものを前へ返し、確保は前から取るため、登録しないレーン同士はblockを再利用し合い、待機中の会話を押し出さない。既存の作法どおりruntime overlay（型付きsamplingフィールドと、要求由来の2つの登録箇所のガード）、image再構築、launcherのknobが要る。新規依存は追加しない | 他のレーンが走る間の長い会話のhit維持率とターン遅延。指定したレーン自身の読み出しは維持されること | **候補、必要性も適用範囲も実測済み、未着手。** 画像profile・KV 2.5 GiBで、80,024 tokenの会話（warm 73,728 token復元・1ターン14.9秒）に対して測った。**no-storeが直すもの**：単体では無害なレーンの蓄積。15,025 tokenのレーンを所有者に触れずに2本流しても無傷だったが、4本で完全に退避し次ターンは182.6秒になった。**直さないもの**：レーン自身の確保量が空きを超える場合。このフラグは実行中の要求が必要とする量を減らさない。42,026 tokenのレーン1本は単独で所有者を退避させており（次ターン184.0秒）、この事例は変更後も残る。プールは70 blockで204,800 tokenの要求1本が61 blockを使うため、所有者だけでほぼ埋まり、ある程度の大きさのレーンなら退避させられる。単発なら9,025 tokenと18,025 tokenのレーンは無傷（73,728 token・15.0秒）で、単発での境界は18Kと42Kの間にある。その背後のgroup別の勘定は未確定で、`server capacity` が会話本数を withheld にしているのはそのためである。着想はMia PR #95（コードは採用しない）。同PRの受入記録では、より大きなプールで同じ損失を作るのに約54Kのレーンが16本必要だった。実装までの運用則は、長い対話の傍らで流すレーンを小さく、かつ少なく保つこと |

## 業務利用の品質・採用ゲート

性能調整は、資料に忠実な応答、ツールの正しさ、ライセンスの扱いやすさと併せて判断します。特定の思想へモデルを寄せることや、安全上適切な拒否を一律に取り除くことを目的にしません。

| 施策名 | 内容 | 期待される効果／示せる証拠 | 今回の到達点／正典 |
|---|---|---|---|
| E01 ライセンス・出所の選択 | モデル・独自コード・採用patch・container・harnessを分け、採用版と通知を固定。MIT/Apache中心の方針を維持 | 採用組織が利用・改造・配布範囲を確認しやすくする | 方針・台帳あり。完成イメージ全体やharnessまで同一licenseとは扱わない。[ライセンス整理](licensing.ja.md)／[通知](../THIRD_PARTY_NOTICES.md) |
| E02 FreedomBench・政治的文脈 | 英語原版、日本語、長い業務資料を分け、通常／MTP／LPA／併用を比較。拒否・誤答・形式・通信失敗を分離 | 対象設問での回答傾向、政治的主張の挿入や資料への不忠実、最適化による変化を可視化 | ローカル評価器と必須仕様あり。原版全問・全構成と独自拡張の検収は未了。[FreedomBench FB-01〜05](freedombench.ja.md) |
| E03 業務機能・運用の検収 | text、tool往復、SSE、cancel、要求間分離、ZCode／Claude Code、長時間負荷、両rank復旧を検査 | ベンチで速い構成が業務でも使えるかを判定し、戻せるprofileを残す | 基礎API等の限定実測あり。harness・本番信頼性は未検収。[検証範囲](validation.ja.md)／[ハーネス](harnesses.ja.md)／[運用](operations.ja.md) |

FreedomBenchは、著者の正答に対する一致を測る、対象範囲の限られた試験です。**高得点でも「思想の偏りが一切ない」とは証明できません。** 採用組織向けには対象・条件・結果・未検証範囲を示し、採用側が用途に即して判断できる証拠を提供します。日本語や長文の独自拡張を公式スコアへ混ぜず、LPAが作動しなかった短文結果を近似品質の証拠に数えません。融合やGraphsを加えた場合も、その実効設定を明記して再評価します。設問・採点法の詳細はFreedomBench文書だけで管理します。

## 次版でも使う比較の進め方

1. **新しい素の基準を作る。** model/revision、license、runtime、SM対応、量子化・cache形式を確認。旧patchを無条件で移植せず、上流が既に解決した項目は「不要」と判定する。
2. **部品から実モデルへ。** exact算術の部品検査と近似の品質検査を区別し、fixture状態、全モデルの同一負荷A/B/Aへ進む。trace採取と速度本測定は分離する。
3. **単体の採否を決め、最後に併用。** まず各施策を独立比較し、採用・保留・不採用まで判断する。依存する部品は基準側にも固定し、比較で変える施策は一つにする。採用候補が揃ってからP18で運用予定の組合せと必要なoff対照を測る。部品を追加するたびに全併用試験を繰り返さず、既存の併用結果は予備結果として保持する。P13/P14、長文拡張、PPを同時に変更しない。
4. **機能受入と性能採用を分ける。** Graphs等の基盤整備は、対応範囲で正しさ・資源上限・復旧性が確認できれば、速度改善が小さくても機能として受け入れられる。高速化としての採用には再現する改善が必要。既定の有効化と最終併用検収も別に判断する。逐語一致、内容・tool正答、拒否、未完了を分け、E02/E03を通す前に業務利用の検収済みとしない。閾値・課題・用途は結果を見る前に固定する。
5. **採用しなかった理由も残す。** 誤差内・遅い・資源不足・品質未達・上流で不要のいずれかと再評価条件を記録。旧版の結果は新しい値で上書きせず、Git履歴と新runで比較する。

### 機能受入と既定設定

単体の判断には、**機能受入（検証範囲付き）／性能採用／既定on・off／併用未検収・検収済み**を別々に記録します。「機能受入・性能差は誤差内・既定off・併用未検収」も有効な結論です。部品や小層fixtureの合格を全モデル対応へ繰り上げません。

P06では、Graph用のメモリ保持、起動時capture時間、対応shapeとbuffer寿命、異常時の復旧を確認します。Graph採用前には、位置別の投機採択率が1.00ちょうどに張り付いていないことも確認します。CUDA graph併用時のその値はvllm-project/vllm#53030の兆候（起動後の検査はMia PR #70に記載）であり、実際の採択率ではありません。Graphは副作用ゼロの設定ではありません。LPAのprefill経路とMTPの投機経路はP18で確認し、速度だけを理由に全経路をGraph化しません。PyTorchの[CUDA Graph制約とメモリ管理](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)を参照。

### 比較記録の共通項目

次版・次実験では、以下の行を実施レポートに使用します。新たな設定ファイル形式や自動採点器を定義するものではありません。

| 項目 | 残す内容 |
|---|---|
| 対象と判断 | 施策ID／旧版・新版／狙う用途・指標／事前の採否基準／採用・保留・不採用と理由。機能受入の範囲・性能採用・既定設定・併用検収を別記 |
| 固定資産 | model revision、source commit、両rank image、runtime/backend、精度、template/tokenizer、projector/view、license台帳への参照 |
| 実効設定 | profile fingerprint、TP/PP、active seq、chunk、KV dtype/容量、context、MTP/LPA/fusion/graph/APCの実効状態 |
| 負荷 | corpus/課題/設問hashとsplit、要求ID、実入力・出力token、effort/sampling、投入順序・同時数・実batch重複、他負荷 |
| 時間 | cold起動、warmup、TTFT、prefill対照、個別TPOT/ITL・queue/E2E、aggregate。反復数と分布を保存し、少数試験の中央値をSLAとしない |
| 内訳と資源 | rank別kernel/launch/NCCL/sync/copy、MTPの平均採択長・step数、host空き最小（MemFreeと2 MiB以上の連続空きを併記）・container peak・swap・温度。重複区間を加算しない |
| 品質と完全性 | 予定／完了／欠落／エラー、拒否、打切り、内容・tool・資料忠実性、数値・状態診断、復帰対照。未検収項目も明記 |
| 再現と出典 | A/B/Aの全試行、profiler offの実時間、raw run ID、参照文書commit、公開要約へのリンク、復旧設定 |

入力長・tokenizerが変わる新版では、同じ文章による業務比較と、同じtoken予算による処理量比較を分けます。公開文書は生ログを複写せず、この台帳から結果の正典へリンクする形で更新します。
