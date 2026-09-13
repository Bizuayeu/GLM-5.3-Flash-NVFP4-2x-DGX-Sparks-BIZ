# 性能調査

[English](performance-investigation.md)

施策の一覧、基準日、次版比較の項目は[性能・品質施策台帳](optimization-catalog.ja.md)（[English](optimization-catalog.md)）が所有します。本書は詳細な調査手順を所有します。

以下は測定のための仮説であり、特定済みのボトルネックや約束された高速化ではありません。prefillのレイテンシ、要求ごとのdecodeレイテンシ、aggregate throughputは分けて扱います。LPDDRの公称帯域だけでは、量子化MoEのrouting、sparse attention、scheduling、通信のコストは決まりません。

部品・全モデルの実測結果とその限界は[部品検証](component-validation.ja.md)に記録します。

## A100事例から採り入れた論点

[shi3z氏の事例](https://note.com/shi3zblog/n/nd5fc5341b342)は、launch overhead、変換に伴う転送、投機実行、負荷のgroup化を測る動機になりました。対象はA100上のDeepSeekであり、そのthroughputの数値はGLM/GB10の目標ではありません。以下の判断は本リポジトリへの適用であって、その実験の再現ではありません。

| 着想 | GLM/Sparkでの判断と必要な証拠 |
|---|---|
| 細粒度のlaunchを減らす | 調査用の計測は実装済み。kernelを回数と累計時間で順位付けし、API launch overhead、NCCL、copy、実際のkernel処理を分離します。融合は参照実装と突き合わせて検証してから、計測hookなしのレイテンシを測ります。 |
| 展開した中間重みを避ける | Marlin／denseの実traceで、unpack・逆量子化・割当・copyの反復を確認します。block／global scaleと現行の算術契約は維持します。load時の並べ替えは、反復する処理を実測した場合にだけ候補になります。 |
| MTPはstepあたりの有効な仕事を増やす | off／k=1／k=3を、採択token、verify時間、要求ごとのレイテンシとともに比較します。単一系列でも複数の投機行をverifyできるため、GEMM寄りの仕事を得るのにmax_num_seqs > 1は必要ありません。より深い先読みは新しい実験であり、前提として改善するものではありません。 |
| 似たタスクをまとめる | 意味に基づくschedulerを実装する前に、以下の同種／混在batchの統制実験を行います。 |
| CPUへのexpert退避・x86の整数kernel | x86のVNNI／AMX経路を、このARM・unified memory構成へそのまま移植しません。CPU／GPUの競合と実際のcopyには別の証拠が必要です。 |
| 独立した複製へ分割する | 現行の2ノード・フルcheckpointでは対象外です。1ノードにモデルが収まりません。PPは別の候補であり、前提となる実装は以下のとおりです。 |

単位とトポロジは[NVIDIAのハードウェア仕様](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)と[ネットワーク手引き](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)に従います。公称のポート速度は200 **Gb/s**（overhead前で25 GB/s）であり、200 GB/sではありません。ローカルで実測した集団通信の速度は別の量です。E2M1のコード変換だけでは、scale付きNVFP4演算のend-to-endな等価性は示せません。

## Kernel launchと同期

専用の起動profileで `profiling.enabled=true` を設定します。ランチャーは固定版vLLMのTorch profilerを有効にし、`records/profiles/` 以下にコンテナごとの新しい出力先をマウントします。warmup済みの排他的な要求1件をサーバーのon-demand profilerで挟み、両rankのtraceを採取します。実際のpromptと完了した出力token数を記録し、cold JIT・prefill・投機decodeは分けて扱います。

`python -m glm53_setup profile-assess <trace.json.gz>` は、GPU kernelのevent、CUDA launch APIのevent、NCCL kernelのeventを別々に数えます。区間の合計は重複しうるもので、end-to-endのレイテンシではありません。出力1 tokenのprefill対照と、同じpromptで出力長を長く固定した実行を比較し、生成token1つあたりの追加launch数を推定します。MTPでは採択数とengineのstep数も記録します。draft tokenは採択された出力tokenではありません。最終のレイテンシはtraceを無効にして測ります。

レポートはkernel名ごとの累計時間も順位付けし、memcpyのevent、判明した転送バイト数、バイト情報を持たないcopyを分けて示します。これらは観測されたeventの証拠にすぎず、変換bufferが不要であることや、すべての割当・読み出しがtraceに現れることの証明ではありません。JITコンパイルと、実行時に繰り返されるlaunchは区別します。

hostへの同期呼び出しも別に数えます。現行の参照attentionはGPUからhostへindex範囲の検査を行い、LPAはhost側で位置を読みます。変更する前に、traceで実際のコストを割り当てます。検証済みの同等な契約なしに安全検査を外すことは、最適化の受け入れ基準になりません。

kernel側の作業はtraceに基づいて優先順位を付けます。LPAの層別profilerが持つCUDA eventはlaunch数を数えません。参照attentionはPythonでquery chunkを回し、LPAはeagerを必要とします。CUDA graphsやNoPE kernelの融合は別の正しさの案件であり、全候補・cache／状態・品質の検査が必要です。

## Throughputと決定性

最初に実装したlaunch削減の候補は `cache.fused_unpack` です。集めたMLA cacheレコードをunpackする際の、中間FP8 copy・FP32変換・scale copy・乗算を、1つのTriton kernelで置き換えます。attentionの候補とFP32のattention算術は変更しません。既定はoffです。FP8コードとscaleの網羅試験は部品のゲートであり、fixtureの状態比較と、計測hookなしの全モデルA/B実行は別の受け入れゲートです。

部品単体の計測は、現在のソースをマウントしたGPUイメージ内で `python -m glm53_setup.validation.benchmark_unpack --output /path/to/new-record` を実行します。出力の厳密一致を検査し、warmupを除外して5回の計時batchを記録し、各経路を別々にprofileします。試験したGB10では、2,176件と17,408件のいずれもkernelが4回から1回に減り、観測された部品の中央値はそれぞれ約0.045→0.011 ms、0.630→0.212 msでした。この合成の部品サイズは候補行1本分と8本分に相当し、モデル全体の高速化を示すものではありません。trace／結果JSONは、イメージとソースの識別情報とともに保持します。

`context.max_num_seqs > 1` にはLPAなしの別profileを使います。LPAのhookは連続した単一系列を必要とします。まずactive 2系列から始め、資源と課題の検査に通った場合にだけ4系列へ進みます。greedyで僅差のtokenが分岐することは、数値・再現性の診断材料であり、ただちに課題品質の不合格を意味しません。その記録は残したうえで、内容、tool引数、終了理由、要求間の分離、cancel、資源の安全性を個別に判定します。aggregate throughputと各要求のレイテンシ・品質を併記し、aggregateの値を単一要求のdecodeと比較しません。

## タスクgroup化の実験

コード・翻訳・要約の課題に事前ラベルを付けた固定の負荷を使います。決定的な混在順序と同種groupを比較し、要求ID、入出力のtoken予算、同時系列数、seed、cache方針、投入量を揃えます。長さの分布も揃え、paddingや短い入力の偏りが意味的な再利用に見えないようにします。LPAなしのthroughput profileから始め、レイテンシの対照は単一要求の測定のままにします。

要求ごとの品質とqueue待ち、aggregate throughput、完了率、実際のbatch重複を報告します。固定版runtimeがroutingされたexpert IDを取得できる場合は、別の診断実行で、実際の選択境界において数えます。logitsや課題ラベルだけではexpertの再利用は示せません。層・batchごとの相異なるexpert数を比較し、投入順序を繰り返して確認してから、共有された重みに高速化を帰属させます。将来のgroup化schedulerは、starvationに上限を設け、cancel、テナント分離、レイテンシ要件も維持しなければなりません。この実験で利得が示されるまで、意味に基づくroutingやpromptの書き換えを追加しません。

## Expert Parallel（P21）

**現在地：全モデルの独立A/B/Aを完了。試験した負荷では不採用、既定off（2026-09-13）。** [実測と限定した品質・容量の結果](benchmarks.ja.md#expert-parallel-の独立評価p21)を参照。これは採用済みの[active 2系列のbatching実験](benchmarks.ja.md#標準batchingの独立評価)や、別のノード対で行った配布だけの起動受け入れとは別です。以下は比較の手順を記録したものです。

固定版のFusedMoEの並列設定では、TP=2／DP=1にEPを加えると、完全なexpertを2つの分割へ対応付けます。Marlinが宣言する並列対応はこの構成を受け付け、その実行経路はexpert mapを受け取ります。DP／PCP／SPがすべて1の場合、EPを有効にしても `use_all2all_kernels` はfalseです。本実験がDeepEPを呼ぶことや、集団通信をすべて無くすことを前提にしません。これらはソース上の互換性の確認であり、読込・所有関係・数値挙動が成功することの証明ではありません。[起動オプション](startup-configuration.ja.md)は対応するイメージのmarkerを要求し、未検証の最適化の組合せを拒否します。

同じ2基のGPU上でTP=2・DP=1を維持します。変えるのはexpert層の分割であり、モデルの複製追加やKV予約の拡大ではありません。このトポロジは[vLLMのEP解説](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/#layer-behavior-with-ep-enabled)が説明していますが、実際の対応は固定版runtimeで確認する必要があります。[tenhksparkの起動例](https://github.com/tenhkspark/glm53-flash-nvfp4-dgx-spark/blob/main/serve/start-head.sh)は本実験の動機になりましたが、そのnative attention、他の設定、aggregate throughputは、ここでのEP単独の利得を示すものではありません。

1. **互換性と実装：** ARM64／SM121上で、固定版GLM／FusedMoEのloader、MarlinのNVFP4 EP対応、expertの配置、通信backendを確認します。未対応の場合はエラーをそのまま残し、互換な最小の変更を別途検討します。精度、attentionの候補、executor、依存関係を黙って切り替えません。既定offの明示的な起動設定、両rankへの引数、設定fingerprint、preflight検査を、既存の起動設定の仕組みで実装します。CPU契約では、offが現行のコマンドを保つこと、onが意図したEPトポロジを選ぶこと、非互換な設定が失敗することを検証します。冗長expert／EPLBやDPの複製を本実験に加えません。
2. **部品ゲート：** 小さな2 rankのfixtureで、期待するexpertの所有関係、読み込まれたtensorの形状・バイト数の整合、出力・状態の挙動、実際の転送、EP offへの復帰を検証します。起動フラグやimportは、EPが動いた証明にはなりません。rankごとの常駐メモリと、一時buffer・通信bufferの割当を記録します。
3. **全モデルA/B/A：** Aは検収済みの2系列TP profile、BはEPだけを有効にし、その後Aへ戻します。候補イメージ、checkpoint、Marlin W4A16、全候補を保持するNoPE attention、各rank 1 GiBのFP8 KV、16Kコンテキスト、chunk 512、active 2系列を同一に固定します。MTP／LPA／融合／Graphs／APCはoffのままにします。既存の短文／2K入力・64出力の負荷を、client 1／2で再利用します。warmupの後、計画どおり各条件5回を計測します。prompt、sampling、reasoning effort、出力予算、投入量も同じにします。全反復、エラー、実際の同時実行の観測をすべて保持します。
4. **品質・容量・復旧：** 既存のtext／抽出課題と、IDを変えたtool往復を再実行し、続いてSSE／cancelとその次の要求、統制した停止・再起動、別枠の16K×2容量確認を行います。正答値、書式、終了、数値診断は分けて記録します。rankごとのhost空きメモリ最小値、peak割当、swapの発生、KVの使用量・preemption、通信コストを比較します。固定したKV poolは、他のすべての割当を上限づけるものではありません。
5. **判断：** 観測されたばらつきを超えて改善が再現し、品質・容量・メモリ余裕・復旧の基準も満たす場合にだけ、実測したthroughput用途で採用します。aggregate throughputは要求ごとのTTFT／ITL／TPOTと併記します。throughputの改善が個別のレイテンシを悪化させることがあります。許容する追加メモリの予算とレイテンシのtradeoffは、全体比較の前に、創作した上限ではなく実機対のbaselineと設定した予約量に基づいて記録します。改善が誤差の範囲に留まる、あるいはメモリ・レイテンシの代償が許容できない場合は、EP offのbatchingを維持します。より広い同時実行やMTP／LPAとの併用は別の実験です。単体の改善率を掛け合わせません。

生の証拠は非公開の新しい `records/<run-id>/` に置き、公開するのは検証した要約だけにして、P21を実際の結果で更新します。EPの結果はリンク先のベンチマーク報告が所有します。本手順は他の負荷や組合せを検収するものではありません。

## TPとPPの比較

**現在地：全モデルの独立した時間A/B/Aを完了。試験した生成負荷では不採用、既定TP2（2026-09-13）。** PPはprefillを改善しましたが、decodeは遅くなりました。限定した課題・tool・切断の検査は通過し、profilerの再開始で障害が出たため、PPの容量とdecodeのtrace採取は未完了です。[結果と限界](benchmarks.ja.md#tp2pp2の独立評価p17)を参照。

未改変の[固定版モデルソース](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py)は、`make_empty_intermediate_tensors` が無いためPPを弾き、その部分的なPP分岐も遅延したmHCの `post`／`comb` 状態を落とします。本リポジトリの固定patchはこれらの契約を実装し、全モデル評価の前に、8層のfixtureで転送される4つのtensorすべてを検査します。[起動設定](startup-configuration.ja.md)には、層分割を明示した実験用のPP2 profileがあります。eager実行、1系列、EP／LPA／MTP／融合／APCの無効が必要です。

今後のPP比較では、精度・prompt・メモリ予算を同一に保ち、層の割当、KDA／MLAの状態、mHCの境界、rankのメモリ、障害からの復旧を確認します。集団通信の回数を「層数×2」と仮定せず、実際のeventを数えます。MTP／LPAへの対応と、より広い容量の検収は別の作業です。

[Indexerの再利用・再採点](indexer-reuse.ja.md)は、段階を分けた別の候補です。まずコストとoverlapを測り、次に必須のcache更新を保ったまま要求内で選択結果を再利用します。改善の累積を主張する前に、LPAと併せて評価する必要があります。
