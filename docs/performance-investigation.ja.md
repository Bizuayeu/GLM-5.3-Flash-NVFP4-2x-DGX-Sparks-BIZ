# 性能調査

[English](performance-investigation.md)

施策の一覧、基準日、次版比較の項目は[性能・品質施策台帳](optimization-catalog.ja.md)が所有します。本書は詳細な調査手順を所有します。

KV容量、256K／1Mコンテキスト、待ち時間、複数同時実行の考え方は[性能と容量のQ&A](optimization-overview.ja.md#性能と容量のqa)を参照してください。

以下は測定のための仮説であり、特定済みのボトルネックや約束された高速化ではありません。prefillのレイテンシ、要求ごとのdecodeレイテンシ、aggregate throughputは分けて扱います。LPDDRの公称帯域だけでは、量子化MoEのrouting、sparse attention、scheduling、通信のコストは決まりません。

部品・全モデルの実測結果とその限界は[部品検証](component-validation.ja.md)に記録します。

## A100事例から採り入れた論点

[shi3z氏の事例](https://note.com/shi3zblog/n/nd5fc5341b342)は、launch overhead、変換に伴う転送、投機実行、負荷のgroup化を測る動機になりました。対象はA100上のDeepSeekであり、そのthroughputの数値はGLM/GB10の目標ではありません。以下の判断は本リポジトリへの適用であって、その実験の再現ではありません。

| 着想 | GLM/Sparkでの判断と必要な証拠 |
|---|---|
| 細粒度のlaunchを減らす | 調査用の計測は実装済み。kernelを回数と累計時間で順位付けし、API launch overhead、NCCL、copy、実際のkernel処理を分離します。融合は参照実装と突き合わせて検証してから、計測hookなしのレイテンシを測ります。 |
| 展開した中間重みを避ける | Marlin／denseの実traceで、unpack・逆量子化・割当・copyの反復を確認します。block／global scaleと現行の算術契約は維持します。load時の並べ替えは、反復する処理を実測した場合にだけ候補になります。 |
| MTPはstepあたりの有効な仕事を増やす | off／k=1／k=3を、採択token、verify時間、要求ごとのレイテンシとともに比較します。単一系列でも複数の投機行をverifyできるため、GEMM寄りの仕事を得るのにmax_num_seqs > 1は必要ありません。深さ1〜5は実測済みです（[投機デコード](speculative-decoding.ja.md#深さ152026-09-1920)）。 |
| 似たタスクをまとめる | P14として実施し（[タスクgroup化の実験](#タスクgroup化の実験)）、不採用。 |
| CPUへのexpert退避・x86の整数kernel | x86のVNNI／AMX経路を、このARM・unified memory構成へそのまま移植しません。CPU／GPUの競合と実際のcopyには別の証拠が必要です。 |
| 独立した複製へ分割する | 現行の2ノード・フルcheckpointでは対象外です。1ノードにモデルが収まりません。PPは別の候補であり、前提となる実装は以下のとおりです。 |

単位とトポロジは[NVIDIAのハードウェア仕様](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)と[ネットワーク手引き](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)に従います。公称のポート速度は200 **Gb/s**（overhead前で25 GB/s）であり、200 GB/sではありません。ローカルで実測した集団通信の速度は別の量です。E2M1のコード変換だけでは、scale付きNVFP4演算のend-to-endな等価性は示せません。

## Kernel launchと同期

専用の起動profileで `profiling.enabled=true` を設定します。ランチャーは固定版vLLMのTorch profilerを有効にし、`records/profiles/` 以下にコンテナごとの新しい出力先をマウントします。warmup済みの排他的な要求1件をサーバーのon-demand profilerで挟み、両rankのtraceを採取します。実際のpromptと完了した出力token数を記録し、cold JIT・prefill・投機decodeは分けて扱います。

`python -m glm53_setup profile-assess <trace.json.gz>` は、GPU kernelのevent、CUDA launch APIのevent、NCCL kernelのeventを別々に数えます。区間の合計は重複しうるもので、end-to-endのレイテンシではありません。出力1 tokenのprefill対照と、同じpromptで出力長を長く固定した実行を比較し、生成token1つあたりの追加launch数を推定します。MTPでは採択数とengineのstep数も記録します。draft tokenは採択された出力tokenではありません。最終のレイテンシはtraceを無効にして測ります。

レポートはkernel名ごとの累計時間も順位付けし、memcpyのevent、判明した転送バイト数、バイト情報を持たないcopyを分けて示します。これらは観測されたeventの証拠にすぎず、変換bufferが不要であることや、すべての割当・読み出しがtraceに現れることの証明ではありません。JITコンパイルと、実行時に繰り返されるlaunchは区別します。

`runtime.index_checks = "sync"` では、参照attentionがGPUからhostへindex範囲の検査を行い、LPAはhost側で位置を読みます。hostへの同期呼び出しはこれらも別に数えます。変更する前に、traceで実際のコストを割り当てます。検証済みの同等な契約なしに安全検査を外すことは、最適化の受け入れ基準になりません。

kernel側の作業はtraceに基づいて優先順位を付けます。LPAの層別profilerが持つCUDA eventはlaunch数を数えません。参照attentionはPythonでquery chunkを回し、LPAはeagerを必要とします。decode Graphs（P06）とNoPE kernelの融合（P04）は測って不採用です（[施策台帳](optimization-catalog.ja.md#性能施策一覧)）。

## Throughputと決定性

最初に実装したlaunch削減の候補は `cache.fused_unpack` です。集めたMLA cacheレコードをunpackする際の、中間FP8 copy・FP32変換・scale copy・乗算を、1つのTriton kernelで置き換えます。attentionの候補とFP32のattention算術は変更しません。既定値は[起動設定](server-configuration.ja.md#配布用の既定設定)が正典です（直列併用の受入後はon）。FP8コードとscaleの網羅試験は部品のゲートであり、fixtureの状態比較と、計測hookなしの全モデルA/B実行は別の受け入れゲートです。

部品単体の計測は、現在のソースをマウントしたGPUイメージ内で `python -m glm53_setup.validation.benchmark_unpack --output /path/to/new-record` を実行します。出力の厳密一致を検査し、warmupを除外して5回の計時batchを記録し、各経路を別々にprofileします。試験したGB10では、2,176件と17,408件のいずれもkernelが4回から1回に減り、観測された部品の中央値はそれぞれ約0.045→0.011 ms、0.630→0.212 msでした。この合成の部品サイズは候補行1本分と8本分に相当し、モデル全体の高速化を示すものではありません。trace／結果JSONは、イメージとソースの識別情報とともに保持します。

`context.max_num_seqs > 1` にはLPAなしの別profileを使います。LPAのhookは連続した単一系列を必要とします。他の要求とstepを共有した要求はcompletionが変わることがあり、その理由と範囲は[同時実行の範囲](validation.ja.md#同時実行の範囲)にあります。内容、tool引数、終了理由、要求間の分離、cancel、資源の安全性は、逐語の一致とは別に判定します。aggregate throughputと各要求のレイテンシ・品質を併記し、aggregateの値を単一要求のdecodeと比較しません。

## タスクgroup化の実験

**P14として実施し、この負荷では不採用**（[結果](benchmarks.ja.md#同種タスクの投入順比較p14)）。

比較の仕方：コード・翻訳・要約の課題に事前ラベルを付けた固定の負荷で、決定的な混在順序と同種groupを比べ、要求ID、入出力のtoken予算、同時系列数、seed、cache方針、投入量、長さの分布を揃えました。LPAなしのthroughput profileで行い、レイテンシの対照は単一要求の測定です。共有された重みに利得を帰属させるには、さらに別の診断実行で、routingされたexpert IDを実際の選択境界で数える必要があります。logitsや課題ラベルだけではexpertの再利用は示せません。

## Expert Parallel（P21）

**試験した負荷では不採用、既定off（2026-09-13）**（[結果](benchmarks.ja.md#expert-parallel-の独立評価p21)）。

比較の仕方：EPで変えたのはexpert層の分割だけで、TP=2・DP=1・2基のGPU・KVの予約はそのままです。固定版のFusedMoEの並列設定では、TP=2／DP=1にEPを加えると完全なexpertを2つの分割へ対応付け、DP／PCP／SPがすべて1なら `use_all2all_kernels` はfalseなので、この実験はDeepEPを呼ばず、集団通信もすべては無くしません（[vLLMのEP解説](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/#layer-behavior-with-ep-enabled)）。動機はtenhksparkのレシピの起動例ですが、その他の設定はここでのEPの利得を示しません。

既定offの[起動オプション](server-configuration.ja.md)（イメージのmarker付き）を2 rankのfixtureで確認し（expertの所有関係、tensorの形状、出力、EP offへの復帰）、全モデルで2系列のTP profileに対してA/B/Aで比べました。他はすべて固定です（イメージ、checkpoint、各rank 1 GiBのFP8 KV、16K、chunk 512、MTP／LPA／融合／Graphs／APCはoff。短文／2K入力・64出力、各条件5回）。続いて課題・tool・cancel・再起動・16K×2容量の検査を行いました。採用の条件は、観測されたばらつきを超える改善と、品質・容量・メモリ余裕・復旧の維持で、aggregate throughputは要求ごとのTTFT／ITL／TPOTと併記しました。

## TPとPPの比較

**試験した生成負荷では不採用、既定TP2（2026-09-13）**（[結果と限界](benchmarks.ja.md#tp2pp2の独立評価p17)）。

未改変の[固定版モデルソース](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/models/glm5next/nvidia/model.py)は、`make_empty_intermediate_tensors` が無いためPPを弾き、その部分的なPP分岐も遅延したmHCの `post`／`comb` 状態を落とします。本リポジトリの固定patchはこれらの契約を実装し、全モデル評価の前に、8層のfixtureで転送される4つのtensorすべてを検査します。[起動設定](server-configuration.ja.md)には、層分割を明示した実験用のPP2 profileがあります。eager実行、1系列、EP／LPA／MTP／融合／APCの無効が必要です。

今後のPP比較では、精度・prompt・メモリ予算を同一に保ち、層の割当、KDA／MLAの状態、mHCの境界、rankのメモリ、障害からの復旧を確認します。集団通信の回数を「層数×2」と仮定せず、実際のeventを数えます。MTP／LPAへの対応と、より広い容量の検収は別の作業です。
