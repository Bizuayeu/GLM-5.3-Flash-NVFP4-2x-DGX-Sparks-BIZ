# CUDA・indexerの部品検証

[English](component-validation.md)

以下のunpack／indexerの測定は、固定したGB10 2台構成、runtime source `74548eeac4ca5abe9f0036b3ac89f77555a31ec7`、両rankのimage `sha256:32394330800422a71df89c89d399b8bd17d2dbe90806572ea4583f15ad46f09a` を対象とします。後半のGraph fixtureは、以降に記す独自の範囲とsource識別子を持ちます。いずれも実験的な測定であり、本番運用や言語品質の検収ではありません。

## 測定条件

TP=2、Marlin W4A16、FP8 KV、eager、1系列、prefix cachingなし、**LPA/MTPは無効**です。部品workerは排他的な要求の間でunpackモードだけを変更します。入力は保持したLLM-jpコーパスのvalidation分割（SHA256 `fe1a7ffbf8fe36e9db925c25459863ce0cd432c6fd4ebe0585482dcd41da0ddb`）で、held-outのtest文書は使いません。各経路でwarmupを行い、その後に1出力を5回、または128出力を3回測定します。temperature=0、seed=42、出力token数は固定です。時間測定中はtraceを無効にしています。

## 全モデルでのunpack A/B/A

prefillと出力生成を含む、クライアント要求時間の中央値（秒）:

| 入力token | 出力token | unpack off | unpack融合 | off復帰 |
|---:|---:|---:|---:|---:|
| 64 | 1 | 0.357 | 0.326 | 0.359 |
| 2,048 | 1 | 6.162 | 5.076 | 6.133 |
| 8,192 | 1 | 24.513 | 20.392 | 24.578 |
| 64 | 128 | 9.361 | 9.349 | 9.370 |
| 2,048 | 128 | 15.231 | 14.068 | 15.171 |
| 8,192 | 128 | 33.535 | 29.285 | 33.418 |

8Kの1出力の対照は約17%短縮しました。128出力の行はprefillを含み、**decode単独の速度ではありません**。短い文脈の生成はほとんど変わりません。少数サンプルであり、本番のtail latencyを示すものではありません。

同じ64token入力で1／33出力を対にしたtraceでは、各rankの追加出力tokenあたりGPU kernelが **1,743→1,710** となり、CUDA launch APIの回数も一致しました。NCCLは追加tokenあたり92回のままです。この33 kernel/tokenの削減はprefillでの短縮より遥かに小さく、decode全体のボトルネックを解消したと説明してはいけません。別のunpack単体ベンチでは、全候補行のサイズで4 kernelを1 kernelへ減らしました。

## 数値一致の範囲

GPU部品の検査は、全FP8 byteコードと選定したFP32 scale（符号付きゼロを含む）でTorchのunpackと厳密に一致し、paddingと空候補を含むattention出力も一致しました。全モデルの1出力A/B/Aは、全ケースでtoken IDが一致しました。

128出力のケースは、native／offの反復どうしでも変動しました。別の32出力logprob診断では、offと融合のどちらのrunでも、共通prefix上で候補確率の同点や変化が見つかりました。nativeの反復における共有tokenのlogprob差は約1.31に達しており、摂動が一様に微小であることの証拠にはなりません。全モデルの決定的な同値性と言語品質の受け入れは未解決です。このbaselineの課題は部品の厳密な算術一致の結果とは分けて扱い、そこから融合固有の劣化も本番検収も推定しません。

## Indexerの観測

観測器は11個の `SparseAttnIndexerKpool` opすべてに接続し、共有bufferが上書きされる前に論理候補IDを取得しました。以下の中央値は各rank内のCUDA区間を合計したものです。**2つのrankを足し合わせないでください。**

| 入力token | rank0のop合計 | rank1のop合計 | 1出力要求 | rank0の割合 |
|---:|---:|---:|---:|---:|
| 2,048 | 5.04 ms | 4.94 ms | 6.103秒 | 0.083% |
| 8,192 | 34.84 ms | 34.03 ms | 24.562秒 | 0.142% |

この区間には、kpool opの採点・選択・cache更新が含まれます。外側の `Indexer.forward` にある **上流のindexer projection、正規化、query量子化、gate計算は含みません**。indexer最適化すべての上限を示す値ではありません。Reindexは対象queryの計算を残し、必須のcache書き込みをそのまま削減分として数えることはできません。

この8K validation入力の最終queryでは、隣接するsparse層のJaccardが0.349〜0.693、対象候補のcoverageが0.518〜0.818でした。取得した集合は両rankで一致しています。2Kの選択は因果的な全tokenを覆い、自明なJaccard=1となるため、有用なsparse再利用の証拠からは除外します。固定した1つのvalidation入力から採取したqueryであり、一般的な重なりの分布や調整済みの層スケジュールではありません。

P16は2026-09-21にコストの門で中止しました（[indexerの再利用](indexer-reuse.ja.md)）。この2K／8Kの観測はその前段の材料です。

## 再現手順

[起動設定](server-configuration.ja.md)で `validation.component_worker=true` を指定し、LPA/MTPをoff、検証済みimageを使います。head側で `python -m glm53_setup.validation.run_components --config /path/to/profile.toml --corpus /path/to/documents.jsonl --output /new/record` を実行します。driverは全応答、実token数、A/B/Aの再現性、時間サンプル、rankごとの重なりを保存します。対応するprofiler traceと資源ログも残してください。

併用した結果は[P18](benchmarks.ja.md#直列併用の評価p18)にあります。

## Decode Graphのfixture独立評価

**decode Graphs（P06）：off。** 4層fixture、GB10 1台、context 16,384、chunk 512、1系列、FP8 KV 512 MiB、Marlin W4A16、seed 42、temperature 0で、64／2,048／8,192入力tokenのそれぞれで32 tokenを生成しました。2026-09-12にはGraphとeagerが2Kの出力token 12からlogprobの同点で分かれ、非同期eagerの対照はeagerと一致しました。2026-09-18にexpertのtoken順を固定して（image `sha256:e7a2a606…`、source `0957c4e`、[MoE順の固定](server-configuration.ja.md#再現性のスイッチ)有効、MTP k=3・融合unpack・非同期index検査、prefix cacheあり・なし）測り直すと、eagerと`FULL_DECODE_ONLY`のGraphsは全長・全実行で生成32 tokenとtop-10 logprobが一致したので、以前の分岐はexpert順によるものでした。prefix cacheにhitした実行はなく（hitには16,384のcontextを超えるprimingが要る）、hit経路は未検証です。全モデルではGraphsはeagerより遅く、採用していません（[ベンチマーク](benchmarks.ja.md#全モデルでのdecode-graphs)）。1.14.0からは`runtime.mla_decode_cpb`とも排他です。非公開run：`graph-component-v13-results/*`、driverは`eb30095`の`glm53_setup.validation.run_graph_fixture`。

## padding付きnative attentionの直接試験

**不採用**（[施策台帳](optimization-catalog.ja.md)のP05）。`glm53_setup.validation.benchmark_native_attention` は、実際のFP8 cache packing、64次元の零RoPE、GLMの `arbitrary_fp32` scaleで、候補をすべて保持したままFlashInferの公開sparse MLA APIを呼びます。最初のrun（`native-attention-v15`、`eb30095`からビルドしたimage `sha256:e18f7ae02e96beeb9af954a4e5600ce6ff534d9f91a9fcc2e440a4d91350c494`）は32 headでBF16 2 ulpの許容範囲を超え（空行で3.03125）、固定のSM120 v32／GLM decode tableは幅128/512/1024/2048しか受けないため、2,051候補を2,176へpaddingできません。prefillのみのrun（`native-prefill-v16`）は `GLM_NSA`・top-k 128・page size 64で拒否されました。

**行の種類別の再検討（2026-09-17）。** source `adf8ca9` の `--by-row-kind` で、配信と同じimage `f6fc154c…`・FlashInfer 0.6.18・幅2048を使い、他の負荷のないGB10 1台で誤差を行の種類ごとに分けました。空行は、そのまま渡す場合と、slot 0を指させて出力を後から0にする場合の2通りで、後者はMiaの `Dockerfile` とdrowzeysのWALLS #4/#5/#7の扱いです（コードは採用しない）。

| head | query行 | 全候補の行 | 17候補の行 | 空行（そのまま） | 空行（slot 0＋0化） | 許容範囲 |
|---:|---|---:|---:|---:|---:|---:|
| 32 | 3（decode kernel） | 0.0078 | 0.0625 | 3.03125 | 0 | 0.031 |
| 32 | 65（prefill orchestrator） | 0.0083 | 0.0389 | 3.03125 | 0 | 0.027 |
| 64 | 3 | 0.0078 | 0.0547 | 3.03125 | 0 | 0.029 |
| 64 | 65 | 0.0088 | 0.0566 | 3.03125 | 0 | 0.030 |

空行を0にすればv15の差は消え、全候補の行は許容範囲内ですが、17候補の行（3行では先頭、65行では末尾に詰めた配置）は、空行の扱いによらず許容範囲の1.3〜2倍の差が残ります。2,048候補がすべて有効な場合、P04の計時条件で1/8/65/512行が0.085/0.092/0.413/2.439 ms、参照は0.187/1.105/8.831/68.837 msでした。

kernelに収めるためにpoolを1つ落とす場合（2,051→2,047候補）は、合成データ（乱数のpacked FP8 cacheとquery、64行×32 head）でのみ測りました。出力の最大変化は、質量が最小・無作為・最大のpoolを落とした場合に0.022・0.039・0.151で、許容範囲は0.016です。合成データのattentionはほぼ一様なので、実際のindexerでの影響はこの数字からは言えません。2,047候補を使うには、モデル全体の品質の証拠と別の判断が要ります。

## SM90 FA2 MLA wrapperの試験

2026-09-17（Asia/Tokyo）、source `adf8ca9` の `benchmark_sm90_attention` で、FlashInferの `BatchMLAPagedAttentionWrapper` を、固定vLLMの `FLASHINFER_MLA_SPARSE_SM90` backendと同じ形で呼びました。NoPE（`head_dim_kpe=0`）、page size 1で各query行の候補をKV pageとし、行ごとの正確な長さと `causal=False` を渡しています。固定vLLMがこのbackendを選ぶのはcompute capability 9だけで、そこでは `fa3` を使います。試験ではGB10（capability 12.1）で `fa2` を使い、上の再検討と同じホスト・imageで、モデルの重みは使いませんでした。着想はsfxnzの `docker/Dockerfile.sm121-v8` とtonyd2wildのPR #17・Issue #20（コードは採用しない）で、どちらもこのbackendを `fa2` でcapability 12へ広げています。

**FlashInfer 0.6.18ではGB10でFP8 KVは使えません。** `plan()` が `FP8 kv_data_type for MLA requires an SM90 (Hopper) device, got SM121.` を返すため、試験では参照計算と同じpacked cacheを展開したBF16 KVを使いました。最初の `plan()` はNoPE用のmoduleをJITでビルドし、8.0秒かかりました。

数値の検査は、P04と同じBF16 epsilon 2つ分の許容範囲ですべて合格しました。

- 幅17/63/64/65/2048/2051/2176、連続とsliceのquery：最大差0.00024〜0.0039（許容範囲0.047）。幅の上限はありませんでした。
- 空行（幅0を含む）：厳密に0。
- 2,051候補中のtail候補が1つだけの行：kernelの差0.0、tailを除くと参照出力が15.6変わる。
- 2,176候補の64・128・256行：有限、最大差0.00049。FlashInfer 0.6.17で報告されたNaNは出ませんでした。

計時はP04と同じ条件（32 head、有効な2,176候補、10回呼び出しを同期して5バッチの中央値）です。backendはstepごとに1回planし、層ごとに1回runします。

| query行 | 参照 (ms) | FA2 run (ms) | FA2 plan (ms) | 参照／FA2の1回あたりkernel数 |
|---:|---:|---:|---:|---:|
| 1 | 0.205 | 0.032 | 0.028 | 27 / 1 |
| 8 | 1.193 | 0.073 | 0.036 | 25 / 1 |
| 65 | 9.082 | 0.569 | 0.041 | 195 / 1 |
| 512 | 70.604 | 3.508 | 0.085 | 1,348 / 1 |

**結果:** このkernelはGB10で部品として数値的に使え、512行では参照計算の約20倍速い。1.6.0でprefillに採用しましたが、KVはBF16で保存しません。cacheは `fp8_ds_mla` のままで、呼び出しが触れる行だけをその都度BF16へ展開します。unpack融合、prefix caching、LPA、[候補順序の正規化](candidate-order.ja.md)がSM120のpacked cache経路の上に作られているためです（[起動設定](server-configuration.ja.md#attentionとcacheとcheckpoint)、[全モデルのprefill](benchmarks.ja.md#160でのprefillとdecode)）。

## NoPE attentionの融合とquery batching（P04）

2026-09-13（Asia/Tokyo）、GB10上で2件の独立した部品実験を行いました。image `e18f7ae…`、32 head、latent幅512、queryあたり2,176候補、packedなFP8 cacheと任意のFP32 scaleを使います。どちらもservingのbackendを変更しません。同期のindex範囲検査は有効のままで、参照実装ではGraphsとunpack融合をoffにしています。各測定は、warmup 3回の後、10回呼び出しのbatchを5回行った中央値で、batchごとに同期しています。profilingは別に実施しました。

**参照実装のquery chunk拡大 — 不採用。** source `10d0fc6` の `benchmark_query_chunks`、非公開run `query-components-v17` で、内部chunk 8/32/64を比較しました。P11のscheduler chunkとは別のものです。padding、空行、末尾1候補は事前に定めた数値範囲を満たしましたが、9 queryの結果にはbitwise一致しないものがありました。512 queryではすべてbitwise一致しましたが、chunkを大きくすると遅くなり、一時メモリも増えました。

| 内部query chunk | 呼び出しの中央値（ms） | 追加のpeak live割当（MiB） | 呼び出しあたりGPU kernel |
|---:|---:|---:|---:|
| 8 | 70.473 | 133.42 | 1,348 |
| 32 | 80.245 | 488.75 | 340 |
| 64 | 81.382 | 956.39 | 172 |

内部の既定値は8のままとします。launch数の削減だけでは、この負荷は改善しませんでした。

**FP32融合attention — 不採用。** source `99c7427` の `fused_nope_attention` と `benchmark_query_chunks --fused-attention`、非公開run `fused-nope-components-v18` は、cacheのdecode、FP32でのscore集約、online softmax、値の累積を、集めたKVを実体化せずにまとめます。kernelは全候補を保持し、64bitのアドレス計算を使います。別のGPU試験では、幅0/17/65/2048/2051/2176についてFP64と比較し、非連続なquery、空行が厳密に0であること、有効な末尾候補が1つだけの場合も含めています。宣言した許容範囲は `2 * BF16 epsilon * max(1, max(abs(reference)))` であり、bitwise同値や、あらゆる出力の大きさで2 ulpを保証するものではありません。

下表の時間測定では候補がすべて有効で、空行の検査は別に行っています。上のquery chunkの測定はmaskした行を使うため、同一の負荷とは扱わないでください。

| query行 | 参照（ms） | 融合FP32（ms） | 参照／融合の追加live割当（MiB） | 参照／融合の呼び出しあたりkernel |
|---:|---:|---:|---:|---:|
| 1 | 0.192 | 0.446 | 9.93／0.031 | 27／5 |
| 8 | 1.060 | 1.945 | 79.42／0.250 | 25／5 |
| 65 | 8.787 | 13.566 | 120.27／2.031 | 195／5 |
| 512 | 68.154 | 105.006 | 134.24／16.000 | 1,348／5 |

全出力は有限で宣言した許容範囲に収まり、最大絶対差は0.000122〜0.000488でした。5つのGPU kernelには、融合attention kernelの前後の範囲検査が含まれます。測定した同期APIの回数は、ベンチ側の同期を含めて両経路とも4回のままでした。これらは部品呼び出しの回数であり、全モデルの生成tokenあたりkernel数ではありません。

範囲を区切ったtilingの追試（`benchmark_fused_tiles`、非公開run `fused-tiles-v19b`）では、tile／warpを8/4、16/4、16/8、32/8で試験しました。いずれも同じ許容範囲を満たし、register spillなしでコンパイルできました。最良値は1 queryで0.389 ms、512 queryで105.039 msです。どちらも参照実装の測定値を上回りませんでした。

これらの試作はlaunch数と一時割当を減らしますが、**測定上の速度改善はありません**。再現用に保持し、起動設定へ接続したり、全モデルでの受け入れを主張したり、すでに有用なP03のunpack融合を否定したりはしません。P04は、当てずっぽうのtile探索を続けるのではなく、profilingに裏付けられた実質的に異なる実行方式が得られた時点で再開します。

### head共有Tensor Core候補 — 不採用

`runtime.fused_nope_dot.dot_nope_tf32x3` は別の追試です。各programが復号したK/Vを16 query head間で共有し、明示的な `tf32x3` の行列積と、FP32のsoftmax／累積、BF16出力を使います。IEEEと同値の算術ではなく、精度の選択は[Tritonのdot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)が定義します。ベンチはPTXのhashを記録し、TF32 MMA命令が生成されたことを確認します。与えられた候補はすべて保持し、既存のshape／device／範囲の検査を共有しています。

GPUのFP64、空行、末尾候補のみの試験は `dot-component-v33-results` で通過しました。同じimage `7cb5f93…` でモデル重みなしの単体部品を実行し、時間測定した出力はすべて有限で、事前に宣言した許容範囲に収まりました。PTXにはTF32 MMA命令が含まれます。ただし、16／32 headでコンパイル時のregister spillが1,522／1,530と報告され、kernelは大幅に遅くなりました。512 query行では、参照実装に対するTF32x3が16 headで約63.75→962.42 ms、32 headで68.95→1,945.97 msでした。この実装は **不採用** で、全モデルへの統合も高速化の主張もありません。register spillの記録は、将来の再設計のために残します。

`benchmark_query_chunks --tf32x3-attention --heads 16` または `--heads 32` で、新しい出力先に部品を再現できます。source archiveのSHA256: `290dd45b32997ecb101e237db74485fbc0437339e1a0419761209b46cf49077d`。servingは変更せず、先のSIMT／query chunkの不採用判断も維持します。

## Pipeline fixtureの準備と起動（P17）

source固定の実験的なモデルpatchは、BF16のhidden／residual tensorと、FP32で遅延評価するmHCのpost／comb tensorをPP stage間で伝搬します。GPU 1台・4層の対照（`pipeline-v17-control-a`）は、64／2,048／8,192入力tokenのそれぞれで16出力・3回の反復を完了し、選択tokenのlogprobは有限、bufferのshapeも想定どおりでした。image: `sha256:b9ae526c6369b7bac909c2043853f60172e66eecc1b34e891b1faa607284eb4a`。これは切り詰めたモデルの対照であり、PPの検収や言語品質の結果ではありません。

最初の2 stageの試行（`pipeline-v17-pp2-a`）はstageの重みをロードしましたが、ready前に失敗しました。KDAのみのstageが対応KV layoutを6件報告した一方、KDA／MLAのstageは `LBHNC` だけを報告し、固定のresolverは積集合が空でないにもかかわらず同一のリストを要求したためです。両コンテナともOOMなしで停止しました。

`patch_pipeline_layout` は、既存の混在cacheのshape検査と要求layoutの検証を保ったまま、PP=2で共通して対応するリストを選ぶようになりました。積集合が空の場合は引き続きエラーで、PP以外の選択は変えていません。CPU側の検査は、異なるリスト、優先順位、空・互いに素な対応をカバーします。sourceのutilityのSHA256は `63f58dd2b29543045109f40897ddc5f57e299bb4259f0e37a64b024cb1f20ddb`、patch後のSHA256は `48222663f4fb1842dd4d93e34f6e356ebaa3d4ebe1c15255d0b5bd670359e4d3` です。両ノードで同じ候補image `sha256:5052db070ec651c0b16f1c72f523d631d4b96ba7a2e2b5c3adce46b3f2cace79` をビルドしました。

再試行（`pipeline-v17-pp2-b`）は **不合格** です。ログにはstageごとのMamba cache仕様の不一致があり、両コンテナともOOMなしで終了しました。固定版のGLMのcacheグループ化は、Mamba層を持つすべてのstageにMLA層があることも明示的に要求します。したがって、4層モデルを2+2に分割する構成はPP fixtureとして不適切です。これを動かすためにcacheの契約を緩めたりはしません。

fixture builderは `--layers 8` を受け付け、各stageにKDA/KDA/KDA/MLAのblockを1つずつ配置します。両ノードで、選択した17,526 tensor（25,606,617,384 bytes）すべてをビルドし、固定した元の重みとbyte単位で照合しました。8層の比較は両側とも元の `b9ae526…` PP imageを使います。両stageが同じ対応を宣言する場合、layoutの積集合patchは不要なため、削除しました。観測器は、転送のhashに加えて、有限なattention出力と使用中のKDAのconv／再帰stateを記録します。[実験的な起動設定](server-configuration.ja.md)では、対応するimageと併せてPPを選択できます。

### 8層PPの観測

非公開run `pipeline-v23-control-a` と `pipeline-v23-pp2-a` は、64／2,048／8,192入力token、16出力token、各長さ3回の反復を完了しました。どちらもeager、TP=1、chunk512、context16,384、各rank FP8 KV512 MiB、MTP／LPA／融合／APCなしで、image・fixtureのbytesも同一です。対照はGPU 1台に8層すべてを保持し、PP2は層0〜3と4〜7を別々のGPUへ配置しました。観測sourceのSHA256: `44fcceb70214a951df0e33739fd7a4b11288f517469d083a848e05db9bf671d4`、driverのSHA256: `a2d0410e98af0bdd337653911e733f156d8b875926de4e4222accacb7e9f36c2`。

**198件のstage転送** はすべて、送信側と受信側でhidden／residual／post／combのbyte hash、shape、dtypeが一致しました。生成された9本のtoken列はすべてPP1の対照と一致し、観測した全tensorが有限でした。64 tokenでは、層の出力、使用中のKDA state、選択tokenのlogprobも厳密に一致しました。

より長い入力ではbitwise不変ではありませんでした。1,584件の層観測（使用中のKDA tensorの観測2,376件を含む）のうち、677行がPP1とPP2で異なりました。最初の差は最初の2K prefill chunkの層6にあり、転送境界より後ろです。token列が同じであっても、PP1・PP2それぞれの反復間でも変動が見られました。

| 入力token | arm間の選択logprob差の最大 | PP1の反復差の最大 | PP2の反復差の最大 |
|---:|---:|---:|---:|
| 64 | 0 | 0 | 0 |
| 2,048 | 0.06609 | 0.06527 | 0.06480 |
| 8,192 | 0.18685 | 0.18307 | 0.18979 |

これは厳密な数値同値、誤差の無視可能性、無制限の言語品質を示すものではありません。転送の不具合を特定するものでもありません。実際に送られたtensorは一致しており、反復間の変動はPPなしでも生じています。診断は保持し、より強い受け入れを主張する前に、prefixを揃えた数値差をbaselineと比較してください。

両runともOOMなしで停止しました。ホストの利用可能RAMの最小値は、48 GiB上限のPP1対照で71.82 GiB、32 GiB上限のPP2各rankで87.69／39.71 GiBで、host reserveは4 GiBです。これらの最小値はロード・warmupを含み、定常推論だけの値ではありません。headはexit 0で停止し、peerは引き続き強制終了（exit 137）が必要でした。観測器のhashは同期とCPU copyを伴うため、これは **TP2とPP2の速度ベンチマークではありません**。全モデルの結果は[P17](benchmarks.ja.md#tp2pp2の独立評価p17)で、不採用です。

## Expert配置のfixture独立評価（P21）

byte照合済みの同じ8層モデルを、TP=2／PP=1でEP off、on、off復帰の順に実行しました（`expert-v24-off-a`、`expert-v24-on-a`、`expert-v24-off-b`）。各armは同じ `b9ae526…` image、eager、1系列、chunk512、context16,384、各rank FP8 KV512 MiB、MTP／LPA／融合／APCなしです。全armが64／2,048／8,192入力token・16出力で3回の反復を完了し、観測したtensorは有限でした。これらは切り詰めたモデルのrunであり、言語課題の品質を測るものではありません。

5つのMoE層すべてで実物の `RoutedExperts` を確認し、意図した遷移を検証しました。EP offでは各rankが288個のtensor分割expertを保持し、EP onでは各rankに144個の完全なexpertを配置して、重複のない担当範囲が288個すべてを覆いました。いずれも `MarlinExperts` が選択されています。MoEのTP sizeは2→1、EP sizeは1→2に変わり、モデルのTPは2、DP／SPは1のままです。固定ランタイムの `use_all2all_kernels` はこのトポロジーではfalseで、DeepEPの呼び出しは主張しません。off復帰では元の担当範囲に戻りました。観測器は、実際にロードされたパラメータのshape、dtype、byte数も記録しています。

ready時点のTorchの割当は全armで各rank約12.682 GiB、予約は14.896 GiBでした。これは小さなfixtureについての証拠であり、全モデルのメモリを保証するものではありません。全コンテナがOOMなしで停止し、headはexit 0、peerは137でした。

64・2K入力では、生成tokenが全armで一致しました。8Kでは各armが自身の反復間で変動し、off→onとoff→復帰のどちらの組でも、共通する出力token 2／4／2個の後に分岐しました。共有prefixのlogprob差の最大値は次のとおりです。

| 入力token | offとEP on | offとoff復帰 |
|---:|---:|---:|
| 64 | 0.11011 | 0.08337 |
| 2,048 | 0.22505 | 0.08900 |
| 8,192 | 0.06813 | 0.06038 |

EPの2Kにおける確率差は復帰対照の差より大きく、数値同値を主張するものではありません。8Kのbaseline変動とEPの差はどちらも明示します。**このfixtureが示すのは、配置・ロード・範囲内での実行です。** 後続の[全モデル比較](benchmarks.ja.md#expert-parallel-の独立評価p21)は限定的な課題・容量の検査を通過しましたが、性能面の採用根拠にはなりませんでした。診断用のhashはtensorの同期とcopyを伴うため、全モデルの速度測定では無効にしています。

EP観測器のSHA256: `38c2e228ab086d179b06a7663a0879b4871950fdefc65632e4768d9a51541642`、検証driverのSHA256: `a12c8aa6705e3df670853ff391bddfea2ff039c72e64b36387117a7e8405a0dc`。生の全応答とstate観測は非公開のままです。

## Prefix cacheのfixture独立評価（P19）

`apc-fixture-v41-off/on/restored` は、byte照合済みの4層fixtureと、[直列併用](benchmarks.ja.md#直列併用の評価p18)に記録したV36 imageを使いました。全armはTP1／eager、1系列、context16,384、chunk512、FP8 KV512 MiBです。MTP、LPA、unpack融合、非同期index検査はoffで、変更した設定はAPCだけです。全armが13種類の入力長、各長さ3組のcold／warm、要求あたり16出力token、入力を交互に挟む検査を完了し、OOMなしでexit 0しました。ホストRAMの最小値は、コンテナ上限32 GiB・reserve 4 GiBのもとで89.20／98.04／100.48 GiBでした。

要求したblock sizeは256でしたが、実際の共通cache blockは8,704 tokenでした。8,704入力tokenまでのwarm要求ではcache済みtokenが0と報告され、8,705と16,319の入力ではそれぞれ8,704 tokenを再利用しました。hitしなかった短い条件の結果を、cache再利用の成功やその品質の確認として数えてはいけません。

| 入力token | APC onのcold中央値 | APC onのwarm中央値 | warmのcache済みtoken |
|---:|---:|---:|---:|
| 8,705 | 2.5634秒 | 0.2518秒 | 8,704 |
| 16,319 | 4.5622秒 | 2.2750秒 | 8,704 |

このfixtureの要求時間は16出力tokenを含み、最初のサンプルにはshape依存のJITが含まれる場合があります。全モデルの性能を示す値ではありません。8,705入力までは、生成tokenが全armで一致しました。16,319では、各armとも最初のtokenが同じ2候補の間で変動し、厳密な同点か0.03125のlogprob差になりました。off／onの組は3／6回、off／復帰は5／6回一致しています。arm内の共有prefix logprob差の最大は約0.0632でした。他の長さでもcache hitなしに確率の変動が見られ、bitwise同値も原因の特定も主張しません。

**このfixtureが示すのは、実際のcache再利用と範囲内での実行です。** 交互に挟んだpromptは2Kのみでhitがなく、実際にcacheされた要求の隔離を示すものではありません。後続の[全モデルA/B/A](benchmarks.ja.md#全モデルのprefix-caching独立評価p19)が、実hitを伴う長文・tool・資源／切断・復帰対照の限定的な証拠を与えます。driver SHA256: `da967a4928ab88846c33523fb11c0b9c65234d27f88274ac752c78b3bd8bc198`。LPA／APCは下記のP22の契約を別に使います。

## APC優先LPAのcache隔離（P22）

[P22のfixture](apc-lpa-design.ja.md)は、byte照合済みの4層checkpointで、意図的に教師由来でないprojectorを使います。実際のcache managerでの同時hit、実際のqueryの省略、返された厳密prefix blockのGPU hash、その後の通常要求を検査します。この合成projectorは混入を調べるためのもので、学習した言語品質の候補ではありません。

`p22-fixture-v50` は、TP=1／eager、1系列、context 32,768、chunk 512、FP8 KV 1 GiB、MTP／融合／非同期検査なし、コンテナ上限32 GiB、host reserve 4 GiBで合格しました。実効workerのsource SHA256は `695039966835f037286c89f7de2657cc1423795991ce99ec85120a61cb85009d`、固定したGPU worker patchのSHA256は `53647920093e7c141f0aad7dcbb033925f867970dcaa9e525a21391474281eb0` です。fixture driverのSHA256は `9bafd1e58c0f94f7b9ce9cbb8f76ec0fc09b7a083cb410110afe4ec937c917ab` です。

N=18,432、H=8,704では、近似suffixが層3で9,216 queryを省略しました。hashした8つのprefix tensor viewはいずれも変化していません。続く厳密要求のHは8,704のままで、その後、再利用できる厳密cacheが17,408まで伸びました。coldな近似要求はH=0で、再利用できるprefixを残しませんでした。3件の厳密対照の16 token出力はすべて一致し、合成した近似は最初の分布が異なりました。コンテナはOOMなしでexit 0し、ホストの利用可能量の最小値は91.398 GiBでした。

`p22-fixture-v57` では、固定image `sha256:59eeba6c94db9f1536010f6c258992c203458703dcf1e23b0bde7cb53d445716`、source `f7b47fc` で、MTP k=3、unpack融合、非同期検査を追加しました。測定したblockは8,960で、MTPの再実行ではH=8,960を復元するために17,921 tokenまで厳密なprimingが必要でした。N=27,904では8件の要求がすべて完了し、共有prefixのhashは変化せず、近似suffixは共有する再利用範囲を伸ばさず、その後の通常計算で17,920まで伸びました。コンテナ上限は48 GiB、host reserveは4 GiBです。

**厳密なtoken一致の基準は不合格です。** 最初の厳密対照と後の2件の厳密対照は、出力位置9で分岐しました。最初の対照では1つの最上位候補が0.0625の差を付け、もう一方の経路では4候補が同点でした。この最初の分岐での共有tokenのlogprob差は最大0.061949です。不合格の結果とexit 1は、数値面の合格と言い換えずそのまま保持します。

別の診断 `native-apc-mtp-v58` では、**LPAのownerを一度も構築せずに**、厳密prime／厳密follow-upの組を6回実行しました。上記の完全な16 token経路が、どちらも再現しています。これにより、観測された分岐はLPAなしでも生じることが確認できました。bitwiseの安定性や一般的な言語品質を証明するものではありません。cache隔離の証拠、厳密な数値一致、その後の全モデルでの課題・性能の受け入れは、それぞれ別の結果です。

続く `p22-fixture-v62` は、source `3df93c9`、image `sha256:0de1ef13b7bfebb088ac7d9399e2d45decf058f16819d72f5a8e89c1bc993b81` で非同期schedulerを動かしました。MTPのdraftが棄却された後はCPUのcursorがGPUより先行することがあり、LPAのguardは生成tokenのdecodeでのみこれを認め、prompt内の不一致は引き続き拒否します。2件の近似要求はどちらもこの補正を13回観測しました。8件の要求、cache境界の検査、prefix hashの不変はすべて通過し、このrunでは3件の厳密な16 token対照も一致して、OOMなしでexit 0しました。先のnativeのみの変動は、別の結果として引き続き保持します。全モデルでの併用は[ベンチマーク](benchmarks.ja.md#apclpamtp融合非同期検査の併用p22)にあります。

## 履歴fixtureの追加

`fixture-v66-target` と `fixture-v66-mtp3` は、source `441384c`、image `sha256:569538ce8b1c259f3ee13242f387320417b2d63a0b4122b6dc74d92ae74dae33` を使いました。それぞれ62要求を完了し、pool境界、scheduler blockの境界、10／50／90%位置での編集・分岐18条件を含みます。元の共有prefix hashはすべての条件で変化せず、編集した要求が変更したprefixより先を復元することはなく、厳密な再訪では元の共有stateが保たれました。`3df93c9` からのランタイム差は読み取り専用のcache layout RPCで、forwardとcache公開のアルゴリズムは変えていません。

targetのfixtureは厳密な対照一致に合格し、exit 0しました。MTP3／融合／非同期のfixtureは **厳密な対照一致に不合格で、exit 1** です。最初の厳密対照は以前に観測した一方の経路をたどり、試行・復帰はもう一方をたどりました。独立に比較したところ、先の6組のnativeのみの `v58` 診断に、完全な16 token経路の両方が見つかりました。18件のstate隔離条件は通過していますが、それによって数値面の不合格が合格に変わるわけではありません。どちらのコンテナもOOMで停止していません。全モデルの課題・運用面の受け入れは、bitwiseの再現性とは別です。

## FA2のcold cache起動（2026-09-20）

GB10 1台のbyte検証済み8層stock fixture（ロード重み23.91 GiB）で、JIT並列制御のenv未指定と `MAX_JOBS=2`・`FLASHINFER_NVCC_THREADS=1` を比較しました。両armとも1.6.0のsource、参照image `3a396af5…`、Marlin W4A16、FA2 prefill、1系列、eager、文脈16,384、chunk 512、FP8 KV 512 MiBです。armごとにNVIDIA driver cacheも含む新しい空のruntime cacheを使いました。既存cache・OS page cache・swapは操作していません。他のGPUサービスは比較中に停止し、終了後に復元しました。

| 観測 | env未指定 | 2／1 |
|---|---:|---:|
| ロード開始からengine ready | 270.36 s | 266.50 s |
| ready後の最初の2,048 token要求 | 9.26 s | 8.91 s |
| 起動・要求全体の最小MemAvailable | 71.38 GiB | 71.11 GiB |
| 同要求中の最小MemAvailable | 80.12 GiB | 80.94 GiB |
| 観測したnvcc／ciccの最大同時数 | 3／1 | 2／1 |

両方ともOOMなし・正常終了し、測定要求でFA2を8回呼び、同じ1 tokenを返しました。FA2 NoPEの共有ライブラリと、Triton・TileLang・Inductor・NVIDIA cacheの新規生成物を確認しています。これは実行の確認で、言語品質の証明ではありません。ホスト／プロセスの標本間隔は2秒で、短いピークを取り逃がし得ます。

**判断：env未指定を維持します。** 実行時にビルドするFlashInferのmoduleはFA2 NoPEの1個だけで（fixtureでも参照ペアでも翻訳単位3本）、制限なしのNinjaが同時に起こすnvccは3個、`MAX_JOBS=2` で減らせるのは1個までです。これが起きるのは新しいJIT cacheごとに1回で、FlashInferのNVCC thread数は既に既定1です。nvccが複数走る区間での両armの差は1 GiB未満で、各arm 1回ではばらつきと区別できません。fixtureの最小値は重みのロード中で、JIT中ではありません。
