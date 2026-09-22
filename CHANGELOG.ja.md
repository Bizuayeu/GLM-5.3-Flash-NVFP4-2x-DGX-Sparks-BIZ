# 変更履歴

[English](CHANGELOG.md)

正典は[英語版](CHANGELOG.md)です。GitHub Releaseの本文は英語版の各版の節から作られます。この日本語版は1.6.0から始めており、それ以前の版は英語版を参照してください。項目は英語版と同じ順に並べています。

## 1.9.0 — 2026-09-22

### Changed

- **hashが既にcacheにあるprefix cacheのpageを二重に登録しない（`runtime.prefix_page_dedup`）。** 固定のblock poolは、同じhashのblockが既にcacheにあっても、満杯になった全blockをそのhashで登録する。draft（MTP）の下ではprefixの検索が最後に一致したblockを落として計算し直すため、同じ履歴を再送するたびにKV cache group数（ここでは3）のblockが、既にblockを持つhashでLRU queueに積まれていた。4層fixture（深さ3、dense retention）では再送ごとに3つ増え、この設定を入れると0になり、cacheされたtoken数とcompletionは固定のpoolと同一（27本中27本）、要求時間も同じ。参照対では、KDAの状態checkpointが長い履歴ほぼ一つ分の余地しか残さないので、この重複が効く：57Kトークンの履歴を15回再送すると、設定offでは28Kトークンの履歴が追い出され、onでは残る。変更はsource-pinnedのpatch（`glm53_setup/runtime/patch_prefix_dedup.py`。image buildで `vllm/v1/core/block_pool.py` に当て、`patch_prefix_dedup --check` が固定fileを検証する）で、`_insert_block_hash` に一問を足す：`GLM53_PREFIX_PAGE_DEDUP=1` のとき、hashが既にcache済みのblockを持つblockはhashを持たず、要求の終了時にfree queueの先頭へ戻る。block idとfree queueの順序はそれ以外変えず、hitは先にcacheされたcopyが担う。profile key（任意。省略＝off＝固定のpoolの挙動。テンプレートは書かない）は両rankに環境変数を書き、このcheckoutからbuildしたimage（marker `GLM53_PREFIX_DEDUP_API=1`）を要する。`server preflight` は `prefix_dedup_support` を報告する。参照対の配信profileはこれを設定する。設定したprofileのfingerprintは変わり、設定しないprofileはfingerprintを保って以前のimageで動く。

### Documentation

- benchmarks：参照対での再送プローブ（57K／28Kの履歴、再送15回、offとon、同じ夜・同じimage）と、採用の裏付けにした起動跨ぎの確認：新imageの6起動（一つのprofileは4回起動）のうち5起動が同じcompletionを出し、1起動が別のものを出した。設定にもprofileにも依らない。imageの中身・起動引数・runtime cache・Tritonのautotune表を比べて原因ではないと確かめ、残る候補は痕跡を残さないので、次にそうなった起動はcompletionのtoken idと両rankのcontainer logで捕まえる。README：反復の行がそう述べ、`server_config.py` の起動時の注記はautotuneの固定が全ての原因を消すとは言わなくなった。
- 施策台帳P25（この変更）と起動設定（`runtime.prefix_page_dedup`）。

## 1.8.1 — 2026-09-22

### Documentation

- 検証：フルモデルの範囲の節を2026-09-22の通常運用の状態から書き起こし、古い「検収を確立しない」の文は経緯として読む。同時実行の範囲の小節を足し、受け入れたprofileでは同時2系列以上は非対応で、同時配信には系列ごとのKV＝rankの追加（TP=4推奨、TP=3非推奨）が要ると明記。READMEの状態表にも同じ行。

### Fixed

- **lockは全ての起動fingerprintの一部で、1.8.0がそれを編集していた。** `config/runtime.lock.json` は各profileのfingerprintにhashとして含まれるため、1.8.0での `status` と `full_model_inference_validated` の変更は全profileのfingerprintを変えていた。1.8.0のcheckoutは配信profileに、その対が起動した値と別のfingerprintを出し、`cluster switch` はその対を扱えなかった（「Frozen launch manifest no longer matches this checkout/lock」）。lockを1.7.1の内容に戻し、fingerprintは公開した値に戻る（配信profileは `948613031b31…`）。1.8.0の通常運用としての受け入れは、README・SETUP §6・ハーネスの受け入れ試験一覧に記録したままで、それが本来の置き場所。次の編集がCHANGELOGの項を伴う意図的なものになるよう、lockのhashを固定する単体テストを足した。

## 1.8.0 — 2026-09-22

### Changed

- **公開した任意設定のKDA input projectionを分割した。** 派生checkpoint（attention projectionと `lm_head` をW4A16 NVFP4、`requant_target = "l"`）は、34層のlinear-attention層のKDA input projectionを、固定vLLMの6本融合 `in_proj_qkvbfg_a` ではなく `q_proj`・`k_proj`・`v_proj` と融合1本の `in_proj_bfg_a`（`b`・`f_a`・`g_a`）として宣言する。重みはbyte単位で同一で、変わったのは `config.json`・`hf_quant_config.json`（`producer.in_proj_layout = "split-qkv-bfg"`）と2枚のsource overlayだけ。kernelの実測で、任意設定のprefillの代価はこの1本のW4A16 Marlin GEMMの幅にあった（2,048行のchunkで、融合幅12,288〜12,800のどれもBF16 GEMMの1.6〜1.8倍。4,096幅3本＋288幅のtailは1.14倍、連結込みで1.3倍）。paddingではない。基準の2台で、融合の配置と配布既定と同じ夜に測ると、分割した任意設定は融合より38,962 tokenで2.6%、199,652 tokenの要求で3.2%、261,461 tokenの要求で3.1%速く（1.7.1が測った代価と同じ大きさ）、既定よりも速い。decodeとメモリは不変、200Kと261Kの要求は正答、融合の配置に対する4層fixtureの一致は同じfixtureの2起動と同じ水準（全語彙KL 1e-4、候補集合Jaccard 0.986）。任意設定のcompletionは配置で変わり（BF16の丸め順）、配置の中ではbit単位で反復する。基準の2台は2026-09-22から分割した任意設定を配信している（fingerprint `948613031b31…`）。
- **2枚のsource overlayを `overlays/` に同梱した**（固定 `kda.py` に重ねる `kda-quant-split.py`、`model.py` に重ねる `mla-quant-split.py`。SHA-256・base fileのhash・markerは `overlays/README.md`）。Hugging Faceのmodel cardは既にこれを要件にしていて、リポジトリが供給するようになった。Ruffはこのディレクトリを除外する（vLLM由来のsource）。

### Documentation

- benchmarksとREADME：1.8.0での測定（同じ夜の3 arm）。READMEの主要な測定値は分割した任意設定を載せる。1.7.1のkernelの数字に注記：rankあたり12,416列、出力のsliceなしで測っていた。実際の幅は12,576で、12,608にpadされ、sliceは層あたりchunkあたり0.48 msかかる。
- 施策台帳P23と起動設定：分割の配置、overlayの対の規則（overlayの対は一つのcheckpoint revisionに属し、食い違えばloadで失敗する）、訂正した幅。
- 配信profileの通常運用としての受け入れを閉じた（2026-09-22）：lockの状態と `full_model_inference_validated`、READMEの状態文、SETUP §6の証拠表、ハーネスの受け入れ試験一覧（npm版ZCode CLIが受け入れた経路。公式DesktopはBLOCKED〔feedback #270〕のまま、Claude Codeは判断で見送り）が「未了」の代わりにそう述べる。
- SETUP：番号付き手順の前に、smokeまでの最短経路の節。

## 1.7.1 — 2026-09-22

### Documentation

- 施策台帳とREADME：tonyd2wildの2026-09-20の記録が、P23と同じattention／MLP射影の集合に独立に到達した（TP=4、品質は未測定、コードは採用しない）。
- 検証：vLLM pull request #55122は2026-09-21に、自身のトラフィックの調査で境界の同点が見つからなかったことから、決定性を副次とする性能変更として言い直された。このstackでは同点を再現し実要求でも捕まえたので、`runtime.stable_indexer_topk` は維持する。
- 実測：公開した任意設定と配布既定を同じ夜・同じimage（1.7.0のruntime）で続けて測った。再パックのprefillは38,962 tokenで1.8%、199,652 tokenの要求で1.2%、261,461 tokenの要求で3.4%遅く、起動間の幅は0.45%。kernelの測定は代価の所在を、prefillの幅でW4A16 Marlinを通るKDAの融合input projectionに置く。2026-09-22朝に1.7.0の節へ足した「prefillは幅の内側で変わらない」は別の夜の起動の比較で、注記した。1.7.0のruntimeでの既定は1.6.0の数字と一致する。
- README：主要な測定値の表は同じ夜の対を載せ、任意設定の短所にprefillの代価を書いた。

## 1.7.0 — 2026-09-22

### Changed

- `runtime.canonical_moe_order = true` の場合、新規の起動にはimageのmarker `GLM53_MOE_ORDER_API=2` が必要になりました。marker 1しか持たないimageを指すprofileは、`server preflight`・`server start`・`cluster switch` の停止前検査で拒否されます。marker 1は、整列のbuffer長を誤っていた以前のimageも持っているためです。このcheckoutから参照imageを作り直し、`reference_image` を更新してください。すでにmarker 1のimageで稼働している対は、引き続き復旧できます。`cluster switch` はその対を復旧先として検査し、切替が失敗した場合はmarker 1のまま再起動して（coordinatorが渡す `server start --recovery`）、許した内容を `warnings` に記録します。fingerprintは変わりません。
- 起動時に `TRITON_CACHE_AUTOTUNING=1` を設定するようにしました。Tritonがautotuneで選んだkernelの構成は、起動のたびに選び直されるのではなく、永続化している `TRITON_CACHE_DIR` にコンパイル済みkernelと一緒に残ります。4層fixtureでは、この設定なしの10起動が2つの数値状態に分かれました。どちらの状態になるかは、KDAのkernel一つ（`merge_16x16_to_64x64_inverse_kernel`）がその起動で選んだ `num_warps` と完全に対応していました。設定ありの6起動は1つの状態になり、起動も約50秒短くなりました。参照ペアはこの設定で3回起動し、3回とも1.6.0と同じcompletionを返しています。全モデルではもともとこの分かれ方を観測していないので、効果を確認できているのはfixtureだけです。profileのfingerprintは変わりません。

### Fixed

- 参照imageは、固定しているvLLMのslot対応付けのkernelにpatchを当て、block tableを行の中だけで読むようにします（marker `GLM53_SLOT_MAPPING_GUARD=1`。issue #53982に対するvLLMのpull request #54296と同じguardで、執筆時点で上流は未マージです）。これがないと、kernelはindexerのtailのscratchを置くKV groupの32要素の行を、position 128から先で1 tokenごとに1 byteずつ遠くまで読み、未割り当てのメモリに届いた時に失敗します。基準の2台の配信profileでは、約25万tokenを超える要求がすべて、両rankのCUDA illegal memory accessに終わっていました（248,954 tokenは完走、252,958 tokenは失敗、261,461 tokenは4回中4回失敗）。配布している既定は、同じ読み込みをしながら失敗していませんでした。guardを入れると、fixtureではcompute-sanitizerの不正な読み込みが0件になり、log確率はguardのないimageとbyte単位で一致します。配信profileは252,914 tokenと261,461 tokenを正答で完走し、decodeの完了文は1.6.0と一致しました。参照imageを作り直して `reference_image` を更新してください。このmarkerを要求する検査はありません。固定しているvLLMが上流の修正より先へ進んだら、patchは外します。

### Documentation

- MTPの深さ：配布のcheckpointにも再量子化した複製にも、一つの深さk=3を使います（ユーザー判断、2026-09-21）。深さ2〜5を基準の2台で、調整用・評価用に分けた10入力で測りました。深さはほとんどのcompletionも変えるので、表にはその旨を書いています。2026-09-19から再量子化したattention projectionを深さ4で配信していた基準の2台は、この版からk=3で配信します（[投機デコード](docs/speculative-decoding.ja.md#両方のcheckpointで深さ32026-09-21)）。
- 再量子化したcheckpoint：attention projectionと `lm_head` をW4A16 NVFP4に再パックしたもの（route l）が、attentionだけの再パック（route g）に代わって日本語散文向けの任意設定になりました。decodeのstepは全入力で12〜13 ms短く、教師強制NLLはroute gから最大1.5%（数学）しか動かず、200Kの合言葉と261,461 tokenの3か所参照は正答です。重みはNVIDIAのモデルカードを添えてMITでHugging Faceに公開しています（[台帳P23](docs/optimization-catalog.ja.md)、[ライセンス](docs/licensing.ja.md#重みのmit通知)、[起動設定](docs/server-configuration.ja.md)）。losslessではないので、テンプレートは固定の重みのままです。
- 全モデルで測って採らなかったもの（数字は各文書に）：decodeのCUDA Graphs（eagerより1 stepあたり7〜9 ms遅い。`runtime.decode_graphs` はoffのまま）、要求の採択履歴から決める深さ、draftの確信度の関門（2台ではhostの同期の費用が得と同じだけになる）、1段目のsparse top-kをdraftの段で使い回さない設定、rank内のdraft argmax、CSA2のindexer再利用（コストの門で中止。indexerはfixtureでprefillの1%未満、全モデルの射影で200Kでも約4%）。これらの設定は配布するコードに入っていません。
- 検証：このstackではdraft側の変更はcompletionを変えると見込みます（draftする候補が変わると、targetのBF16 logitsの同点が別の側に倒れる）。そうした変更は採択とNLLで判定し、文章の一致では判定しません。
- ベンチマーク（2026-09-22、タグ後）：公開した任意設定で、READMEが「任意設定では未測定」としていた項目を測りました。38,962 tokenのprefill、固定の短いpromptの後のdecode、255,950 tokenの合言葉、262,080＋64の容量2回、3か所参照3回と、両rankのメモリ最小値です。READMEの比較表に数値を載せました。
- README（2026-09-22、タグ後）：読者の順（導入するもの、必要な環境、始め方、確認した範囲、取り組み、関連研究）に並べ直しました。主要な測定値の二つの表は、配布既定と公開した任意設定の一つの比較表と長所・短所の表になり、文種の順を一つにしました。確認した範囲は、ツール・fixture・全モデル・ハーネス・テンプレート・任意・不採用・未検証で区分した状態表になり、各行は状態と所有文書へのリンクだけを持ちます。版から版への注記は数値を所有するbenchmarksへ移しました。ライセンスの早見表に公開した再パックを足し、起動設定TOMLとLPAの段落は導入するものの隣へ移しました。
- 文書一覧：文種は 数え上げ／散文／コード、教師強制の文は 日本語／英語／コード／数学 の順に統一しました。ベンチマークの役割から施策IDの列挙を外し、READMEの主要な測定値の節を実測値の唯一の写しとして明記しました。
- ベンチマーク：同一要求の文でpromptを正準の順に並べました。
- README：主要な測定値の見出しは測定した版数を持ち（`tools/check_publication.py` が最新のbenchmarksの節に縛る）、数字を持つbenchmarksの節を指すようにしました。確認済みの表にCSA2の行を足し、公開した任意設定を記述しています。最適化概要の「次の候補」から1.6.0と1.7.0で測った項目を外し、用途別の構成の表にコードの既定・散文向けの任意設定・バッチprefillの任意設定を並べました。

## 1.6.2 — 2026-09-20

### Fixed

- テスト：1.6.1で加えたホスト判定のテスト10本は、`os.name` を書き換えて別のプラットフォームを装っていました。pathlibも同じ値を読むため、Linuxではパスを作れずにエラーになり、1.6.1のCIはubuntuで失敗していました。`server` にだけ別の `os` の見え方を渡す形に直しています。判定するコード自体は1.6.1から変わっていません。

### Documentation

- 運用：異常終了後に報告されたGB10の電力制限とモデル由来の低速化を区別し、固定KV byte予算ではmemory profilingが省略され、prefill chunk増加時のactivation peakを検証しないことを明記しました。
- 検証：無改変armのばらつき、実行物の照合、cold／APCの分離、completion hash、要求／token集計を判定手順にしました。保存済みの再量子化・深さ比較を再点検し、測定値と設定の採否を維持しながら、主張を測定promptと残された証拠の範囲に限定しました。同じ深さでは、再量子化の利得はstepあたりでも成り立ちます（step/sの目安で3 promptとも+20〜22%）。
- 部品検証：8層fixtureのFA2 cold cache起動でJIT並列制御の未指定とMAX_JOBS=2／FLASHINFER_NVCC_THREADS=1を比較しました。両方完走しました。実行時のJITはfixtureでも参照ペアでも翻訳単位3本で、この制限で減らせるのは新しいcacheの最初の起動でのコンパイラ1個分です。runtimeの既定値は変更していません。
- README：単機0xSeroレシピと別配布のmosaic checkpointを比較用の参照先として追加し、配布物別のライセンスとコード・重み未採用を明記しました。主要な推論測定値は1.6.0のままです。
## 1.6.1 — 2026-09-20

### Changed

- 入口層の構造を、挙動を変えずに整えました。`server.main`・`cluster.main`・`server_config.validate`・`server_config.serve_args` は名前の付いた小さな手順の列になり、7本のfixture runnerはGPUなしでengine設定を示せ、`apc_history` の補助関数はテストから呼べます。既定値、生成される `serve` 引数、起動のfingerprint、コマンドラインは変わりません。fixture runnerのうち2本は、コンパイルモードを引数で受け取るようになりました。照合器では、設定・検証・CLIの459件の結果が1.6.0とbyte一致でした。参照ペアでは、1.6.0のcheckoutとこの版のcheckoutから `server plan`・`server preflight`・`prepare` RPCを実行し、両rankで同じ結果を得ています。`apc_history` は、決定論の偽サーバーに対して1.6.0と要求単位で一致しました。本番profileはLPAを含まないため、このハーネスを実サーバーでは実行していません。テストは325本から388本に増えました。

### Documentation

- README：検証済みの表とその説明を1.6.0の内容に合わせ、参照ペアの配信profileが何を測った値かを、条件と代価つきで示す節を加えました。
- 1.5.0と1.6.0が追い越していた記述：画像入力へのリンクから「200Kでの」を外し、最適化概要の日本語版から、英語版が持っていなかった旧い既定値3つを外し、検証ガイドは深さ1〜5とdecode graphsを「未検証」ではなく実測済みと書き、ライセンスのページに日本語版だけが持っていた見出しを加えました。

## 1.6.0 — 2026-09-20

### Documentation

- 施策台帳：P23（BF16のまま残されたattention projectionをW4A16 NVFP4へ再量子化）は、この版の中で二度読みました。最初の全モデルA/B/Aでは不採用として閉じました。rankあたり4.0 GiB減、decodeは無改変どうしの幅の中、数学の文でNLLが約5%上がる、という結果でした。ただしその読みは、同一の実行の間でまだ11%の差が出ていた頃のものです。同一要求が反復する基準の上で測り直すと、採択長は変わらないままdecodeが22〜29%上がり、NLLは4文のうち3文で4〜6%上がりました。基準の2台は2026-09-19から、これを試験採用として配信しています。テンプレートは固定のcheckpointのままです。再パックの対象にshared expertsを足す変種も測り、採用しませんでした。P05（prefillのFA2、採用）とP06（decode Graph、選択肢として残し既定はoff）にも、1.6.0の結論を書いています。
- README：DGX Spark 2台向けの他の公開GLM-5.3-Flashレシピの表を追加しました。リンク、ライセンス、このリポジトリがそれぞれから何を得たかを載せています。これらの事実はこの表が持ちます。部品検証・運用・施策台帳・サーバー設定・投機デコードの各文書の引用は、名前とpull requestだけを残してライセンスを繰り返さない形にし、`tools/check_publication.py` は `docs/` の下のレシピへのリンクやライセンスの再記述を拒否します。
- README：現在の既定での主要な測定値の表を追加しました（prefill、decode、200Kと256Kの要求、不安定な3か所参照、最小空きメモリ）。数値の正典は引き続き `docs/benchmarks.ja.md` です。

### Added

- **`runtime.fa2_attention`**（任意、未指定はfalse）。候補を保持するNoPE attentionのうちprefillの大きさの呼び出しが、BF16に展開した行をFlashInferのMLA paged wrapper（`fa2`、page size 1、候補をKV pageとして渡す）に通します。decodeのstepは参照経路のままです。基準の2台で、38,962 tokenのprefillは572〜577 tok/sから1,242〜1,272 tok/sになり、同一要求はbit一致で反復し続けます。LPAとは排他です。測定は `docs/server-configuration.ja.md` にあります。
- **`runtime.stable_indexer_topk`**（任意、テンプレートは `true`、未指定はimageの既定）。固定しているvLLMのkpool indexerのtop-k kernel（decodeの `persistent_topk` と `top_k_per_row_prefill`）は、512位の境界にpoolの同点があると、同じ入力から違う集合を返します。そうした同点一つで、completionが決まった場所で時々割れていました。参照imageは両方の呼び出しを `glm53_setup/runtime/stable_topk.py` に通します。decodeは安定なsort、prefillはkernelのまま同点のある行だけを選び直すので、同点は必ず低いpool indexに決まります。`server preflight` はimageに `GLM53_INDEXER_TOPK_API=1` を要求します。`false` は比較用のarmです。測定は `docs/server-configuration.ja.md` にあります。
- **`runtime.canonical_moe_order`**（任意、テンプレートは `true`、未指定はimageの既定）。参照imageは、固定しているMarlin MoEの経路にpatchを当て、kernelの前で各expert内のtokenをtoken id順に並べます（`glm53_setup/runtime/moe_token_order.py`、source固定の `patch_moe_order`）。`GLM53_CANONICAL_MOE_ORDER=1` と、目印 `GLM53_MOE_ORDER_API` を設定します（この版から `2`。preflightは `1` も受けるので、切替は動いている古いimageを復旧先として用意できます）。8層fixtureでは同一要求がbit一致で反復するようになり、decodeは約1%遅く、prefillは変わりません。`server preflight` は、目印の無いimageに対する `true` を拒否します。キーの無いprofileはfingerprintを保ちます。**使うには参照imageを作り直してください。** 基準の2台でも同一要求がbit一致で反復し、decodeは変わらず（中央値31.1対31.0 tok/s）、MTPの採択長は上がりました（3.09→3.35）。
- **`server agreement`。** 保存した参照実行との、教師強制での一致を見ます。自作の4文を `prompt_logprobs` に通し、実際の次tokenの順位とlog確率を取り、`--reference` を付けると以前の記録に対するargmax一致・top-5の重なり・driftを出します。このコマンドを書いた時点では、配信中の全モデルは同一要求を正確には反復しませんでした（基準の2台でargmax一致が約0.96）。そのため、すべての要求に検査した形で答えが返れば合格とし、反復の差は物差しとして報告します。原因はこの版の中で二つ見つかり、上の `canonical_moe_order` と `stable_indexer_topk` で直りました。
- **4層fixtureでの再量子化の検査**：`quant-error`（NVFP4のtensorと元のBF16との重み空間の誤差をtensorごとに出し、他がbyte一致であることを確かめる）、`agreement-fixture`（top-5の行、全語彙のlog確率、層3の候補集合）、`agreement-compare`（全語彙のKL、argmax一致、候補集合のJaccard）。[検証](docs/validation.ja.md)に、無改変のfixtureが二回の起動の間でどれだけ動くかを記録しています。この種の比較の物差しです。
- **`runtime.derived_checkpoint`**（任意、既定では無し、既存profileのfingerprintは不変）。固定のcheckpointを手元で再量子化したコピーを、必要なsource overlayとともに配信します。`server preflight` は、checkpointが宣言するtarget、MTPのdraft層が量子化されていないこと、各overlayのhash・目印・置き換えるimage内ファイルのhashを確かめます。基準の2台はP23のA/B/Aで使った後、2026-09-19から試験採用として配信に使っています。
- **`validation.run_repeat_trace`**：一つのprocessの中で要求を繰り返し、最初の実行と出力が違う最初のmoduleを名指しします。8層fixtureでは、それは常にrouted expertsです。各expert内のtokenの順序が呼び出しごとに変わり、Marlin MoEの結果がそれに依存します。`--canonical-align` は診断としてその順序を固定し、すべての実行をbit一致にします。`--zero-moe-buffers` は古いscratchメモリの可能性を除きます。`--verify-canonical` は、固定が結果をそのノイズの大きさでしか変えないことを示し、`--timing` はfixtureでdecodeが約1%遅く、prefillが変わらないことを測りました。上流はこの原因をvLLM issue #52525として追っています。
- **`fixture-build --with-mtp`**：4層fixtureにBF16のdraft層を残します。残した層に続く番号へ振り直し、両方の量子化設定ファイルで全体のNVFP4設定から除外するので、GPU 1台でMTPを動かせます。
- **リリースはtagに従います。** `vX.Y.Z` のtagをpushすると `.github/workflows/release.yml` が走り、`tools/release_notes.py` を通して、その版のChangelogの節をGitHub Releaseとして公開します。1.2.0〜1.5.0のtagはReleaseなしでpushされていたので手で作成し、Changelogの節はあったがtagの無かった1.2.1は、そのリリースcommitにtagを付けました。

### Fixed

- **`cluster resume` が、しばらく配信している対を確認できるようになりました。** headの準備確認は、起動完了の行をログの末尾200行の中だけで探していました。supervisorの `/metrics` の読み出しが数分でその行を窓の外へ押し出すので、観測を失った後のresumeは、headをreadyと見ることができませんでした。末尾にその行が無い時は、ログ全体を読みます。
- **FP8 unpack融合のkernelが、入力の大きさごとにkernelをcompileしなくなりました。** 要素数が `tl.constexpr` だったので、大きさが違うたびにTritonのkernelを一つcompileして保持していました。参照attentionは毎回同じ少数の大きさしかunpackしないので、表に出ませんでした。`runtime.fa2_attention` では、触るcache行の数が呼び出しごとに違うため、配信中の各workerのheapが1回の呼び出しにつき約0.2 MiB増え（100K tokenのprefillで約100 MiB、256Kの要求1件で0.15〜0.4 GiB、戻らない）、`~/.cache/triton` にも呼び出しごとにkernelのディレクトリが一つ増えていました。要素数は実行時の引数にしました。結果はbit一致です。

### Changed

- **テンプレートが `runtime.fa2_attention` をonにします。** このキーより前に書いたprofileは、参照経路とfingerprintを保ちます。FA2経路はLPAと排他なので、`lpa.enabled` をonにする時は、併せて `fa2_attention = false` が要ります。テンプレートのMTPの深さは3のままです。固定のcheckpointでは、深さ4は文が予測しやすい所で0〜3%得をし、散文で7%損をします。深さ4が効くのは `runtime.derived_checkpoint` と組にした時です（`docs/speculative-decoding.ja.md`）。
- **`cluster resume` が、中断された切替のやり残しを済ませます。** readyと確かめた対に、warmupの段と、`--config` を渡した場合は両rankへのprofile本文の書き込みを行います。切替と同じく記録に残し、対を巻き戻すことはありません。復旧した旧い対には、どちらも行いません。これまでは、resumeで確認された対は、warmupの段が流れず、rankにあったファイルもそのまま、という状態でcompleteになっていました。
- **`cluster switch` が、SSH接続が切れた時に `start` と `stop` を再送します**（読み取り専用の確認と同じく最大3回）。`stop` は元から冪等でした。`start` は、走り出している試行に対しては拒否せずに `replayed` を返すようにし、終了済み・取消済みの試行は引き続き拒否します。切替が新しいrankを起動する時点では古いrankを両方止め終えているので、その段で通信が一度失敗するだけで切替が失敗し、復旧に回っていました。起動枠の予約、profileの書き込み、warmupは、引き続き再送しません。
- **参照imageがFA2経路を持ち、そのことを示します（`GLM53_FA2_ATTENTION_API=1`）。** FA2のprofileは引き続き、この経路・そのdispatch・修正済みのunpack融合をcheckoutからbind mountします。目印のあるimageでは冗長ですが、それらより前に作られたimageが切替の復旧先として有効であり続けるためです。
- **`cluster switch` が、両rankにprofileのファイルを書きます。** これまでの切替は、設定を固定化したmanifestだけで渡し、`--remote-config` のファイルには触れませんでした。そのため、そのファイルを読むrank側のコマンドが、動いているものとは別のprofileを見ることがありました。新しい対がcompleteになると、各rankが `--config` の本文をその経路に書きます。起動に使ったprofileに解釈される本文だけを受け付け、内容の違う古いファイルは `<name>.bak-<UTC時刻>-<fingerprintの先頭>` として残し、renameで一度に置き換えます。失敗した切替・復旧した切替は何も書きません。書き込みの失敗は記録の `config` に残り、対を巻き戻すことはありません。`--no-send-config` で従来の挙動になります。両方のcheckoutがこの版である必要があります。
- **supervisorの `resources.jsonl` が、コンテナのcgroupメモリとそのprocessのRSSを記録します。** 2秒ごとに `MemAvailable` と並べて取るので、`memory-reserve` による停止を後から読めます。cgroupとRSSが平らで `MemAvailable` が下がるなら、GB10の共有メモリ上でのdevice側の増加です。RSSが上がるなら、process側の増加です。
- **`validation.memory_probe` が、配信中のworkerの中で要求をtraceします。** `trace_begin`／`trace_end` は、traceするmodule・draftモデル・sparse NoPE attentionの入出力の指紋を取り、一つの要求を基準にして、後の同一要求で実行順に最初に食い違った呼び出しを名指しします。`host_census` は生きているPythonオブジェクトを数えます。`fa2_stage` はFA2経路の一部だけを走らせ、残りを参照経路で答えます。触ったcache行のgatherは、decodeの大きさの呼び出しでだけ行います（prefillのchunkでは、GPUがホストと共有するメモリの上で、MLA層ごとにGB級になるためです）。各メソッドの説明は[サーバー設定](docs/server-configuration.ja.md)にあります。
- **`validation.memory_probe` が、host側も読みます。** `host_stats` は、workerの常駐匿名メモリ、glibcの `mallinfo2` の内訳（使用中、保持している空き、mmapしたblock）、torchのpinned host cacheを返します。`trim` を付けると二回の読みの間で `malloc_trim(0)` を呼ぶので、workerのメモリが増えた時に、保持している空き・生きているオブジェクト・mallocの外のメモリを切り分けられます。
- **`validation.memory_probe`**（任意、既定は `false`）。worker拡張で、dev経路の `/collective_rpc` から呼ぶ `allocator_stats` メソッドが、rankごとのtorchのcaching allocatorの状態（reserved、allocated、segment、retry、`mem_get_info`）を返します。GB10の共有メモリの上で、allocatorの増加と他のhostメモリの増加を切り分けるためのものです。
- **`runtime.derived_checkpoint.enabled`**（任意、既定は `true`）。`false` にすると、profileに表を残したまま固定のsnapshotを配信します。再量子化したcheckpointも、他の施策と同じくキー一つで切り替えられます。
- **`runtime.decode_graphs`**（任意、テンプレートは `false`）。decode Graphを一箇所で切り替える、肯定形のスイッチです。`runtime.enforce_eager` は以前の綴りとして引き続き読み、既存のprofileはfingerprintを保ちます。
- **decode Graphを、1系列に限ってMTP・prefix cachingと併用できます。** これまで `runtime.enforce_eager = false` は、MTPかprefix cachingがonだと拒否され、captureの大きさも `[1]` に固定されていました。固定のruntimeは、MTPがonだとこの大きさを受け付けません（decodeの大きさは `num_speculative_tokens + 1` の倍数に切り上げられます）。ランチャーは、MTPがonなら `[num_speculative_tokens + 1]`、それ以外は `[1]` を渡します。`max_num_seqs > 1`、LPA、同期のindex検査とGraphの併用は、引き続き拒否します。根拠は、expert内のtoken順を固定した4層のMTP fixtureで、eagerとGraphの実行が、prefix cachingの有無によらず、どの長さでもtokenとlogprobが同一だったことです（部品検証）。全モデルでの速度と採択は主張しません。
- `validation.run_graph_fixture` に `--apc`（prefix cachingをon）、`--lengths`、`--seqs`（別々のpromptのbatch）を足し、各標本のcached tokenを記録します。
- `mtp.num_speculative_tokens` は、測定済みの1と3だけでなく1〜5を受け付けます。profileから深さの掃引を起動できます。テンプレートは3のままです。
- `tools/check_publication.py` は、READMEの主要な測定値の表の版が、英日のどちらかでも、benchmark文書の最新の `Measurements on X.Y.Z` の節と違うと失敗します。
- 非公開の計画書は `docs/plans/` に置き、この経路を無視対象にしました。それまで監査は、計画書を公開ファイルとして数えていました。
