# 変更履歴

[English](CHANGELOG.md)

正典は[英語版](CHANGELOG.md)です。GitHub Releaseの本文は英語版の各版の節から作られます。1.5.0以前の節は後から訳して加えました。項目は英語版と同じ順に並べています。

## Unreleased

### Removed

- 退役した `runtime.mla_decode_cpb` を取り除いた：source固定patch（`glm53_setup/runtime/patch_mla_decode_cpb.py`）、helper（`glm53_setup/runtime/mla_decode_cpb.py`）、Dockerfileのpatchの `RUN` と `ENV GLM53_MLA_DECODE_CPB_API=1`、keyの受理。keyを持つprofileは、`true` でも `false` でも起動前に拒む（"runtime.mla_decode_cpb was retired in 1.16.0 and removed in 1.18.0; delete the key from the profile"）。1.14.0から1.17.0までに作ったimageはmarkerと届かないpatchをなお持つが、害はない。外すには参照imageを作り直す（[起動設定](docs/server-configuration.ja.md#再現性のスイッチ)）。

### Documentation

- 日本語の変更履歴がすべての版を扱うようになった。英語版にしか無かった1.0.0〜1.5.0を訳して加え、文書一覧も「1.6.0から英日の対」と書かなくなった。
- 第三者の告知に、公開オプションの重み（Hugging FaceのNVFP4 BIZ AXL、MIT、NVIDIAの重みを重みだけで再パックしたもの）を載せた。
- 起動設定：日本語版のKV容量の節を英語版と同じ順に並べ、`profiling` の行の説明を日英で揃えた。部品検証は、初期のfull-targetのrunが何を確かめなかったかを書き、後の再現性の節を指すようにした。

## 1.17.0 — 2026-09-26

### Changed

- MTPと組むLPAは深さ1・2・3を受ける。LPAとAPC/LPAのworker、`lpa-fixture` と `apc-lpa-fixture` の `--mtp` が深さ2を許し、profileの検査はMTPの深さ4か5と組むLPAを起動前に拒む（"LPA with MTP accepts depth 1, 2 or 3"）。これまではそのprofileが起動でき、native LPAの要求がすべて失敗していた。深さ2はLPAと組んで測っていない。`glm53_setup/runtime/lpa.py` と `apc_worker.py` はimageに焼き込まれるので、深さ2にはこの版から作ったimageが要る。古いimageは要求のたびに拒む（[起動設定](docs/server-configuration.ja.md#lpaとmtp制約)）。
- `runtime.fa2_attention` が `true` のとき、`server preflight` はimageのmarker `GLM53_FA2_ATTENTION_API=1` を要求する（行 `fa2_attention_support`、復旧先も含む）。1.6.0以降の参照imageはすべてこれを持つ。FA2のファイルは今もimageの上にmountする（[image契約](docs/server-configuration.ja.md#現行イメージの契約)）。
- fixture runnerが一つの門を共有するようになった。`run_graph_fixture`・`run_indexer_fixture`・`lpa-fixture`・`agreement-fixture`・`run_repeat_trace` は、`fixture-run` と `apc-lpa-fixture` と同じく、fixtureのdownload statusが `complete` で、`all_tensor_bytes_verified` が真らしい値でなく `true` そのものであることを要求する。5本とも、これまではstatusを見ず、旗が `true` そのものであることも要求していなかった。拒むときはどのrunnerも一文 "Only a complete, byte-verified test fixture with 4 layers is allowed" を返し（`agreement-fixture` は4か8層、`run_repeat_trace` は層数を問わない）、`fixture-run` と `apc-lpa-fixture` の "Only a byte-verified four-layer test fixture is allowed" を含む以前の文に代わる。
- `lpa-fixture`・`run_repeat_trace`・`freedombench` が失敗を記録するようになった。例外が抜けると、`result.json` は `"loading"` や `"running"` のまま残らず、`"status": "failed"` と `error`（例外のrepr）を持つ。keyは足すだけで、成功時の記録と終了コードは変わらない。
- attention部品の検査は、数でない誤差をBF16の許容幅の外と読む。SM90の各caseとtail、query chunkの検査、nativeのrevisitの計時の検査は、出力そのものが有限ならそれを通していた。参照にNaNが出たときだけ効き、成功時の記録は変わらない。
- サーバーが受け付けなかったprefix cacheのリセットは、`apc-history` と `apc-lpa-benchmark` を "Prefix cache reset was not acknowledged" で失敗させる（以前は "Dedicated server cache reset failed" と "Cache reset did not complete; do not compare these conditions"）。測る前に失敗するのは変わらない。
- `fixture-build` は `fixture-status.json` を、`quant-error --write-status` が既にそうしていたように、atomicに書く（一時ファイル、fsync、置き換え）。中断したbuildが途中で切れた門のファイルを残すことはない。どちらのファイルも末尾に改行が付き、JSONはそれ以外同じ。
- `tools/prepare_mtp_view.py` はlockをpackage経由で読み、revisionかimageが固定されていないlockを拒む。
- `server agreement` はnative LPAのprofileを、rank 0の状態を読む前に拒む（"agreement requires LPA off or APC-first; native LPA rewrites prefill"）。headが動いていなくてもこの理由を返す。
- CIは `requirements/dev.lock.txt` からnumpyを入れ、これまで常にskipしていたCPUテスト7本（NVFP4の逆量子化、agreementのKL）を走らせる。
- 内部の整理。起動引数・環境変数・fingerprint・成功時の記録は変わらない：`server_config` がvLLM引数の雛形を `host` から、imageのcapability検査とfreeze／thawを `server` から引き取り、`host` をimportしなくなった。サイト設定の検査とNCCLの環境は `fabric` へ、`resume` は `switch` へ移した。rankの状態のパス、container名、headのorigin、prefix cacheのリセット、metricsの読み出し、projectorのパス、任意キーの既定値、MTPのdraft設定、LPAのmode keyは、それぞれ持ち主を一つにした。退役した試作5本（`fused_nope`・`fused_nope_dot`・`indexer_candidates`・`indexer_reindex`・`indexer_shared_pool`）は `runtime/` から `validation/` へ移した。`apc-history` の合否の規則、`lpa-fixture` のcaseの判定、Graph起動の数え方、BF16の一致の許容幅はテスト付きのmodule関数になった。`release_notes --match-project` は版をpackage経由で読む。参照Dockerfileのmarkerとpatch、MTPのテンプレート、overlayのhash、切替の語彙は、テストがそれぞれの持ち主に結び付ける。

### Removed

- `lpa-fixture --graphs` と、`glm53_setup/runtime/patch_graph_prefill.py`（decode Graphの下でのLPAのprefill）。これは実行できなかった：profileの検査はdecode Graphと組むLPAを拒み、参照imageはどれもこのpatchを当てず、`--graphs` はどのimageも持たないpatchの記録を確かめるので、読み込みの前に失敗していた。`lpa-fixture` はこのoptionを拒むようになった。engine設定は変わらず、結果は常に `false` の `decode_graphs` を持ち続ける。
- memory probeの `zero_moe_scratch` RPC。呼ぶものは無く、配信workerで呼ぶとprocessが終わるまでMarlin MoEの呼び出しを置き換えていた。repeat traceのfixtureは自前のゼロ埋めの診断を持ち続ける。`validation.memory_probe` はcheckoutのprobeをimageの上にmountするので、次の起動からは古いimageでもこのRPCは無い。
- 内部：`glm53_setup/runtime/patch_pipeline_layout.py` と `pipeline_state.common_layout_names`（参照imageは一度も当てていない）、`glm53_setup/validation/benchmark_attention_graph.py`（呼び手なし）。

### Documentation

- 2系列のbatch不変性の調査を閉じた。2026-09-26に配信中の対で、decode規模の呼び出しでMarlin MoEの分け方を固定し、shared expertとrouterのGEMMを系列ごとに計算し、FA2を切って試した：スイッチは効いたが、2系列のcompletion 8本すべてが単独のものと違ったままだった。不採用。反復性は `max_num_seqs = 1`（配布既定）のまま、公開した任意設定は2系列を保つ。施策台帳（P26）とbenchmarksの到達性の節にそう書いた。
- 構成：memory probeに全methodを挙げた独立の行を置き、`examples/` の行は二つのprofileを挙げ、層の表に `switch.resume`・capability検査・base imageの判定を足して `server_config` が副作用moduleをimportしないことを書いた。新しい節で、worker拡張の置き場所（配信の起動が読み込むものは `runtime/`、fixture専用のworkerは `validation/`、`expert_worker` が例外である理由）と、`python -m glm53_setup.validation.<module>` としてだけ走るrunnerを述べた。
- launcherの文言：`server --help` は "a serial TP=2 reference experiment" ではなく配信のlauncherを説明し、監視のbannerは止まる理由にengine stallを挙げ、`tools/check_prefix_cache.py` の使い方の行は再び一つのコマンドになった。
- exampleのコメント（英語と日本語の行を対で）：新規起動には `GLM53_MOE_ORDER_API=2`（1は起動済みの対の復旧だけ）、`decode_graphs` は不採用（P06）と明記、memory probeで読めるもの、component workerの範囲、torch 2.13だけでなく2.12.1でも起きるDynamoの状態復元、FA2以前の数値の代わりにbenchmarksを指すprefillの時間。どちらのファイルもTOMLとしては前と同じに読める。
- 1.6.0〜1.16.0が追い越していたコードのコメント：参照attentionとそのpatchは検証専用の予備ではなく配信での役割を述べ、memory probeのdocstringは全methodを挙げ（mountするファイルではコメントだけ）、`server_config` はMTPの深さ2・4・5を未実測と呼ばず、不採用のPPとEPの先送りの注記を持たない。

## 1.16.0 — 2026-09-26

### Removed

- `runtime.mla_decode_cpb` を退役した。servingでは一度も実行されていなかった：imageは `GLM53_REFERENCE_ATTENTION=1` を設定し、sparse MLAのforwardは、DSAの11層でもMTPのdraft層でも、keyのpatchが変えるFlashInferのdecodeの呼び出しより前に参照のNoPE attentionを通ってreturnする。1.14.0のkernelの根拠は、このflagを切った単体の結果だった。2026-09-26の到達性の調査では、配信中の対のtraceがsparse MLAの呼び出しを単独の要求で434、2本の組で448数え、すべて参照attentionを通っていた。keyが2系列のdecodeの分け方を固定する、要求のattentionがstepを共有する他の要求に依らなくなる、という1.14.0の記述は撤回する。1.14.0の測定（単独のcompletion・decodeの速度・NLLが不変、2系列のcompletionが1.13.0とbyte一致）は、効果が無かったことと整合する。両方のexampleからkeyを外した。keyを持つ既存のprofileは、環境変数・fingerprint・検査とも変わらずに起動し（`true` なら今も `GLM53_MLA_DECODE_CPB_API=1` を要求し、decode Graphとは排他）、`server preflight` は `warnings` に `mla_decode_cpb_retired` を記録する。patch・helper・imageのmarker・keyの受理は、次のimageのbuildで外す（[起動設定](docs/server-configuration.ja.md#再現性のスイッチ)）。

### Documentation

- attentionの経路の記述を正した。文書はdecodeが参照経路のままだと書いていたが、参照attentionはquery行が6を超える呼び出しをFA2へ送る。1系列のdecodeのstepは参照経路のままだが、MTPの深さ3では相方がいると検証stepが8行になってFA2を通り、ある行の結果が相方の行数と長さで1 BF16 ulp動く。これは2系列の差の主因ではない：同じ調査で配信中の対のattentionの呼び出しを全部eagerにしても、2系列のcompletion 8本のうち7本が単独のものと違ったままで、容疑者の先頭はMoE。運用の結論は変わらない：`max_num_seqs = 1` ならどんな負荷でもcompletionは反復し、2系列では反復しない（[測定](docs/benchmarks.ja.md#servingでの到達性2026-09-26)）。起動設定・検証・README・施策台帳（P26は退役、P05）・最適化の概観・運用・構成、両方のexampleと `glm53_setup/runtime/fa2_attention.py` の `fa2_attention` のコメント（コメントだけ。このファイルはimageの上にmountする）をそのように直した。
- benchmarksの1.15.0：参照対でのCPU配置（2026-09-26）。公開しているdecodeの数値はすべて両rankのworkerが高性能コアにいるときのもので、どちらかのrankが高効率コアにいるとdecodeは約3分の1に落ちる（completionと受理長は同じまま）。`nodes[].cpuset_cpus` で防げる。READMEの見出し表と起動設定からそこを指す。
- 文書を1.15.0に合わせて更新し、短くし、事実ごとに正典を一つにした。後の版が追い越していた記述を現状に直した：同時2系列profileの受け入れ（2026-09-23）、「検証中」ではなく1.12.0と1.14.0の再現性のスイッチ（後者は上で退役）、任意の実験ではなくテンプレートのMTP k=3、decode GraphsとExpert Parallelは測定して不採用、FreedomBenchは閉じた、ハーネスの経路は決定済み、「この版から作ったimage」はすべて版番号に置き換えた。
- 起動設定を組み直した：配布既定と公開オプションを先に置き、続いてキーの解説（再現性のスイッチ、attention・cache・checkpoint、並列化、画像入力、APIと診断〔memory probeのmethodは表〕、prefix cacheと併用するLPA）、最後にコマンド。image契約は、すべてのcapability markerについて、preflightがそれを要求する設定と、imageがそれを持つ版を表にし、文書一覧の正典表にも載せた。訂正が二つ：MTPの深さ1〜5を測ったのは再量子化したcheckpointで、固定checkpointは1・3・4（両方で5つすべてではない）。LPAは `runtime.fa2_attention` と両立せず、MTPとは深さ1か3でだけ組める。
- LPA：有効化の手順で `runtime.fa2_attention = false` を設定するようにした。テンプレートはFA2を有効にしているので、これが無いとlauncherがprofileを拒む。
- 検証：同時実行の範囲をprofileごとの表にし、証拠と反復性の規則を添えた。フルモデルの節に小節（読み込み・API・ベンチマークの確認、マルチバイト出力、再現性）を足した。起動の三つの数値状態の説明は一段落にし、数値はbenchmarksの1.9.0、探索の手順はこの変更履歴（1.10.0〜1.12.2）に置いた。
- benchmarksの冒頭に版ごとの節の索引を置き、施策ごとの判断の段落は台帳を指す結果の一行にした。最適化の全体像・施策台帳・性能調査は役割を保ったまま、数値と判断を書き写さずに正典を指すようにした。ハーネス、FreedomBench、画像入力、NCCL、ライセンス、部品検証、投機decoding、APC優先LPA、候補順、indexer再利用も同じ方針で短くした。
- セットアップ手順の受け入れ証拠のうちメモリの行は、配布既定の受け入れの下に公開オプションの数値を引くのをやめ、1.8.0で同じ夜に測った両profileの測定を指すようにした。
- 2026-09-22の受け入れを、実際のとおりに書いた：両profileの同時1系列を、2026-09-23の生成AIなんでも展示会#6での公開に向けて受け入れ、以後も同じ範囲で通常運用として続けている（README、セットアップ手順、検証、運用）。

## 1.15.0 — 2026-09-26

### Added

- `nodes[].cpuset_cpus`（任意、rankごと。circlemouthさんの#2による）：rankのコンテナ作成時に `--cpuset-cpus` へ渡すDockerのCPUリスト。#1はBIZ 1.11.2の対で、同じ2.4 GHz上限・出力hashとMTP一致率が一致した条件のまま、rank 0を高性能コアに置くと短い入力のdecodeが36.13 tokens/s、高効率コアでは17.01と測った。launcherは形式の不正な指定と逆順・重複した範囲を拒み、`server preflight` は指定CPUがlauncherのaffinityの範囲にあるかを確かめ、`docker run` の後は `HostConfig.CpusetCpus` が同じCPUを指すときだけrankを起動済みとして記録し、違えば新しいコンテナを止める。未指定ならDockerのコマンドは従来どおりで、コア番号はホストごとに違うため両exampleともkeyはコメントのまま（[サーバー設定](docs/server-configuration.ja.md#cpu配置の任意指定)）。参照対のrank 0のホストは2.8 GHzのコア10個（0〜4、10〜14）と3.9 GHzのコア10個（5〜9、15〜19）を持ち、2026-09-26に見たときはCPU setなしでworkerが3.9 GHzのコアにいた。固定あり／なしの比較はここでは測っていない。

## 1.14.0 — 2026-09-26

### Added

- `runtime.mla_decode_cpb`：sparse MLAのdecodeが、FlashInferの `chunks_per_block` をstep全体ではなく1系列あたりのtoken数から決める（draftのstepは2、深さ3の検証stepは6。rankあたり32 head、top-k 2,048）。FlashInfer 0.6.18はstepのtoken数から値を選ぶため、2本目の系列が来ると要求の候補の足し方が変わっていた（GB10一台で、要求のattentionの全行）。固定値は1系列のときのその値そのものなので、単独の要求は従来とbit単位で同じに計算する。keyは両exampleでon、1.14.0以降でbuildしたimageが必要（marker `GLM53_MLA_DECODE_CPB_API=1`、preflightで必須。SM120 backendへのsource固定patch `glm53_setup/runtime/patch_mla_decode_cpb.py`）、`runtime.decode_graphs` との併用は拒む。参照対では両profileとも、単独のcompletion、decodeの速度、NLL、長文の答えは不変（[測定](docs/benchmarks.ja.md#1140での測定)）。reference imageを作り直して `reference_image` を更新するか、keyを `false` にする。

### Documentation

- 複数系列での反復性を両方の重みで測った：他の要求とstepを共有した要求は、NVFP4のMarlin MoE（K方向の分け方がstepのexpert block数で決まる）と、prefillと共有したstepのprefill用のkernelを通して、なお変わる。どんな負荷でも反復するのは `max_num_seqs = 1` のときだけで、同時に送った要求は待ち行列に入る（18本中18本が単独のcompletionと同じ）。要求が重なるときの処理量は2系列の方が約4分の1多い。検証の同時実行の範囲、READMEの行、サーバー設定に書いた。
- 4,608トークンのcache block一つより長いpromptを、他の長いprefillの前後でprefix cacheから再利用しても、毎回正答し同じcompletionになる（1.13.0のkpool seedの修正）。
- 上流へ報告した：flashinfer-ai/flashinfer#5553（decodeの分け方が呼び出しのtoken数で決まる）と、vllm-project/vllm#46639（Marlin MoEのbatch不変化）へのNVFP4の実測。施策台帳にP26を足し、採らなかった案も記録した。

## 1.13.0 — 2026-09-25

### Fixed

- 参照imageは、固定しているvLLMのkpoolのprefill seed kernelにpatchを当て、indexerの生tailのblockをtail自身のstrideで番地付けするようにする（marker `GLM53_KPOOL_SEED_STRIDE=1`、`glm53_setup/runtime/patch_kpool_seed.py`。vLLMのpull request #57477と同じ変更で、上流では2026-09-20にmerge）。tailはindexerのcacheを、indexerの詰め物入りのblock strideで共有しているが、vLLM 385dce36のseed kernelは詰め物なしで番地を計算していた。prefillのたびに要求自身のtail blockがseedされず、最後のtokenの生のkeyとgateが番号の小さいindexer blockへ書き込まれる。そのblockは別の要求のものであり得る。壊れるのは長い文脈での疎なtop-kの選択で、prefix cacheから再利用する長いpromptで最も効く。MLAのKVは無傷。上流の回帰テストは1.12のimageでfailし、作り直したimageでpassする。基準の2台では、測った出力は一つも変わらなかった：decode検査、教師強制のNLL、同時2系列のcompletion、配布既定の重みのdigestは前後でbit単位で同一で、長文の検査は両profileで正答した（[実測](docs/benchmarks.ja.md#1130での測定)）。参照imageを作り直して `reference_image` を更新すること。このmarkerを要求する検査はなく、固定しているvLLMが上流の修正より先へ進んだらpatchは外す。

### Documentation

- 同時2系列：サーバが2本を同じ順でprefillすれば、対のcompletionは反復する。1台のGB10でのkernel単体の試験では、NVFP4のMarlin MoEが、別の要求のdecode行と同じ呼び出しに載るだけで行の結果を変える。cuBLASのBF16 GEMMはbitを保ち、Marlinのdense GEMMはprefill規模の行数でだけ変わる。kpool seedの修正は原因ではない。検証の同時実行の段落にそう書いた。
- 3か所参照の正典を、記事を背景として囲み `=` の後の値を問う問題文にした。両profileとも3回とも正答し、READMEの行もこの結果にした（問題文はベンチマークに掲載）。古い問題文は今回も両profileで分かれた（AXLは3回とも記事を書き写し、配布既定はコードを返した）。
- READMEの公開した任意設定のNLLを、`runtime.inductor_deterministic` 付きの同時2系列profileの値にした（1.6270／1.9946／0.9601／0.6275。1.12と1.13のimageで同一）。同時1系列profileの1.6645／2.0024／1.0031／0.6279より4文とも低い。短所の行は「配布既定より4文のうち3文で1〜6%上がる」とした。
- このhybridモデルではprefix cacheは4,608 tokenの丸ごとのblock単位でしか当たらない。修正の効く範囲を述べる箇所にそう書いた。

## 1.12.2 — 2026-09-25

### Documentation

- 検証の多バイト文字の段落に、tonyd2wildのcheckpoint guard（`abb38bb`）が、attentionを量子化したModelOptのbuildを多バイト文字が化けるbuildとして拒否することを書き足した。vLLM #54150について、本リポジトリの書くgate／upのscale不一致とは別の読みになる。BIZ AXLはattention射影を量子化している（ModelOptではなく自前のW4A16 repack）が、`server mojibake` に2026-09-21と2026-09-25に合格した（6回答すべて、置換文字・サロゲート・制御文字なし）。6回答では二つの読みのどちらが正しいかは決まらない。
- 最初にcompileしたframeの後で `TORCHINDUCTOR_DETERMINISTIC` を切るDynamoの状態復元（pytorch/pytorch#198563）は、torch 2.12.1でも起きる。GB10での報告（vllm-project/vllm#58636）があり、PyTorchのissueの再現例でも確認した。サーバー設定と検証の記述をtorch 2.13に固有のものとして書かない形に直した。下の1.12.0の項は、imageが載せているtorchについての記述。

## 1.12.1 — 2026-09-25

### Documentation

- `runtime.inductor_deterministic` を付けた配布既定（出荷どおりのテンプレート）を3回起動した：両rankがindexerのkey正規化を一つのconfigで動かし、すべての呼び出しで同じkeyを受け取り、rank間で違うInductorのconfigはなく、3起動とも同じcompletionで、decodeは配布既定の公開値と同じ（count／prose／codeで32.80〜32.84／20.76〜20.82／27.53〜27.64 tok/s、公開値は32.01／20.67／26.68）。検証・ベンチマーク・READMEの反復性の行から「配布既定はkeyを付けて起動していない」を外した。

## 1.12.0 — 2026-09-25

### Added

- `runtime.inductor_deterministic`（両方のexample profileでon。未指定か `false` はInductorの計測による選択のまま）：Inductorがreductionのconfigを計測せずに両rankで同じものに決め、専用のcacheにcompileする（`TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor-deterministic`）。複製されたkpool indexerはkeyを候補configが三つの `torch.compile` のleafで正規化する。各rankが起動のたびに計測で一つを選び、そのうち一つは行の足し算の順が違う。両rankが別のclassを引くと、poolが同点になる所でcompletionが分かれていた。keyを付けても、torch 2.13のDynamoは最初にcompileしたframeの後でモードを切ってしまう（[pytorch/pytorch#198563](https://github.com/pytorch/pytorch/issues/198563)）。そこでlauncherは `glm53_setup/runtime/inductor_pin.py` と一行の `.pth` もmountし、設定を強制値で保つ。imageの対応は要らない。keyを付けた最初の起動ではindexerのleafを作り直す。keyを足すとprofileのfingerprintが変わる。詳細は[サーバー設定](docs/server-configuration.ja.md#配布用の既定設定)。
- memory probeが、配信中の各workerで実際に何が動いているかを読む：`autotuners`（Inductorの各kernelのファイル・size hint・使っているconfig・それがcacheからかこのprocessでの計測か）と `inductor_state`（Inductorの設定、`TORCHINDUCTOR_*` の環境変数、強制値、決定性スイッチへの書き込みと呼び出し元、その場での一回のcompile）。`tools/kernel_hashes.py` はその両方をhashの前後でrankごとに記録する。indexerのstage hashは配信の呼び出しをstrided viewのまま動かし（`k_norm_served_rows*`）、rotaryをMLAのwrapperからも探す。

### Fixed

- **起動が三つの数値状態のどれかに落ちることがなくなった。** これまで対は起動を跨ぐと三つの状態のどれかで計算していた（状態を確かめた配信imageの16起動で11・3・2）。原因はrankごとに計測されるindexerのcompileされたkey正規化（Addedを参照）。しかもその計測は起動のたびにやり直されていた：永続のInductor cacheには2026-09-15にcontainerの `/tmp/torchinductor_root` から写したgraphがあり、選択をそこに保存し、そこから探していたため。決定性のkeyは新しいgraphを専用のcacheにcompileする。keyを付けた3起動はすべて状態1で同じcompletionになり、decodeの速さは以前の状態1の起動の幅の中だった。 同じ型の問題を [vllm-project/vllm#58636](https://github.com/vllm-project/vllm/issues/58636) としてvLLMに報告した。

### Documentation

- 検証（起動の数値状態：原因、各状態を再現した介入、修正）、ベンチマーク1.9.0、READMEの反復性の行と引用の版、サーバー設定とアーキテクチャ（`inductor_pin`）。

## 1.11.3 — 2026-09-23

### Documentation

- README、主要な測定値：表に分類の列を足し、prefill・decode・長文入力・品質・反復性・メモリの順に並べ替えた。数値は不変。同一要求の状態行は配信imageの13起動が示したことを言う：同じ起動の中ではbit一致、起動を跨ぐと三つの数値状態（9・3・1。2026-09-23の5回の切替のうち3回で状態が変わった）、切替ごとの検査、状態の差が乗る一箇所。
- 検証とベンチマーク1.9.0：Probe5の起動（状態1、1.11.1のcheckout）は16のkernel hashとindexerのkeyのすべての呼び出しで両rankが一致し、状態2の起動と比べるとrank 0のprefillは同一でrank 1のkeyが最初のMLA層から違う。よって起動の状態とはrank 1のprocessがindexerのkeyをrank 0と同じbitで作るか否かで、決め手は状態2の起動でのcompiledのnormのhash。

## 1.11.2 — 2026-09-23

### Documentation

- **公開した任意設定の同時2系列profileを通常運用として受け入れた（2026-09-23）。** 範囲は同時2系列・1要求あたり約200K tokenまで。根拠は同日のこのprofileの3起動（状態2・1・2）で、各起動が切替後の重みのdigest・decode検査・traceを通り、3起動目はkernel hashも両rankで一致。タスク単位の検査は1起動目、sparkDashとtool-evalは2起動目で走り、3回の切替は復旧なしで完了。判定と範囲の正典は検証範囲の「同時実行の範囲」、SETUP手順6に証拠の表、READMEの要約と状態の行、運用手順も同じ内容。同時2系列でcompletionが変わることは宣言した挙動のままで、受け入れた欠陥ではない：同じstepを共有する他の要求からcompletionを独立にするsource固定patchが書けるかは、起動状態の追及と併せて検証中。

## 1.11.1 — 2026-09-23

### Added

- `kernel_hashes` が、indexerの射影とopの間の段階を一つの層の実際の重みでもhashする：compileされたlayer norm（`_fused_indexer_k_norm`）とそのeager fp32の参照、indexerのrotary embeddingのqueryとkey、keyの射影（`layer`、既定19）。1.11.0の起動（状態2）は固定入力のkernelで両rankが一致し、入力まで取った3段のtraceは同じ射影の出力から両rankの複製されたindexerが8つのMLA層のprefillで違うkeyを受け取ることを示した。keyが作られるのがこれらの段階である。この部分の失敗はmethod全体を失敗させず、答えの中で名乗る。

### Documentation

- セットアップ手順書（冒頭の状態と手順6の判定）と運用手順が、READMEと検証範囲と同じ言い方で同時実行を述べる：配布既定では同時1系列が受け入れた範囲、公開した任意設定の同時2系列profileは1起動でタスク単位の検査を通し、通常運用は起動の積み上がりを待つ。正典は[同時実行の範囲](docs/validation.ja.md#同時実行の範囲)。
- ベンチマークと検証範囲の日本語版で、「証拠であり、本番認定ではない」節を指すアンカー2件が見出しと違う文言（本番の検収）だった。見出しに合わせた。
- 検証とベンチマーク1.9.0：Probe4の起動と深いtrace、残る二つの候補（Inductorのlayer norm kernel、rotaryのkey側）。

## 1.11.0 — 2026-09-23

### Added

- **memory probeが配信workerの中でindexerのkernelをhashし（`kernel_hashes`）、`tools/kernel_hashes.py` が両rankと前の起動を比べる。** 2026-09-23に、複製されたkpool indexerの一方のrankの写しが、同一入力で二つの起動の間で最初に違った呼び出しであり、GB10 1台の新しいprocess 13本ではindexerのkernelがbit一致したので、差は配信processそのものにある。methodはfp32のhead gateとbf16のgate score、融合したFWHT量子化、pool cacheのprefillの書き込みとdecodeのtail update、DeepGEMMのpaged MQA logitsとstable top-kを、配信の形状の固定入力で走らせ、全出力のhashを返す。toolは両rankに問い、互いと前の起動の記録とを比べ、差があれば終了状態1でkeyを名指しする。切替ごとに重みのdigestと並べて取れば（[起動安全](docs/launch-safety.ja.md#切替の後のdecode検査)）、次の別の状態の起動がその場で計算を名指しする。probeの他のmethodは不変。

## 1.10.5 — 2026-09-23

### Documentation

- README、主要な測定値：公開した任意設定の列を参照対が配信するprofile＝同時2系列のAXL profileの2026-09-23の測定にした（2,048 token後のdecode 45.04／28.16／37.67 tok/s、約200Kの合言葉165.7 s・2本同時330.2 s、sparkDashとtool-evalの行、KV 6 GiBでのメモリ）。その夜に測り直していない行は最後の測定値を日付つきで残す。段落は旧版を列挙しない。
- 検証とベンチマーク1.9.0：2026-09-23に名指ししたindexerの四つの計算（fp32のhead gate、融合したFWHT量子化、pool cacheの書き込み経路、DeepGEMMのpaged MQA logitsとstable top-k）をGB10 1台の新しいprocess 13本で配信の形状で回し、bit一致した。よって起動状態の差は配信processそのものの状態に属し、次の測定は稼働中の両rankでそれらを走らせるprobeのmethod。

## 1.10.4 — 2026-09-23

### Documentation

- CUTLASS W4A4のfixtureの行をREADMEの状態表と検証の証拠の表から外し、その差を「切り分けが必要な観測」と呼んでいた文も消した。この経路は最初の週に4層fixtureで一度試しただけで配信したことはなく、以後のprofileはすべてMarlin W4A16で走る。現在形の行はスタックの品質の未解決の欠陥として読まれていた。経緯はその週の変更履歴に残り、精度の段落とSETUPは引き続きW4A4の挙動を主張しないと言う。
- 検証とベンチマーク1.9.0：二つの起動状態の差を名指しした。2026-09-23の別の状態の起動は同じ重みをload（digestが両rankで等しい）し、要求のtraceが最初に違う呼び出し——rank 1のlayer 19の複製されたkpool indexer、decodeの検証step、散文とコードの両要求で入力同一・候補集合が違う——を名指しした。前の起動ではそのrankのindexerがまさにその呼び出しでrank 0と食い違っていた。二つの起動の差は複製されたindexerの一方のrankの写しがほぼ同点を違って採点したことで、三つ目の状態は数え上げ要求でしかtraceしていない。processを跨いでindexerの採点の内側か書いたpool cacheの何が違うかが次の測定。
- ベンチマーク1.10.4：sparkDash DecodeBenchとtool-eval-benchを参照対が配信するprofileで変えずに繰り返した。sparkDashのdecodeは4つのpromptで48.2／31.4／41.3／34.9 tok/s、1.5.0は36.2／26.7／31.7／26.3で、差はそれ以後の配信profileの伸び。tool-eval-benchは88／100（122／138）、55 pass・11 partial・3 failで、failは1.0.0と同じ3件、Safety GateはTC-43で未達のまま。READMEの主要な測定値は両方を指す。

## 1.10.3 — 2026-09-23

### Added

- testが両方の例のdocker commandと環境変数をrankごとに同じimageを与えて組み立て、差を公開した任意設定の5設定（derived checkpointのmount、そのoverlay 2枚のmount、dedupの環境変数、2系列目、2倍のKV）と、1点目から従う2つの引数（model引数、fingerprintのlabel）に固定する。

### Documentation

- 起動設定：公開した任意設定と配布既定の差を、5設定と各々が起動に加えるものの表で。2026-09-23の確認——AXLの例の仮値を参照対の値に置き換えて `server freeze`・`server plan` と両rankのpreflightの全検査を通した（配信中の対では成り立たない `startup_memory` を除く）。例と配信profileの差はprobe・dev経路・warmupの長文段・未使用のLPA仮値だけ。
- 検証、同時実行の範囲：公開した任意設定の同時2系列profileは2026-09-23の1起動でタスク単位の検査（約200Kの合言葉2本の同時、tool呼び出し2本の同時、画像と散文の同時）にpreemptionなしで合格し、同時2系列のcompletionはbatch-invariant modeが使えないため単独時と一致しない。通常運用としての受け入れは起動の積み上がりを待つ。数値はベンチマーク1.10.2（200K 2本で330 s・単独166 s、同時2系列のdecode 32.1／21.8 tok/s・単独45.6／28.2、KV最大54%）。READMEの状態の行・同時実行の行・主要な測定値の段落も同じことを言う。起動安全：切替後のdecode検査は同じ理由で他の要求が走っていない時に取る。KV容量の節は2,112 tokenの評価を「受け入れた同時2系列の範囲」と呼ばなくなった。

## 1.10.2 — 2026-09-23

### Added

- **`examples/server.axl.example.toml`、公開した任意設定のprofile。** 配布既定は `examples/server.example.toml` のまま。任意設定の例は、再パックした重み（NVFP4 BIZ AXL）とリポジトリ同梱のoverlayの `runtime.derived_checkpoint` 表、`runtime.prefix_page_dedup`、同時2系列（`max_num_seqs = 2`）、rankあたり6 GiBのKVを足す。再パックした重みは固定の重みよりrankあたり4.4 GiB軽く、それが2系列目の財布になる。2026-09-23に参照対で初起動：KV 606,881トークン（engineの数えで256K要求の2.32倍）、warmup合格、同時2要求を配信（両方走行中でcounting 32.2・prose 22.3 tok/s）。testが両方の例を有効に保ち、共通の節が等しいことを固定する。
- **ランチャーは再パックしたcheckpointなしで3 GiBを超えるKVを拒む**（`check_kv_budget`）。参照対では固定の重みがKV 3 GiBでheadに5.5 GiBを残し保護は3 GiBなので、KVを倍にするとcontainerが止まる。再パックした重みは10.5 GiBを残す。拒否の文がそう言う。

## 1.10.1 — 2026-09-23

### Fixed

- `weight_digest` が0次元のparameter（scale）で「self.dim() cannot be 0 to view Float as Byte」を出していた。fingerprintはtensorをbyteとして見る前に平らにする（`flat_bytes`）。他のtensorのprintは変わらない。probe付きの最初の起動で参照対にて計測。
- `tools/weight_digest.py` はリポジトリのrootを `sys.path` に置くので、文書どおり `python3 tools/weight_digest.py` がどのディレクトリからでも走る。

### Documentation

- READMEが要約で始まる：スタックが何か、受け入れの状態と範囲、配信する二つのprofile、動作する精度、ライセンスの形、未検証のもの。それぞれ所有する節へのポインタで、置き換えた導入の二段落を吸収し、実測値と版数を持たない。
- 検証範囲に、NVIDIAのモデルカードはこの配信を記述しないことを明記：そのBF16対NVFP4の表はGB200上でvLLMとSGLangを通しカードのW4A4 recipeで測ったもので、このスタックはGB10上のMarlin W4A16で動く。この配信を記述する数値を名指しした。READMEの精度の段落はそこを指す。
- 文書一覧：GitHubのdescriptionとtopicsはREADMEの要約を実測値と版数なしで言い直したもので、要約を変えたら `gh repo edit` で揃える。

## 1.10.0 — 2026-09-23

### Added

- **memory probeがloadされた重みをfingerprintし（`weight_digest`）、`tools/weight_digest.py` が起動を跨いで記録・比較する。** 配信モデルとdraftの全parameterとbufferを、各rankにloadされたままの姿で、要求traceと同じ二つのbyte和でfingerprintする。記録は行と層ごとのdigestを持ち、後の起動の記録は違うtensorを名指しする。別の数値状態の起動（[1.9.0](#190--2026-09-22)）にまず問うべきこと——同じbitを違う計算で処理したのか、違うbitをloadしたのか——に答える道具。`trace_end` に `export` が加わり、traceの行を返すので `trace_differences` で二つの起動を記録どうしで比べられる。共通のfingerprintとmodel探索はmodule関数に移した。traceの挙動は不変。

### Documentation

- 起動の安全：profileが `validation.memory_probe` を持つときの、切替の後の検査としての重みのdigest。起動設定と構成表がmethodとtoolを載せる。README：FreedomBenchの行は「検収は未了」でなく配信profileで完了と言う。

## 1.9.5 — 2026-09-23

### Documentation

- FreedomBenchを配信profileで完了：2026-09-22に固定の英語原版60問が初回で60問正解・拒否ゼロ、長文付きpilotが6問中6問、いずれも参照対が配信するprofileで。4構成の一覧とLPAの項目は配信profileが一つになったことで退役し、未実施（日本語訳・対立的な言い回し・証拠配置）を列挙した。検証一覧も一文で同じことを言う。
- benchmarks 1.9.0：起動状態の問いを絞る二つの検査——対のGEMMとKDAのkernelを13の新しいprocessで計算してbit一致（固定env込み）、4層のMTP fixtureを対にTP=2で20回起動してcompletion同一——により、残る候補はフルサイズのモデルのloadと起動に絞られた。

## 1.9.4 — 2026-09-22

### Changed

- image buildで当てるsource固定のvLLM patch群が一つの骨格 `glm53_setup/runtime/pinned_patch.py`（固定fileのhash検査、`--package`／`--check` コマンド、package脇に書くrecord）を共有するようになった。各 `patch_*` moduleは対象・pin・anchorだけを純粋関数 `patch_text(text)` として述べるので、7本すべてが固定fileの抜粋でテストできる（以前は3本ができなかった）。Dockerfileが走らせる入口・旗・印字するhash・recordファイルは変わらず、契約テストがDockerfileを読んでそれを固定する。

- `apc-lpa-fixture` がfixtureを `fixture-run` と同じ門（`run_fixture.read_fixture`）で読むようになった。byte検証済みの4層test fixtureでないものは、その門の一文 "Only a byte-verified four-layer test fixture is allowed" で拒み、`all_tensor_bytes_verified` は真らしい値でなく `true` そのものを要求する。engine設定・測定の条件・報告は変わらない。

## 1.9.3 — 2026-09-22

### Documentation

- 2026-09-22の通常運用の受け入れが追い越していた記述：リポジトリ指示（`AGENTS.md`）、運用手順、検証範囲は、TP=2が未検収であるとも、ZCode DesktopとClaude Codeが必須対象であるとも言わなくなった。受け入れと各項目の証拠の所在の記録であるSETUP手順6と、ハーネスの判断を指す。セットアップのチェックリストとAIへの依頼例は、撤去した `service` launcherがかつて門にしていた「証跡生成手順」を求めず、手順6の項目とその証跡パスを求める。運用手順と検証範囲は日本語でも同じ読みになる。
- 構成：モジュール表が `glm53_setup/`、`glm53_setup/runtime/`、`glm53_setup/validation/`、`tools/` の全モジュール（1.7.0〜1.9.0のsource固定patch、FA2経路、一致・再量子化の検査、decodeの道具、`release_notes.py`）と、`overlays/`・`.github/workflows/` のディレクトリに名前を持つ。「vLLMのソース2ファイル」と数えていた文は、build時patchの全部と起動時のoverlay照合を述べる。
- 文書一覧：overlayの台帳（出所記録として英語のみ）とZCodeガードhookの導入手順（英日の対になった）に行がある。施策台帳の行は施策IDを列挙しない。切替の後のdecode検査と通常運用の受け入れは、正典の所在の表に所有者を持つ。
- `tools/check_publication.py` に検査4件（それぞれ単体テストつき）：READMEの略称の引用は `pyproject.toml` の版を持つこと、各文書一覧は自分の言語の `docs/*.md` を全部リンクすること、`glm53_setup/` と `tools/` の全モジュールが構成の頁にファイル名か `benchmark_*.py` のような型で名前を持つこと、`--plans` で追跡外の `docs/plans/` の相対リンクが実在すること。このtreeでは監査は通る。

## 1.9.2 — 2026-09-22

### Added

- `tools/decode_check.py` と `tools/decode_divergence.py`：切替の後のdecode検査（固定の約2,048トークンのpromptの後にgreedyで512トークン、課題3種、`return_token_ids` でcompletionの本文とtoken idを残す）と、二つの起動を最初に分岐したtokenで比べる道具。1.6.0からのbenchmarksのdecode行と同じ測定を、recordsディレクトリでなくリポジトリに置いた。

### Documentation

- 起動の安全：「切替の後のdecode検査」。1.9.0の起動状態の発見が求める定型（起動の中で3標本が一致、hashは同じprofileの前の起動と比べる、違ったらtoken idと両rankのcontainer logを残す）。
- benchmarks 1.9.0：2026-09-18からのdecode記録で見た別の状態の稀さ（一つのprofileのcompletionは日を跨いで繰り返し、一度しか見ていない二つの状態はどこにも再出しない）と、vLLMのbatch-invariantモードが `CUBLAS_WORKSPACE_CONFIG` を自分で設定すること。このスタックは別の状態の起動がBF16 GEMMを指すまで未設定のまま。

## 1.9.1 — 2026-09-22

### Documentation

- 略称。スタックと公開した重みをそれぞれ一語で引用できるようにした：**NVFP4 BIZ** はこの配信スタック（引用は「NVFP4 BIZ 1.9.1」）、**NVFP4 BIZ AXL** は公開した任意設定の重み（attention projectionと `lm_head` をW4A16にしたもの）。README・GitHubのdescription・Hugging Faceのmodel cardに載せた。リポジトリ名・パッケージ名・本文の用語（配布既定、公開した任意設定）は変えない。

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

## 1.5.0 — 2026-09-18

### Changed

- **テンプレートが画像入力を256Kで配信する**：`context.max_model_len = 262144`（従来204800）、`cache.kv_cache_memory_bytes = 3221225472`（rankあたり3 GiB、従来2.5）、`resources.reserve_gib = 3.0`（従来2.5）。1.3.1のNCCL 8チャネルで、200Kの要求中もheadの空きは6.40 GiB以上残っていたので、KVを0.5 GiB増やしても、保護とsupervisorの1.6 GiBの行き過ぎに対して約5.9 GiBが残ると見込んだ。参照対での実測は、headで起動中5.89 GiB、255,950 tokenの要求中5.82 GiB、peerで8.44 GiB。poolは301,645 tokenを収める（84 block、全長の要求1件あたり73、1.15×）。262,080 + 64の要求は488.6 s、最長の256Kの要求は492.5 sで、600 sのtimeoutに収まった。prefillは569.8 tok/sで、画像・tool・動画の検査は合格した。新しいprofileの200Kでは、速度は1.4.0と同じで、各rankの空きメモリは0.5〜0.7 GiB少なかった。変更のないprofileは200Kとfingerprintを保つ。先にKV 4 GiBの300K profileを見積もったが、実行していない：同じ計算では余裕が0.3 GiBしか残らず、全長の要求は570 s近くになる。テキスト専用の代替は残し、同じ長さとKVにした。

### Documentation

- ベンチマーク：1.5.0での測定（起動時のpoolとメモリ、prefillとdecode、256Kの合言葉、容量と3か所参照の要求、1.4.0と比べて繰り返したsparkDashと200Kの要求）。3か所参照は256Kで3回中2回正答し、200Kでは容量の要求の後にまた失敗した。1.4.0と同じ読み違い。
- 起動設定：測定した二つの画像profile構成の両方で、KV poolがblockをどう数えたか（1 GiBあたり28 block、全長の要求1件あたりceil(L / 4608) + 16）。これで切替の前に256Kのpoolを予測できた。テキスト専用の代替は既定からvision towerを除いたものと説明し、その3 GiBの保護は未検証とした。
- 画像入力：256Kの設定、それを選んだ理由、新しいprofileでの検査。README、運用、ハーネス、LPA、施策台帳と最適化の全体像も256Kの既定に合わせた。
- ハーネス：ストリームのidle timeoutの段落が、試験した設定を60000 msとも700000 msとも書いていた。正しくは700000。

## 1.4.0 — 2026-09-17

### Added

- **他のコンテナのGPU利用に対する起動ガード。** `server preflight`・`server start`・`server assets` に `exclusive_gpu` を足した。このランチャーのラベルを持たない稼働中のコンテナがGPUを要求している間は不合格になり、そのコンテナを `foreign_gpu_containers` に並べる。参照機のホストでは、`--gpus` とCDIの要求はどちらも `HostConfig.DeviceRequests` に現れ、GPUのデバイスノードが `Devices` に現れることはない。このランチャーの対はfingerprintによらずラベルを持つので、稼働中の古い対が `cluster switch` を妨げることはない。空のラベル値は数えず、一覧に載ったままinspectできないコンテナは検査を不合格にする。上書きの手段も、新しいprofileキーもない。着想はsfxnz PR #12（MIT、コードは採用しない）。
- **`MemAvailable` と並べるメモリの読み。** supervisorの各行が、`mem_free_gib` と、2 MiB以上のbuddy blockにある空きメモリ `free_2mib_gib` を記録する。NVRMはそうしたblockを必要とし、ページキャッシュを回収しない。配信中のheadは、availableが7.1 GiBあるのにそうしたblockは0.49 GiBだった。rankを止めるのは `MemAvailable` だけで、読めなかった標本は `memory_sample_error` として記録する。
- **`server mojibake`。** temperature 0で日本語と韓国語の長い回答を3本ずつ取り、置換文字・孤立サロゲート・制御文字を回答とreasoningの中で数える。短い・別言語・空・形の崩れた・失敗した回答は判定不能とする。1.3.1では6本すべて合格した。low effortではreasoningが空で、そちらは試せていない。固定したNVIDIAのcheckpointは、vLLM #54150のgate／upのscale不一致の影響を受けない。
- **attentionの部品プローブ。** `validation.benchmark_sm90_attention` は、FlashInferのMLA paged wrapperを固定のSM90 backendと同じ形で、`fa2` を指定してGB10上で呼ぶ。候補の全幅ですべての数値検査に合格し、512行を3.5 msで処理した（参照は70.6 ms）。ただしBF16 KVだけ：FlashInfer 0.6.18はSM90以外でFP8のMLA KVを拒否する。`validation.benchmark_native_attention --by-row-kind` はSM120のプローブを再検討する：その3.03125は空行から来ており、0にすれば直るが、候補17個の行はなお上限を超える。どちらも配信は変えない。着想はsfxnz、tonyd2wild、drowzeys、Mia（コードは採用しない）。

### Changed

- **テンプレートが `context.max_num_batched_tokens = 2048` を設定する**（従来512）。1系列の200K画像profileで、新規の39K promptのprefillは512／1024／2048で476.9／545.2／563.0 tok/s、199,652 tokenの合言葉の要求は2048で361.3 s、512で410.8 sで、正答した。peerの最小空きメモリは最大0.7 GiB下がり（測定中はhead 6.71、peer 8.77 GiB、保護は2.5 GiB）、2048での読み込みではpeerにNVRMの割り当て再試行が28回記録され、decodeは実行ごとのばらつきを超えては変わらなかった。変更のないprofileは512とfingerprintを保つ。2系列では、P11が1024で測ったとおり、chunkを長くすると最長の停止が長くなる。200Kの3か所参照は、同じstackで2048では3回中1回、512では2回中1回正答した：どちらの大きさでも、モデルが各値をその後に続く記事として読むことがあるので、この検査では両者を区別できない。

### Documentation

- 部品検証：SM120のゼロ埋めの行の種類別の再検討と、新しいSM90 FA2 wrapperのプローブ。施策台帳のP05の行と最適化の全体像もこれに合わせた。
- 運用、起動契約、起動設定、セットアップ手順：他のコンテナのGPU利用に対するガードとそれがいつ走るか、新しいメモリの読みの意味、checksumの実行と大きな読み出しが統合メモリ上でページキャッシュに及ぼす影響、`server mojibake`、他のレシピでの長い要求の崩壊に対する2系列の範囲、将来のcompile済みGraphとMTPの組み合わせに向けたvLLM #53366。
- 検証、ハーネス、投機デコード：多バイト出力の検査と固定のcheckpointが影響を受けない理由、固定のvLLMに既に入っているXGrammarと投機デコードの修正、平均採択長による深さの比較。
- ベンチマーク：200K画像profileでのchunk予算の比較。
- 運用と施策台帳のP10：chunk 2048での `vm.swappiness` 0と60の比較。60ではswapに出たページを持つengineプロセスはなく、0ではswapは空のままで、prefillは再起動ごとのばらつきの中で動き、headの空きは0.45 GiB少なかったので、ホストは60のままにする。
- ベンチマーク：1.4.0（chunk 2048）でのsparkDashと200Kの検査。decodeの中央値は36.51／25.67／29.31／26.29 token/sで、1.3.1と同じ。合言葉の要求は361.4 s、容量の要求は380.0 sで、1.3.1より12%と19%短く、headの空きは一度も6.40 GiBを下回らなかった。3か所参照の実行は、reasoningの長さと、それぞれの直前の要求とともに並べた。
- 運用：2026-09-16のホストメモリの原因をどう突き止めたかと、修正の後に何が保たれたか。Bluetoothとディスプレイマネージャのグリーターは症状だった。新しいプロセスIDの数（10 sあたり6,642対176）、ログインメッセージのスクリプトの実行、SSHのログを数えて、ダッシュボードがメトリクスごとに張るログインに行き着いた。Bluetoothを再び有効にして電源を入れ直し、ダッシュボードが2台のホストを監視する状態で、`polkitd`・`bluetoothd`・`wireplumber` は300 sの間同じ大きさのままで、2026-09-17のダッシュボードの再起動では新しいログインはなかった。
- 運用とテンプレート：timeoutの注記は、測定した最長の要求として200Kの506 sを挙げていた。今はchunk 2048の200Kで380 s、256Kの代替で582 sを挙げる。起動設定は、NCCL 64チャネルで測ったheadの余裕4.0〜4.2 GiBを現在の値としたままだった。今は8チャネルでの6.97 GiBとchunk 2048での6.40 GiBを添える。LPAの有効化へのリンクが1.1.0で改名した見出しを指したままで、英日の対4組でリンクか文言が片側にだけあった。
- ベンチマーク：リリース候補のsparkDashと200Kの検査を1.3.1（LPA off、NCCL 8チャネル、MTU 1500）で繰り返した。decodeの中央値はstructured／prose／code／jsonで36.67／25.71／30.19／26.48 token/s、TTFTは4つとも短くなった。199,652 tokenの合言葉の要求は410.8 s（2026-09-15は506.1 s）、容量204,736 + 64は470.3 s、3か所参照は435.1 sで、すべて正答し、headの空きは一度も6.97 GiBを下回らなかった。実行の間には複数の変更があるので、どれか一つの効果とはしない。
- NCCL検証とQSFPネットワーク：MTU 1500の全モデルの実行ではheadの監視ダッシュボードが動いていて（65 MiB、CPU約2%）、MTU 9000の実行では動いていなかった。1.3.1はすべての実行で止めていたと書いていた。したがって、MTU 9000によるprefillの2.3〜2.6%の向上は上限値。各MTUでのチャネル数の比較は影響を受けず、メモリの代価も同じで、ダッシュボードはそれを小さく見せることしかできない。

## 1.3.1 — 2026-09-17

### Changed

- **テンプレートが `runtime.nccl_channels = 8` を設定する。** 参照対では、NCCLに任せると64チャネルを開き、各engineはcommunicatorを二つ開く。8にすると、MTU 1500で全モデルの最小空きメモリがheadで2.8 GiB、peerで3.0 GiB増え、prefillは1%速くなった。両MTUで64／32／16／8／4チャネルを掃引した2 rankのAllReduceでは、8がメモリと4 MiB（prefill chunk）での最適点だった。キーのないprofileはNCCLの選択とfingerprintを保つ。数値は新しい[チャネル数の節](docs/nccl-validation.ja.md#チャネル数)にある。

### Documentation

- **MTU：** 9000はprefillを2.3〜2.6%上げる代わりに、ホストあたりの空きメモリを1.1〜1.7 GiB減らしたので（NICの受信バッファ。モデルを動かしていないホストでも同じ1.4 GiBの差）、参照対は1500のまま（QSFPネットワーク、NCCL検証）。
- **warmup ladder：** その10個のカーネルは起動のたびに報告されるが、コンパイルではなく読み込みである。固定のTritonは、バイナリがディスクのcacheから来た場合でも、プロセス内でカーネルを初めて使うときにpost-compile hookを呼び、jit monitorはそのhookから警告する。5回の起動でruntime cacheにファイルは一つも増えなかった。運用と画像入力は、cacheが温まればladderは空になるとは書かなくなった。

## 1.3.0 — 2026-09-17

### Added

- **`runtime.nccl_channels`**（任意。未指定＝NCCLが選ぶ）。両rankの `NCCL_MIN_NCHANNELS` と `NCCL_MAX_NCHANNELS` に同じ値を設定する。NCCL 2.30.7は参照対の両rankで64チャネルを選び、2026-09-11と2026-09-17で同じ記録だったので、これは不一致の修正ではない：保護を数GiB上回る所で動いているheadに、チャネルを減らせばメモリが戻るかを測定で確かめるためのもの。値を測るまでテンプレートは値を設定しないので、稼働中の対はprofileのfingerprintを保つ。値は次の切替から効く。着想はMia PR #200（つまみだけ。そのTP=2ランチャーも値を空のままにしている）。

### Documentation

- 運用：待機中の片肺の対はどちらのsupervisorにも止められず、ホストが固まるとそのホストのsupervisorも一緒に止まり、peer喪失の停止理由は設計メモのまま。Mia issue #193による。これは2026-09-16の片肺の対を逆向きにした事例を報告している。

## 1.2.1 — 2026-09-17

この版の内容はすべて、1.2.0を参照対で初めて動かしたことから出たもの。その実行は `docs/plans/MIA_ADOPTION_2_PLAN.md` のStage 6が持つ。

- **長いwarmupの段が目標の長さに収束する。** 最初の実機のladderは、65,536の目標に対して82,006 tokenを組み立てていた。行の番号の桁が一つ増えるとその行のtoken数が増えるので、256行の標本では少なく数えてしまう。組み立て側は、実測したtoken数に合わせて縮尺を直すようになった。
- 片肺の対から得た、運用手順の**ホストのデーモンの衛生**：peer rankが2.5 GiBの余裕に対し2.49 GiBで停止した。もう一方のホストの監視ダッシュボードがメトリクスごとに新しいSSHログインを張っており（毎秒3.6回）、ログインのたびに `polkitd` が太り（3.40 GiBまで）、`wireplumber` と `bluetoothd` が登録し直して太り、MOTDのスクリプトが全部走っていたためである。ダッシュボード側でSSH接続を再利用させて解消した。ログイン数の数え方とデーモンの大きさの比べ方、`MemoryMax` のdrop-inに `Restart=on-failure` も要る理由、片肺の対でも `/health` には200で答えること。
- **測っていたものを文書にした。** 82,018 tokenのprefillの間、token計数2つは202.8 s凍結したままだったが、進行の4信号がそろって凍結したのは最長でも8.3 sだった。これがKV使用率を信号の集合に残す根拠である。prefix cachingはblock単位で働く。N tokenのpromptを繰り返すと `(floor(N / block) - 1) x block` が復元され、2 block未満では何も復元されない。4,608 tokenのblockで、3,625／14,025／28,025 tokenに対して0／9,216／23,040 tokenと測った。
- **P24を起票**：要求単位のprefix cache no-store。73,728 tokenがcacheされてwarmな、1ターン14.9 sの80,024 tokenの会話に対して、触れずに流す15,025 tokenのレーン2本では無傷、4本では完全に追い出され（次のターン182.6 s）、42,026 tokenのレーン1本は単独で追い出した（184.0 s）。このflagは前者を直すが、後者は直さない。走っているレーンが確保する量は減らないためである。候補にとどまり、runtime overlayとimageの作り直しが要る。

## 1.2.0 — 2026-09-16

### Added

- **readiness後のwarmup ladder**（`server warmup`、`generation.warmup`、`generation.warmup_long_tokens`）。配信中にkernelのコンパイルが観測された要求の形（短文、tool呼び出し、画像1枚、配信中のtokenizerで長さを合わせる任意の長いprompt）を送り、段ごとの時間と、jit monitorがladderの前と最中に報告したkernelを記録し、dev経路が載っていれば最後にprefix cacheをリセットする。`cluster switch` は両rankがreadyになった後にrank 0でこれを流し、結果を `warmup` に保存する。ladderの失敗で切替が失敗したり巻き戻ったりすることはない。Mia PR #170を参考にした。段はこのキット自身の記録から選んでおり、そのPRのものではない。
- **engineの停滞検知**（`resources.stall_seconds`、テンプレートは600、rank 0のみ）。supervisorは既存の2 sの周期で `/metrics` を読み、要求がrunningなのに生成token数・prompt token数・KV cache使用率・running数がすべてその秒数凍結していれば、`stop-reason: engine-stall` で停止する。engineが固まっても `/health` は200のままなので、生存の信号だったことは一度もない。prefillの進行は活動として数え、届かない `/metrics` は決して数えない。運用手順にはswapの出入りの繰り返しについての注記も載せた。Mia PR #70を参考にした。
- **`server capacity`。** 稼働中のheadの起動ログと `/metrics` を読み、KV poolをそのまま表示する。stockの `GPU KV cache size` 行をblock数、最大長の要求1本あたりのblock数、group別のblock幅に分解し、stockの数字が `max_concurrency × max_model_len` であることを明記する。cacheした会話の推定は、LPA worker extensionを載せたprofile（そのRPCがgroupのspec種別を名指しする）でだけ表示し、モデル化していないgroup種別があれば出さない。画像入力の文書の「KV容量235,016 token」という書き方はこの読み違いで、訂正した。Mia PR #94を参考にした。runtime patchはない。
- **`api.dev_endpoints`**（任意、既定はfalse）。分散profileで、vLLMのdev経路（`/reset_prefix_cache`、`/collective_rpc`、`/sleep`、…）をloopbackのAPIに載せ、ベンチとwarmupの後始末が再起動なしでprefix cacheをリセットできるようにする。起動契約に既に書いてあるとおり、これらの経路は無認証である。Mia PR #37を参考にした。
- **P23を台帳に起票**：NVFP4 checkpointがBF16のまま残す射影のFP8 weight-only化（両rank合計で `self_attn*` 11.29 GiB、`shared_experts*` 1.97 GiB、`lm_head` 1.18 GiB。safetensorsのheaderから読んだ値）。候補のみで、品質の門を通すことが前提、未着手。adaptive verification lengthは見送りのまま。P06の再開条件に、spec acceptanceが1.00に張り付いていないかの検査（vllm#53030）を加えた。

### Changed

- 新しいprofileキーはすべて任意で、既存の `state/server.toml` と稼働中の対はそのまま有効で、復旧もできる。`supervise` はrankを受け取るようになった。長い段はフルprefillを払うため、SSHの転送はwarmupの呼び出しにreadinessのタイムアウトまでの時間を許す。

## 1.1.1 — 2026-09-15

- README（両言語）：ソースを配る場合には、ライセンスのページと第三者通知に既に書いてあるとおり、取り込んだ第三者コード（MITとApache）の著作権表示・許諾文も伴う。「Apache-2.0のみ」と言っていた文は、その次の文と矛盾していた。
- Dockerのbuild contextに `examples/server.example.toml` を含めた。`.dockerignore` が1.1.0より前の名前を許可リストに残したままだったため、参照imageがbuildできない状態だった。
- 運用手順：`server preflight` は検査を表示するだけで、`records/` に書くのは `server start` だけである。1.1.0の文はpreflightが記録すると言っていた。source archiveで配置する場合のruntime stateのリンクを `ln -sT` で張る方法と、正しいリンクと入れ子の `state/state` を見分ける `readlink -f` の出力を記載した。

## 1.1.0 — 2026-09-15

### Changed

- **ランチャーを一つに。** `service` コマンドと、その `state/site.json`／`state/kernel-validation.json` の受領証の門を撤去した。これは当初の固定設定のランチャーだった。`service start` はネイティブGB10経路が塞がっているlockのbase imageを選んだままで、どのワークフローも生成しない適格性の受領証を要求していたため、フルモデルを一度も起動しなかった。すべての測定、メモリの監視、両rankの切替・復旧はもう一方の経路にあり、今はそれが唯一の経路である。profileが実験的か通常運用として受け入れ済みかは、コマンド名ではなく記録された状態（READMEの状態表、ハーネスの受け入れ試験一覧）で示す。`examples/site.example.json` は削除した。同じ値はserver TOMLの `[nodes]` にある。共通のホスト補助関数（サイトの検証、serve引数、fabricの検査、snapshotの解決、subprocessの実行）は `glm53_setup/host.py` に移した。
- **`startup` は `server` になった。** サブコマンドは `python -m glm53_setup server {plan,freeze,assets,preflight,start,stop,status,ask}`、profileは `state/server.toml`（テンプレートは `examples/server.example.toml`）、moduleは `server.py`／`server_config.py`、文書は `docs/server-configuration.md`（両言語）。「startup start」はprofileと動作を同じ語で呼んでいた。旧い綴りの別名は残さない。**運用者は各ホストで `state/startup.toml` を `state/server.toml` にコピーし**（内容は変わらないので、profileのfingerprintも変わらない）、次の切替の前に両方のcheckoutを更新する。最初の1.1.0の対がreadyと確認できるまで、旧いファイルは残しておく。稼働中の対は起動したprofileのパスを記録しており、切替の復旧はそのパスから旧いprofileを起動し直すためである（[起動の安全](docs/launch-safety.ja.md#全レール検査と両rankの切替)）。
- **`--experimental` を廃止した**（`server start` と `cluster switch` から）。このflagは、通常の `service` ランチャーではない経路の印として存在していた。ランチャーが一つになれば、何の情報も持たない。
- 安定したruntimeの識別子は、BIZへの改名の時と同じく変えていない：containerのlabel `glm53.experiment.startup`、container名 `glm53-startup-r<rank>-<run>`、rankごとの記録 `state/startup-rank<N>.json` と `state/startup-request.lock`。そのため、1.0.xで起動した対も `server status`・`server stop`・切替から引き続き認識される。
- 文書：運用手順は「ランチャーは一つ」の節と、`server preflight` が何を検査し何を保証しないかを述べる「フルモデルの起動検査」の節を一つずつ持つようになった。SETUPの手順6は、生成できなかった受領証の代わりに、通常運用の受け入れで未了の項目を挙げる。README、文書一覧、構成表、起動の安全、検証、投機デコード、ハーネス、LPAの各頁から、ランチャー二本立ての枠組みを外した。この変更で受け入れ済みになったものはない。

## 1.0.1 — 2026-09-15

- LPAの文書（両言語）に明示的な「起動profileでLPAを有効にする」節を加えた：設定するTOMLキー、preflightが要求するimageのmarker、projectorのchecksum検査、テキストのみの条件、preflightから切替への順序、要求単位のopt-out。文書一覧と運用手順の表がこの節を指す。`lpa-train` の再現例を、cut 40のpilotから、配布projectorの設定である `--cut 32` に変えた。
- 実測したcut32の補助projector（Release `lpa-cut32-v1`）を、Apache-2.0の独立したGitHub Release assetとして配布する。固定したmanifest、学習データの帰属表示、教師モデルのモデルカード・notice、SHA-256 checksumを添える。元のtensorのbyteを保ち、重みはGitの外に置く。
- README、運用手順のパッケージ配置、構成、LPAの取得・モデルカード、ライセンス、起動の前提条件、文書一覧を両言語で更新した。runtimeの既定値と適格性の状態は変わらず、LPAはテキストのみのbatchのopt-inのまま。
- CLIのsmoke testを、退役したbetaのリテラルではなく `pyproject.toml` と版を比べ、正常終了も要求するように直した。

## 1.0.0 — 2026-09-15

### Changed

- 1.0.0をリリースし、betaの表示を退役した：版は `pyproject.toml` が持ち、公開監査はbetaの開示ではなく素のsemantic versionを要求するようになり、「beta」と書いていたREADME・手順書・文書の見出しはすべて、同じ事実をbetaなしで述べる。`service` launcherはkernel検証の証跡を待って門を閉じたまま。検収については何も変わっていない。
- 共通のハーネスcase H-01〜H-11を初めて一巡した記録（`20260915-harness-h-sandbox`）：11件すべてをnpmの `zcode-app-cli` 3.11.2-24のTUIでfixtureのリポジトリに対して一度ずつ走らせ、5件PASS・6件PARTIAL（caseごとに未検証の部分を名指し）。公式のZCode DesktopとClaude CodeはNOT RUNのまま。READMEの状態の行と受け入れ試験一覧がcaseごとの結果を載せる。
- テンプレートの `generation.max_tokens` を512から4096に上げた。この値が形を決めるのは `startup ask` の要求だけだが、固定のGLMテンプレートは必ず先に思考し、vLLMはそのblockを `max_tokens` に数える：`reasoning_effort=low` で一行の答えが約600 tokenを使い、既定の512では見える内容のないまま `finish_reason: length` が返っていた。既存の `state/startup.toml` は変更しない。編集するとprofileのfingerprintが変わるので、稼働中の対は `settings_changed` を報告し、次の切替まで `startup ask` はそれを拒む。

- 正典を一つにし鮮度を保つための文書の整理（英日とも）：測定文書にあった古い既定値の記述（APC、fused unpack、`index_checks`、8 GiBのreserve）を、起動設定を指す日付つきの経緯に書き直した。`startup`／`service` launcherの区別を、運用手順の正典の節一つと文書一覧の行として足した。構成表を、実在するruntime・cluster・validation・tools・hookのmoduleまで広げた。READMEのprefix cachingの状態行を分けた。起動既定値の表で取り残されていた `Lifetime` 行、言語切替が自分自身を指していた日本語の4ページ、英語版をリンクしていた日本語ページ、benchmarksの散文で数字の前に落ちていた空白を直した。このファイルで重複していた `Changed` 節を一つにした。日本語文書にbenchmarkとNCCLのコマンドblockを埋め込み、各言語が単独で読めるようにした。測定値と実行条件は一つも変えず、既存の見出しも改名していない。

- statusと結果のJSON（`prepare-status.json`、`checksum-status.json`、`preflight.json`、`command.json`、fixtureとbenchmarkの `result.json`）をすべて共通のatomic writerで書くようにし、書き込みの途中で落ちても、途中で切れたファイルではなく前のファイルが残る。各ファイルは末尾に改行が付く。imageのcapabilityのpreflight表、hostの要求lock、rank 0の探索は、検証コマンドごとの写しではなく `startup.py` の単一のhelperになった。imageの `Env` が無いとき、参照attentionの検査は例外を投げずに失敗として閉じる。

- 配布する起動既定を200Kの画像入力に切り替えた：`runtime.vision = true`、`max_model_len` 204,800、rankあたりKV 2.5 GiB、`mm_processor_cache_gb` 0.1、`reserve_gib` 2.5で、動画は引き続き拒む。256Kのテキスト専用profileは代替として文書に残す。それに対する2.5 GiBのreserveは検証していない。設定・選んだ経緯（量子化の除外、動画のprofiling、head専用のprocess、JIT cacheの事故、reserve）・測定を載せた `docs/vision.md`（英日）を足し、README・セットアップ手順書・起動設定・ハーネス・benchmarks・最適化・文書一覧の参照を更新した。起動既定値の表は、LPAが無効で出荷されることも述べる。既存の `state/startup.toml` は変更しない。
- ハーネスの文書を、接続設計・受け入れ試験一覧・caseごとの状態の単一の正典として書き直した：状態の列を足し（API-01〜03 PASS、API-04 PARTIAL、公式ZCode DesktopはNOT RUN／同梱CLIはTUI packageの欠落でBLOCKED、npm配布のsmokeは補助証拠、Claude Codeと共通のH caseはNOT RUN）、ZCodeの配布形態を分け、ぼかしていたtelemetryの記述をbundleの静的な読みによる配布形態ごとの表に置き換え（npm CLIはOTLP endpointを設定したときだけtraceを送る。Desktopはtraceとeventのendpointを直書きし、継承したスイッチを剥がす）、例のportを起動テンプレートに揃え、ハーネスが自身のconfigディレクトリへ書くのを拒む規則を足した。READMEと検証は状態を書き写さず一覧を指す。npm配布について日付つきのTCP標本（headlessのprompt一回の間に観測したのはloopbackのtunnelだけ）を、H-09の補助証拠として記録した。
- 配布物の名前を「GLM-5.3-Flash on 2× DGX Spark Enterprise Setup」から **GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ** に改めた。BIZは保守者と業務利用の意図を示す。「enterprise」という語はサポートや認証を含意すると読まれたので、散文では「business use」と書く。安定した識別子は変えない：参照imageのtag `glm53-enterprise:reference`（既存の記録に焼き込まれている）と台帳のID E01〜E03。Pythonの配布名は `glm53-flash-biz-setup` になる。
- 配布する起動既定を、直列の最適化profileに揃えた：256K context、rankあたりKV 3 GiB、MTP3、LPA cut32/tail512/B128、dense checkpoint retentionつきのAPC、fused unpack、非同期のindex検査。寿命の期限は既定で無効にし（`run_seconds=0`）、hostメモリのreserveによるメモリ監視は残す（reserveは下にある後の項で3 GiBに下げた）。テンプレートがdense retentionを明示的に選んでいても、省略した／数値のretention設定は保つ。image ID、projector／hash、nodeの詳細はインストールごとの仮値のまま。既存の運用者のTOMLには明示的な移行が、稼働中のsupervisorには再起動が要る。
- 分散起動のcontainerのTriton・TileLang・TorchInductorのcacheを、mountした `state/tp2-runtime-cache` に向けた（`TRITON_CACHE_DIR`、`TILELANG_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`）。既定の場所（`/root/.triton`、`/root/.tilelang`、`/tmp/torchinductor_root`）はcontainerのlayerにあったので、起動のたびに約2,000件をcompileし直し、その一部は配信中に走っていた。headでのそうしたcompileの集中の一つは、200Kの画像profileでのmemory-reserveによる停止と重なった。
- `resources.reserve_gib` に小数を受け付け（テンプレートは `3.0` と書く。整数も引き続き有効）、「測り直す」という注記を、2秒周期のsupervisorが捕まえられることに置き換えた：閾値割れからcontainer終了まで約9秒、実測した0.15 GiB/sの下降、16 GiBのswapではOOM killの記録なし、そして下限の目印にならないdriverの `NV_ERR_NO_MEMORY` メッセージ。後の項が、画像profileとともにテンプレートの値を2.5にする。
- 施策ごとの機能の受け入れ、性能面の採用、既定値を分けた。最終的な組み合わせの検収は、候補が選ばれるまで先送りする。
- 実験的なnon-eager起動が、compileしないprefillと1系列のdecode Graphを明示的に選ぶようにし、imageの検査と、先送りした組み合わせのguardを付けた。フルモデルの検収は保留のまま。

- module、worker class、RPC、mountのパス、テスト、文書を通してLPAを正式名にした。旧来のinterfaceなしで現行のimage契約を要求する。
- 起動profileを読み込み／コマンドの境界で検証し、serve引数の組み立て中に繰り返していたschema全体の検証をなくした。

- `resources.run_seconds = 0` で実行期限を無効にでき、低メモリの保護は残る。自動再起動や本番の可用性の検収を足すものではない。

- 生成したvLLMのpatch出力に、実行される構文木を変えずに、目立つ改変通知を足した。
- 運用者向けのコマンド、検証の道具、runtimeの適応、設定、Dockerの資材を責務ごとにまとめた。
- モデルIDとrevisionを `config/runtime.lock.json` に集約した。
- downloadをprocess lockとatomicなstatus更新で直列化した。一時停止したdownloadは検証の待機を終わらせる。
- 公開の入口から、手元の実験へのリンクとhost固有のimage識別子を外した。

### Added

- 画像profileで、ZCodeのterminal sessionの中での画像読み取りを一件記録した：モデルがshellコマンドでscreenshotを保存し、ファイル読み取りのツールで読み、画像の中にしかない文字を説明した。ハーネスのpromptへの画像の直接添付とClaude Codeは未確認のまま（`docs/vision.ja.md`、README）。

- 新規・更新したhostのkernelの指針（`docs/operations.md`、英日）：現在のDGX OSの更新は `7.0.0-1019-nvidia` を入れ、既定で有効なKexec HandOverによって、2台のRoCEの登録が `ibv_reg_mr_iova2 ... Cannot allocate memory` で失敗することがある。NVIDIAのadvisory、未解決のNV-Kernelsの分析、二つの選択肢（`6.17.0-1032-nvidia` に留める、または `kho=off` で起動してそれを確かめる）、同じエラーのfirmware関連の別の報告を記録した。セットアップ手順書は `/proc/cmdline` を記録し、2台の手順の前にkernelを選ぶようにした。READMEの前提条件はそこを指す。`7.0.0-1019-nvidia` は本リポジトリでは検証していない。

- 任意の起動キー `runtime.vision`（未指定はfalse、テンプレートでは明示）。`true` は両rankから `--language-model-only` を外してvision towerを読み込み、`--limit-mm-per-prompt '{"video": 0}'` を足す：**動画入力は無効のまま**。そうしないと起動時のprofilingが30,000 tokenの動画をencodeするため。`--mm-processor-cache-gb` も、headで二重に持たれるvLLMの4 GiBではなく、任意の `cache.mm_processor_cache_gb`（未指定は0.1）から設定する。後の項がこれを配布既定にする。

- 既存の組み合わせimageで、実入力による256Kの容量と3か所参照の検査（rankあたりKV 3 GiB、既存の4 GiBのメモリreserve）。以前の200Kの速度・tool-eval・FreedomBenchの測定は、元のprofileの下に残す。

- Desktop同梱のruntimeを静的に読んで得たZCodeの権限modeの事実（modeのenumと正規化、未実装の `auto`、ツールごとの危険度の記述子、判定の順序、PreToolUse hookの統合）と、`yolo` で動かしつつ、既存ファイル・clientの設定ディレクトリの変更や破壊的なshellコマンドの前には確認を求める既存ファイルguard hook（`examples/zcode-hooks/`）。WriteとBashのcaseをheadlessで検証した記録を残し、同梱CLIがheadless modeで手元のモデルに届くことを書いた。`limit.context`／`limit.output` がcontext window、`max_tokens`、自動compactionの予算（実効window − min(output, 21000)、閾値 − 13000）にどう効くか、そこから来るvLLMの下での `limit.output` の上限、長いprefillを打ち切るstreamのidle timeoutを文書にした。`/effort max` は実用上の上限なく思考し、日常の設定は `high` 以下という運用者向けの注記を足した。任意の起動キー `api.prompt_tokens_details`（`--enable-prompt-tokens-details`）を足し、LPA優先のprofileでハーネスのcache表示がゼロになる理由を、ZCodeで設定した要求ごとの `glm53_lpa_mode = "off"` によるopt-outとともに文書にした。実測したblock単位の再利用（4,608 tokenのblock。16,859 tokenのpromptを繰り返して9,216がcache）を記録した。256K profileでsupervisorによるメモリ停止が2回あった後、配布の `resources.reserve_gib` を4から3に下げた。テンプレートは `lpa.enabled = false` で出荷し、LPAを共有prefixの再利用を手放すbatch向けのopt-inとして文書にし、256Kを配布既定に保った。長いpromptを2回送り、「サーバーがcached tokenを一度も報告しない」と「要求が近似された」を切り分ける `tools/check_prefix_cache.py` と、ZCodeのguard hookの導入手順を足した。

- 英日の性能・容量のQ&A：256KのKVの収まり、1Mを見込んだメモリ予算、固定のMTP重み、長いprefillの待ち時間、同時配信／EP／PPの得失。実測の証拠にリンクし、例示としての1Mで40分という外挿を実測と区別した。

- 正準順imageでのrelease candidateの測定：有効なsparkDashのstream 12本、tool-evalの全69 scenario（90/100、TC-43はSafety Gateに不合格）、FreedomBenchの英語原版60問すべて正解、200Kの容量／3か所参照の検査の完了。無効だった通信の試行は別に残し、業務・ハーネスの検収の限界も保つ。

- 文書の役割・言語の有無・事実ごとの単一の正典を持つ英日の文書一覧（`docs/README.md`）と、各施策がどこに効くか・直列の組み合わせの結果・用途別のprofileを示す英日の推論最適化の全体像（`docs/optimization-overview.md`）。READMEにprefix caching／retentionの状態行と外部のEuryale研究プロジェクトの説明が加わり、施策台帳の文書役割の表は文書一覧へのポインタに置き換えた。運用・構成・検証・部品検証・性能調査・indexer再利用・contributingに日本語版を、APC優先LPAの設計に英語版を足し、利用者向けの文書はすべて英日の対になった。

- GLMのsparse-MLAが共有するruntimeの境界での、論理的な候補の正準順。新しくbuildした参照imageで有効。選択とpaddingを保ち、物理cacheへの対応付けの前に正規化する。source固定の自動導入、比較のための明示的な上書き、GB10での通常／投機の回帰の限られた証拠を文書にした。

- model originのBearer client、未設定／空を保つ固定化したallocatorの上書き、構造化した全railのRoCE検査、実験的な2 rankの停止前の資材確認／切替／復旧の経路。CPUでの契約と実hostでの検収は、起動安全のガイドに別々に記録した。

- APC優先で、cacheされていない後半だけに効くLPA：境界はschedulerが持ち、共有cacheはexactのみ、exactの明示的なpriming、source固定のclient検証、GPUのcache隔離のfixture、損益分岐の測定の道具。

- NVIDIAのcheckpoint、GB10互換のhardware、MSI EdgeXpertでの測定を区別する導入の入口と、正規のcache・MTP view・LPA projectorの置き場所の指針。

- 検査つき非同期indexの単独の結果と、起動時のauto／sync／asyncの明示的な選択。検査は必須のままで、組み合わせmodeの受け入れは別に扱う。

- フルモデルのTP/PPの時間と復元の結果。PPのprefillは速く、生成は遅く、メモリの余裕は限られ、profilerの再起動失敗の後のprofiler／容量の結果は未完であることを、そのまま残す。

- フルモデルのEP off/on/offの結果：配置を確認し、品質・取消・容量を範囲を限って確かめ、測った処理量の用途についてはEPを採らないと判断した。内部abortのtelemetryは通常の完了counterと区別する。

- 実験的なTP2/PP2の明示的な選択と、設定可能なstage境界。hybridの配置と組み合わせのguardつき。
- 同期と非同期の検査つきindexをGraph captureと独立に比べる、排他の診断RPC。範囲検査は一つも無効にしない。

- hybrid cacheで有効なPPのstage境界のための、byte検証済みの8層fixtureのoption。attention／KDAの状態のhashと、実際のexpert配置の観測も加えた。

- 実験的なdeferred-mHCのpipeline bufferと、source固定の共通cache配置の選択。PPの起動に失敗した証拠を残し、GPUでの検収は保留。
- 条件を揃えたtask順の測定と、試した用途についてはtask専用のgroup分けを採らないという、範囲を限った判断。

- prefill chunkの単独の掃引（処理量と最大ITLの得失）と、別の16K x 2要求の容量の証拠。
- FP32のNoPE融合と、範囲を限ったtile／query chunkの診断。launchと中間値が減っても速度は負だった結果を、そのまま残す。

- 明示的で既定offのExpert Parallelの実装と、単独の検証計画。EPの採用を、検収済みの2系列batchや、配布構成だけの起動の受け入れと分ける。

- batchの単独のA/B/A測定と、別の16K × 2要求の容量検査。KV固定／RAMの条件を明示。
- paddingしたnative attentionの部品probe。配信を何も変える前に、数値の不合格と非対応の形状の証拠を残した。

- 英日のenterpriseの目標と、性能・品質の施策台帳。比較用の安定したID、証拠へのリンク、先送り候補の基準、次のGLMの評価欄を持つ。政治的偏りの評価は、普遍的な中立の主張とは区別する。

- 単独で試験したCUDAのunpack融合とCSA2の研究部品：範囲を限ったindexerのcapture、要求内での候補の再利用、候補の選択／tailの展開、nativeで支えた共有poolの採点。部品のbenchmarkは、数値の正しさと実際の速度の向上を区別する。
- 明示的な部品検証の起動profileと、フルのtargetでのCUDA A/B/A／indexer重ね合わせのdriver。LPA／MTPの統合とは別。

- FreedomBenchの収集と、英語原版／長文文脈での予備的な観察。政治的な拒否、実行／形式のエラー、出典への忠実さを、まだ検収していない全構成の一覧と人手の査読から分けた。

- 明示的な実験用TP=2 launcherと直列のchat clientが共有する、分類したTOML設定。不変の成果物の検査、範囲を限った監視、MTPを考慮したLPAのopt-inつき。

- opt-inの後段prefillのattention入力近似、teacher replayの検査、範囲を限ったLLM-jp corpusの標本抽出、対角／低ランクのprojectorのfit。実験の範囲についての英日の指針つき。

- MTP k=3の例と、条件を揃えたk=1との比較。位置ごとの採択と、用途によって変わるlatency／処理量の結果を含む。
- opt-inのMTP k=1のmetadata viewの準備、BF16のdraft設定、CPUでの拒否の検査、英日の実測比較／ロールバックの指針。
- 最初のフルモデルTP=2の直列参照benchmarkの結果と、benchmarkのCLIが0で終わっても未完の実行を拒む独立のchecker。
- 商用利用・改変・再配布の英日のガイド。上流モデルのMIT通知を含む。
- 公式ZCodeと実験的なClaude Codeの接続計画。それぞれに必須のend-to-endの受け入れcase（未実施）。
- 再現可能な2 rankのPyTorch／NCCL診断。transportの読み方と英日の手順つき。
- WindowsからのSSH接続、永続的なprofile、routingの確認、復旧を扱う英日のQSFP／NetworkManager実践ガイド。
- betaの配布物に名前を付けた：**GLM-5.3-Flash on 2× DGX Spark Enterprise Setup**。
- Apache-2.0ライセンス、保持したthird-partyの通知、英日の入口の文書。
- 準備・起動の点検・検証のための統一CLI `python -m glm53_setup`。
- digestで固定した参照imageのbuildコマンドと、移植可能なcheckoutのパス。
- CPUのCIと、clean checkoutでのCLIの動作と公開の境界の検査。
- 前提条件、順序つきの関門、受け入れのチェックリスト、AIへの引き継ぎ指示を持つ英日のセットアップ手順書。

### Validation status

- MTP k=3は測定要求21件と基本APIの検査11件すべてに合格し、短・中の入力のdecodeをk=1より改善し、報告されるモデルのメモリは変わらなかった。以後の実験ではこちらを優先する。k=2／k≥4と実際のハーネスは未試験のまま。
- フルモデルTP=2のMTP k=1は、同じ測定要求21件と基本APIの検査11件を完走した。速いdecodeはrankあたり約7 GiBを要し、すべての用途を改善するわけではない。実際のハーネスと分散の復旧は未検収のまま。
- 文書にした直列の参照profileで、45層全部のTP=2での読み込み、最終回答／ツールのAPI smoke、benchmarkの測定要求21件が完了した。ハーネスと通常運用の導入は未検収のまま。
- 固定のGLMテンプレートではthinking-offを使わない。最終回答／ツールの受け入れは、推論文の完全一致やbatchを跨いだ数値の診断とは別物で、生の不一致はそのまま残す。
- 2台のRoCEのcollectiveのpatternは合格。最後の256 MiB AllReduceの測定はMTU 1500で1.18〜1.21 GB/sで、別のdisk checksumのjobが動いていた。これはフルモデルのTP=2を検収するものではない。
- GB10 1台の4層fixtureは、Marlin W4A16と候補を保持するNoPE参照経路で、試した生成／状態の整合のcaseに合格した。
- 整理したDocker imageを作り直し、GPUの部品検査と長い入力のfixture検査を、packageしたCLIで繰り返して成功した。
- CUTLASS W4A4は生成を完了したが、試した実行の形状の間で結果が違った。
- 固定しているSM120のsparse-MLA backendではbatch不変性が使えない。
- **文書にした実験的なprofileの範囲を超えるモデルの品質、本番の信頼性、性能は未検証のまま。**
