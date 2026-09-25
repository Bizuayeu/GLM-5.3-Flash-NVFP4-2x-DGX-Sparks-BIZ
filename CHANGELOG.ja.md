# 変更履歴

[English](CHANGELOG.md)

正典は[英語版](CHANGELOG.md)です。GitHub Releaseの本文は英語版の各版の節から作られます。この日本語版は1.6.0から始めており、それ以前の版は英語版を参照してください。項目は英語版と同じ順に並べています。

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
