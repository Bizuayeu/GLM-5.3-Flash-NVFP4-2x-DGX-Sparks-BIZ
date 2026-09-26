# 構成

[English](architecture.md)

本プロジェクトは、checkout内で完結する運用者向けのツールキットです。ソースアーカイブにはモデル重みも遠隔管理サービスも含みません。任意のLPA重みは独立したRelease添付物で、配布ファイル構成と導入先は[運用手順](operations.ja.md#資材の保管場所とパス)が正典です。

## どこから読むか

枠組みではなく「テストから何ができるか」で分けた四種類です。

| | 持つもの | 名前の付いた継ぎ目 |
|---|---|---|
| **入口** | 引数の解釈と、action がどのハンドラに届くか | `server.ACTIONS`、`cluster.ACTIONS`、`__main__.COMMANDS` |
| **編成** | 手順の順序と、失敗時に何を巻き戻すか | `switch.switch`、`switch.resume`、`server.act_launch`、`cluster.act_switch` |
| **判断** | 設定の検査、引数の組み立て、報告の整形——純粋関数 | `server_config.VALIDATORS`、`server_config.SERVE_STEPS`、`server_config.image_capability_checks`、`images.probe_verdict`、`validation.*.engine_kwargs`、`runtime.apc_policy`、`runtime.patch_*.patch_text` |
| **副作用** | docker、HTTP、subprocess、torch、vLLM | `host.run`、`model_http`、各runner内のGPU遅延import |

判断層には「ある測定を次の測定と比較可能にしている数値」が置かれるため、torch・vLLM なしで import できる状態を保ちます。`engine_kwargs(args)` は、実行できないホストの上でも fixture runner の起動設定を述べられます。`server_config` は副作用 module を import しません。必要な argv の雛形とサイト設定の検査は純粋関数で、`server_config` 自身と `fabric` にあり、`host` が入らないことをテストが import を読んで確かめます。入口層と編成層は副作用を引数で受け取るのでテストが差し替えられ、副作用層は差し替え点であって検査対象ではありません。

順序が契約の一部である箇所が二つあります。`VALIDATORS` が profile の規則を固定した順に走らせるのは、**最初の raise が運用者の読む一文**だからです。`SERVE_STEPS` は一つの引数リストに書き込み、後段が前段の残したものを参照します。

| 場所 | 責務 |
|---|---|
| `glm53_setup/__main__.py` | 固定したコマンド振り分け。利用者が指定するモジュールの動的読込は行わない |
| `glm53_setup/config.py` | checkout内のパスと、検査済みの固定設定 |
| `glm53_setup/server.py`、`server_config.py`、`capacity.py`、`warmup.py`、`mojibake.py`、`agreement.py` | 起動・監視・headへのクライアント。カテゴリ別のTOML設定と、そこから導くもの（vLLM引数の雛形、imageのcapability検査と警告、凍結した起動manifest、devモード）。KV起動行の分解、readiness後の要求ladder、日本語・韓国語の化け文字検査、参照runとのtoken単位の一致（`server agreement`） |
| `glm53_setup/host.py` | ランチャーが共用するホスト側の補助：fabric検査、snapshot解決、メモリ標本、container検査、subprocess実行 |
| `glm53_setup/download.py`、`verify_download.py`、`images.py`、`build_reference.py` | 資材の準備（固定checkpointの取得、downloaderを待つchecksum検証、base imageの確認とその合格規則、reference imageのbuild）と、ガード付きのローカル操作 |
| `glm53_setup/cluster.py`、`switch.py`、`launch_assets.py`、`fabric.py` | 両rankの停止前検査、所有権つきの切替・復旧とその再開（resume）、読み取り専用の起動識別情報、サイト設定の検査・NCCL環境・RoCEレール検査（[起動契約](launch-safety.ja.md)） |
| `glm53_setup/model_http.py`、`io.py` | モデルAPIに限定しredirectに従わないHTTP transport、ローカル状態の永続化helper |
| `glm53_setup/runtime/pinned_patch.py`、`patch_*.py` | image buildが当てるsource固定のvLLM patch群。`pinned_patch` が共通部分（固定ファイルのhash検査、`--package`／`--check` コマンド、package脇に書くrecord）を持ち、各 `patch_*` moduleは対象・pin・anchorだけを、逸脱した・適用済みのsourceを拒む純粋関数 `patch_text(text)` として述べる |
| `glm53_setup/runtime/reference_attention.py`、`patch_nope_reference.py`、`fa2_attention.py` | 候補を保存するeagerなNoPE MLA参照計算、そのsource固定の導入、6行を超える呼び出しのFA2経路（`runtime.fa2_attention`） |
| `glm53_setup/runtime/candidate_order.py` | 共通のsparse-MLA境界での論理候補順序の正規化（[候補順序](candidate-order.ja.md)） |
| `glm53_setup/runtime/moe_token_order.py`、`patch_moe_order.py` | Marlin MoE kernelの前で各expert内のtoken順を一つに固定（`runtime.canonical_moe_order`）と、そのsource固定patch |
| `glm53_setup/runtime/stable_topk.py`、`patch_indexer_topk.py` | kpool indexerのtop-kの同点を規則で決める（`runtime.stable_indexer_topk`）と、そのsource固定patch |
| `glm53_setup/runtime/prefix_dedup.py`、`patch_prefix_dedup.py` | 同じ内容のprefix pageをcacheに一つだけ持つ（`runtime.prefix_page_dedup`）と、そのsource固定patch |
| `glm53_setup/runtime/patch_slot_mapping.py` | source固定patch：slot対応付けのkernelがblock tableを行の中だけで読む |
| `glm53_setup/runtime/patch_kpool_seed.py` | source固定patch：kpoolのprefill seedがtailのblockをtailのstrideで番地付けする（vLLM #57477） |
| `glm53_setup/runtime/inductor_pin.py`、`inductor_pin_pth.txt` | Dynamo の状態復元が最初の compile の後に切ってしまう Inductor の決定性モードを保つ（`runtime.inductor_deterministic`）。テキストファイルを image の site ディレクトリに `glm53-inductor-pin.pth` として mount し、インタプリタ起動時に読み込ませる |
| `glm53_setup/runtime/lpa.py`、`lpa_query.py` | LPAのworker制御、Attention入力の近似、要求単位のquery省略 |
| `glm53_setup/runtime/apc_policy.py`、`apc_runtime.py`、`apc_worker.py`、`patch_apc_lpa.py` | APC優先LPAの適用判定、通常計算由来のprefixだけを共有登録する境界、workerへの伝達（[設計契約](apc-lpa-design.ja.md)） |
| `glm53_setup/runtime/fused_unpack.py`、`graph_policy.py` | FP8 unpack融合kernel、LPAがeager実行を要する条件 |
| `glm53_setup/runtime/indexer_capture.py`、`indexer_worker.py`、`component_worker.py` | CSA2のindexer観測と、排他的な部品診断worker（[Indexer再利用](indexer-reuse.ja.md)） |
| `glm53_setup/runtime/memory_probe.py` | dev の `/collective_rpc` 経由で配信workerを調べるprobe。`validation.memory_probe` がこれを読み込み、checkoutの版をimageの上にmountする：`allocator_stats`、`host_stats`、`host_census`、`weight_digest`、`kernel_hashes`、`autotuners`、`inductor_state`、`fa2_stage`、`trace_begin`／`trace_end`（[起動設定](server-configuration.ja.md#apiと診断)） |
| `glm53_setup/runtime/pipeline_state.py`、`patch_pipeline.py` | PP fixtureの転送と、そのsource固定patch（P17） |
| `glm53_setup/validation/make_fixture.py`、`run_fixture.py`、`summarize_fixture.py`、`inspect_runtime.py`、`probe_attention.py`、`reference_check.py`、`parity.py` | fixtureの作成・実行・判定、コンテナ内の確認、NoPE dispatchの探査、参照Attentionの一致と、Attention部品ベンチが共有するBF16の許容幅と判定（[検証範囲](validation.ja.md)） |
| `glm53_setup/validation/run_agreement_fixture.py`、`compare_agreement.py`、`quant_error.py`、`run_repeat_trace.py` | fixture上の再量子化検査と、反復実行で最初に出力が違うモジュールの特定（[検証範囲](validation.ja.md#フルモデルtp2の実験範囲)） |
| `glm53_setup/validation/run_components.py`、`run_graph_fixture.py`、`run_indexer_fixture.py`、`run_apc_lpa_fixture.py`、`indexer_overlap.py`、`expert_worker.py`、`pipeline_worker.py`、`apc_fixture_worker.py` | 部品A/B/A、Graph、indexer、APC/LPA、EP、PPの各fixtureと、fixture専用のworker（[部品検証](component-validation.ja.md)） |
| `glm53_setup/validation/run_lpa.py`、`lpa_corpus.py`、`train_lpa.py` | LPA fixtureの検査、コーパスの準備、projectorの学習 |
| `glm53_setup/validation/freedombench.py`、`freedom_scoring.py`、`apc_history.py`、`profile_trace.py`、`benchmark_*.py` | FreedomBenchの実行と採点、APC履歴の回帰試験、traceのevent集計、部品ベンチ |
| `glm53_setup/validation/fused_nope.py`、`fused_nope_dot.py`、`indexer_candidates.py`、`indexer_reindex.py`、`indexer_shared_pool.py` | 再現のために残す退役した試作。呼ぶのはそれぞれのベンチとテストだけ：融合NoPE attention（[部品検証](component-validation.ja.md)）とindexer候補の再利用（[Indexer再利用](indexer-reuse.ja.md)） |
| `config/` | モデル・imageの固定値と`lpa-projector.lock.json`（Release URL、checksum、教師・学習来歴）。認証情報や実測したサイト設定は持たない |
| `examples/` | 二つの起動設定 `server.example.toml`（配布既定）と `server.axl.example.toml`（公開した任意設定）、MTP投機設定のテンプレート。値はすべて例示 |
| `examples/zcode-hooks/` | ZCodeの既存ファイルガードhookと導入手順（[ハーネス](harnesses.ja.md)） |
| `overlays/` | 公開した任意設定のcheckpointが要するvLLM source overlay 2件と、その台帳（[overlays/README.md](../overlays/README.md)） |
| `docker/` | imageの構築。base digestはビルドコマンドがロックから渡す |
| `requirements/` | ホスト側ツールの固定した依存 |
| `tests/` | CPU契約 |
| `tools/` | `check_publication.py`（公開監査）、`release_notes.py`（tagが公開するChangelogの節）、`kernel_hashes.py`（indexerのkernelを両配信workerの中でhash）、`assess_benchmark.py`、`check_prefix_cache.py`、`decode_check.py`・`decode_divergence.py`・`weight_digest.py`（切替の後のdecode検査と重みのdigest、[起動契約](launch-safety.ja.md#切替の後のdecode検査)）、`nccl_probe.py`、`prepare_mtp_view.py` |
| `.github/workflows/` | CI（LinuxとWindowsでのCPUテスト・Ruff・公開監査）と、tagで起動するGitHub Release |
| `LICENSES/` | 上流ライセンス原文の保持 |
| `state/`、`records/` | ローカルの可変状態と実験の証跡。配布対象外 |

CLIは、選択したコマンドが実際に必要とする場合にだけGPU依存をimportします。help、設定、CPUテストは、ホストにTorchやvLLMが入っていなくても動きます。GPUプログラムは固定imageの中で実行します。

コメント付きの `examples/server.example.toml` は、起動設定の完全なスキーマも兼ねます。TOMLの全体検査は `server_config.load` と、単独で呼び出せる `server.command` の境界で行います。`server_config.serve_args` は検査済みのprofileを受け取り、スキーマを読み直しません。内部の組み立て工程であり、入力検査の入口ではありません。fabric固有の小さなガードは独立したままです。

モデルIDとrevisionの設定元は[runtime.lock.json](../config/runtime.lock.json)の一つだけです。可変ファイルは、呼び出し元の作業ディレクトリに関係なくcheckoutを基点にします。本ツールキットは保守されたcheckoutから実行してください。汎用のPythonライブラリとしては提供していません。

reference imageのbuild時patchは、変更する固定vLLMファイルの完全なSHA-256を確認してから当てます。公開した任意設定のoverlayも起動時に同じ方法で照合します。選択したAttention候補はすべて保持します。runtimeの数値計算と検証ハーネスは別のモジュールにしてあるため、CLIコードを移動しても数学的な実装は変わりません。

## モジュールの置き場所

**worker拡張。** `runtime/` には配信の起動が読み込むworkerクラスを置きます。`server_config.apply_worker_extension` が指すのは `lpa.LPAWorkerExtension`、`component_worker.ComponentWorker`、`memory_probe.MemoryProbeWorker` で、`ComponentWorker` は `indexer_worker.IndexerCaptureWorker` を継承します（`run_indexer_fixture` もこれを単独で読み込みます）。`validation/` にはfixture runnerだけが読み込むworker（`apc_fixture_worker`・`pipeline_worker`）を置きます。例外は `expert_worker` です。配信はEP観測の起動でこれを読み込みます（`validation.expert_worker`）が、`validation.pipeline_worker.PipelineFixtureWorker` を継承するため `validation/` に置いています。ベンチとテストからしかimportされない退役した試作も `validation/` に置きます。

**コマンド。** 検証runnerの多くは `python -m glm53_setup` のsubcommand（`__main__.COMMANDS`）です。次は `python -m glm53_setup.validation.<module>` としてだけ実行します：`benchmark_apc_lpa`（`apc-lpa-benchmark` として登録）を除くすべての `benchmark_*`、`run_components`、`run_graph_fixture`、`run_indexer_fixture`、`run_repeat_trace`。`indexer-overlap` はこの群に当てはまるのに登録されています。`runtime/indexer_capture.py` が記録し `run_indexer_fixture` が駆動する候補の行を、CPUで比較するものです。

## 検証の境界

ダウンロードの完了、checksumの合格、GPUスモーク、設定の解釈、Attentionの一致、fixtureの統合、フルモデルTP=2の検収は、それぞれ別種の証拠です。ある水準の結果を、別の水準の代わりにはできません。とくに、GPU 1台のfixtureは2 rankの証拠の代わりになりません。
