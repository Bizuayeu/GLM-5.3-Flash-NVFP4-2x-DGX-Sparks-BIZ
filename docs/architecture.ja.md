# 構成

[English](architecture.md)

本プロジェクトは、checkout内で完結する運用者向けのツールキットです。ソースアーカイブにはモデル重みも遠隔管理サービスも含みません。任意のLPA重みは独立したRelease添付物で、配布ファイル構成と導入先は[運用手順](operations.ja.md#資材の保管場所とパス)が正典です。

## どこから読むか

枠組みではなく「テストから何ができるか」で分けた四種類です。

| | 持つもの | 名前の付いた継ぎ目 |
|---|---|---|
| **入口** | 引数の解釈と、action がどのハンドラに届くか | `server.ACTIONS`、`cluster.ACTIONS`、`__main__.COMMANDS` |
| **編成** | 手順の順序と、失敗時に何を巻き戻すか | `switch.switch`、`server.act_launch`、`cluster.act_switch` |
| **判断** | 設定の検査、引数の組み立て、報告の整形——純粋関数 | `server_config.VALIDATORS`、`server_config.SERVE_STEPS`、`validation.*.engine_kwargs`、`runtime.apc_policy` |
| **副作用** | docker、HTTP、subprocess、torch、vLLM | `host.run`、`model_http`、各runner内のGPU遅延import |

判断層には「ある測定を次の測定と比較可能にしている数値」が置かれるため、torch・vLLM なしで import できる状態を保ちます。`engine_kwargs(args)` は、実行できないホストの上でも fixture runner の起動設定を述べられます。入口層と編成層は副作用を引数で受け取るのでテストが差し替えられ、副作用層は差し替え点であって検査対象ではありません。

順序が契約の一部である箇所が二つあります。`VALIDATORS` が profile の規則を固定した順に走らせるのは、**最初の raise が運用者の読む一文**だからです。`SERVE_STEPS` は一つの引数リストに書き込み、後段が前段の残したものを参照します。

| 場所 | 責務 |
|---|---|
| `glm53_setup/__main__.py` | 固定したコマンド振り分け。利用者が指定するモジュールの動的読込は行わない |
| `glm53_setup/config.py` | checkout内のパスと、検査済みの固定設定 |
| `glm53_setup/server.py`、`server_config.py`、`capacity.py`、`warmup.py`、`mojibake.py` | 起動・クライアント制御、カテゴリ別のTOML設定、KV起動行の分解、readiness後の要求ladder、日本語・韓国語の化け文字検査 |
| `glm53_setup/host.py` | ランチャーが共用するホスト側の補助：サイト設定の検査、serve引数、fabric検査、snapshot解決、subprocess実行 |
| `glm53_setup/download.py`、`images.py`、`build_reference.py` | 資材の準備と、ガード付きのローカル操作 |
| `glm53_setup/validation/` | 明示的なCPU／GPU検査、fixtureの作成と判定 |
| `glm53_setup/runtime/` | sourceを固定したNoPE適合と、候補を保存する参照計算 |
| `glm53_setup/runtime/lpa.py`、`lpa_query.py` | LPAのworker制御、Attention入力の近似、要求単位のquery省略 |
| `glm53_setup/validation/run_lpa.py`、`lpa_corpus.py`、`train_lpa.py` | LPA fixtureの検査、コーパスの準備、projectorの学習 |
| `config/` | モデル・imageの固定値と`lpa-projector.lock.json`（Release URL、checksum、教師・学習来歴）。認証情報や実測したサイト設定は持たない |
| `examples/` | 例示値だけを含む起動設定・MTP投機設定のテンプレート |
| `docker/` | imageの構築。base digestはビルドコマンドがロックから渡す |
| `requirements/` | ホスト側ツールの固定した依存 |
| `glm53_setup/validation/freedombench.py`、`freedom_scoring.py`、`apc_history.py`、`profile_trace.py`、`benchmark_*.py` | FreedomBenchの実行と採点、APC履歴の回帰試験、traceのevent集計、部品ベンチ |
| `glm53_setup/runtime/candidate_order.py` | 共通のsparse-MLA境界での論理候補順序の正規化（[候補順序](candidate-order.ja.md)） |
| `glm53_setup/runtime/apc_policy.py`、`apc_runtime.py`、`apc_worker.py`、`patch_apc_lpa.py` | APC優先LPAの適用判定、通常計算由来のprefixだけを共有登録する境界、workerへの伝達（[設計契約](apc-lpa-design.ja.md)） |
| `glm53_setup/runtime/fused_unpack.py`、`fused_nope*.py`、`graph_policy.py`、`patch_graph_prefill.py` | FP8 unpack融合kernel、実験的な融合NoPE attentionの試作、decode Graphの方針 |
| `glm53_setup/runtime/indexer_*.py`、`component_worker.py` | CSA2のindexer観測・再利用部品と、排他的な部品診断worker（[Indexer再利用](indexer-reuse.ja.md)） |
| `glm53_setup/runtime/pipeline_state.py`、`patch_pipeline*.py` | PP fixtureの転送とlayoutのpatch（P17） |
| `glm53_setup/cluster.py`、`switch.py`、`launch_assets.py`、`fabric.py` | 両rankの停止前検査、所有権つきの切替・復旧、読み取り専用の起動識別情報、RoCEレール検査（[起動契約](launch-safety.ja.md)） |
| `glm53_setup/model_http.py`、`io.py` | モデルAPIに限定しredirectに従わないHTTP transport、ローカル状態の永続化helper |
| `tests/` | CPU契約 |
| `tools/` | `check_publication.py`（公開監査）、`assess_benchmark.py`、`check_prefix_cache.py`、`nccl_probe.py`、`prepare_mtp_view.py` |
| `examples/zcode-hooks/` | ZCodeの既存ファイルガードhookと導入手順（[ハーネス](harnesses.ja.md)） |
| `LICENSES/` | 上流ライセンス原文の保持 |
| `state/`、`records/` | ローカルの可変状態と実験の証跡。配布対象外 |

CLIは、選択したコマンドが実際に必要とする場合にだけGPU依存をimportします。help、設定、CPUテストは、ホストにTorchやvLLMが入っていなくても動きます。GPUプログラムは固定imageの中で実行します。

コメント付きの `examples/server.example.toml` は、起動設定の完全なスキーマも兼ねます。TOMLの全体検査は `server_config.load` と、単独で呼び出せる `server.command` の境界で行います。`serve_args` は検査済みのprofileを受け取り、スキーマを読み直しません。内部の組み立て工程であり、入力検査の入口ではありません。fabric固有の小さなガードは独立したままです。

モデルIDとrevisionの設定元は[runtime.lock.json](../config/runtime.lock.json)の一つだけです。可変ファイルは、呼び出し元の作業ディレクトリに関係なくcheckoutを基点にします。本ツールキットは保守されたcheckoutから実行してください。汎用のPythonライブラリとしては提供していません。

reference imageは、vLLMのソース2ファイルについて完全なhashを確認してから変更します。選択したAttention候補はすべて保持します。runtimeの数値計算と検証ハーネスは別のモジュールにしてあるため、CLIコードを移動しても数学的な実装は変わりません。

## 検証の境界

ダウンロードの完了、checksumの合格、GPUスモーク、設定の解釈、Attentionの一致、fixtureの統合、フルモデルTP=2の検収は、それぞれ別種の証拠です。ある水準の結果を、別の水準の代わりにはできません。とくに、GPU 1台のfixtureは2 rankの証拠の代わりになりません。
