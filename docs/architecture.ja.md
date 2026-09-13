# 構成

[English](architecture.ja.md)

本プロジェクトは、checkout内で完結する運用者向けのツールキットです。モデル重みも、遠隔管理されるサービスも含みません。

| 場所 | 責務 |
|---|---|
| `glm53_setup/__main__.py` | 固定したコマンド振り分け。利用者が指定するモジュールの動的読込は行わない |
| `glm53_setup/config.py` | checkout内のパスと、検査済みの固定設定 |
| `glm53_setup/startup.py`、`startup_config.py` | 実験用の起動・クライアント制御と、カテゴリ別のTOML設定 |
| `glm53_setup/download.py`、`images.py`、`build_reference.py`、`service.py` | 資材の準備と、ガード付きのローカル操作 |
| `glm53_setup/validation/` | 明示的なCPU／GPU検査、fixtureの作成と判定 |
| `glm53_setup/runtime/` | sourceを固定したNoPE適合と、候補を保存する参照計算 |
| `glm53_setup/runtime/lpa.py`、`lpa_query.py` | LPAのworker制御、Attention入力の近似、要求単位のquery省略 |
| `glm53_setup/validation/run_lpa.py`、`lpa_corpus.py`、`train_lpa.py` | LPA fixtureの検査、コーパスの準備、projectorの学習 |
| `config/` | モデル・imageの固定値。認証情報や実測したサイト設定は持たない |
| `examples/` | 例示値だけを含むサイト設定のテンプレート |
| `docker/` | imageの構築。base digestはビルドコマンドがロックから渡す |
| `requirements/` | ホスト側ツールの固定した依存 |
| `tests/`、`tools/` | CPU契約と公開監査 |
| `LICENSES/` | 上流ライセンス原文の保持 |
| `state/`、`records/` | ローカルの可変状態と実験の証跡。配布対象外 |

CLIは、選択したコマンドが実際に必要とする場合にだけGPU依存をimportします。help、設定、CPUテストは、ホストにTorchやvLLMが入っていなくても動きます。GPUプログラムは固定imageの中で実行します。

コメント付きの `examples/startup.example.toml` は、起動設定の完全なスキーマも兼ねます。TOMLの全体検査は `startup_config.load` と、単独で呼び出せる `startup.command` の境界で行います。`serve_args` は検査済みのprofileを受け取り、スキーマを読み直しません。内部の組み立て工程であり、入力検査の入口ではありません。fabric固有の小さなガードは独立したままです。

モデルIDとrevisionの設定元は[runtime.lock.json](../config/runtime.lock.json)の一つだけです。可変ファイルは、呼び出し元の作業ディレクトリに関係なくcheckoutを基点にします。本ツールキットは保守されたcheckoutから実行してください。汎用のPythonライブラリとしては提供していません。

reference imageは、vLLMのソース2ファイルについて完全なhashを確認してから変更します。選択したAttention候補はすべて保持します。runtimeの数値計算と検証ハーネスは別のモジュールにしてあるため、CLIコードを移動しても数学的な実装は変わりません。

## 検証の境界

ダウンロードの完了、checksumの合格、GPUスモーク、設定の解釈、Attentionの一致、fixtureの統合、フルモデルTP=2の検収は、それぞれ別種の証拠です。ある水準の結果を、別の水準の代わりにはできません。とくに、GPU 1台のfixtureでTP=2の起動ゲートは開きません。
