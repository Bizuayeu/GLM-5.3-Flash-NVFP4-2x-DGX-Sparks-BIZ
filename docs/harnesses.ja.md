# ハーネス連携と受け入れ試験

[English](harnesses.md) · [セットアップ](../SETUP.ja.md) · [ライセンス](licensing.ja.md)

ハーネスは、モデルと会話し、ファイル操作・ツール実行・履歴・承認を管理するクライアントです。GPU上の推論サーバーとは別の部品です。**公式ZCodeとClaude Codeの両経路を必須対象とし、クライアント試験はNOT RUNです。基礎APIには別途スモーク結果がありますが、ハーネス連携確認済みではありません。** フルモデルTP=2の起動・API検収が前提です。

## 接続方針

| 経路 | 接続先 | 位置付け・判断 |
|---|---|---|
| 基礎API検査 | ローカルvLLMのChat Completions／Messages API | ハーネス固有の問題とサーバー側の問題を切り分ける |
| ZCode | Custom Provider → SSHトンネル → vLLMのOpenAI互換API | Z.ai公式ハーネス。標準経路として検証する |
| Claude Code CLI | Anthropic形式 → SSHトンネル → vLLMのMessages API | 実験的互換性経路。Anthropicによる非Claudeモデルのサポートとは区別する |

ZCodeは[公式サイト](https://zcode.z.ai/en)でGLM向けの公式ハーネスとされ、[公式設定ガイド](https://zcode.z.ai/en/docs/configuration#custom-providers-anthropic--openai-compatible)はローカル等の互換provider追加を案内しています。「公式ハーネス」であっても、このNVFP4・TP=2構成まで検証済みという意味ではありません。

固定vLLM版の[Claude Codeガイド](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/serving/integrations/claude_code.md)と[APIルーター](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/entrypoints/anthropic/api_router.py)には`/v1/messages`、`/v1/messages/count_tokens`の実装があります。そのため**追加ゲートウェイなしの直接接続を第一候補**とします。変換部品を減らせる一方、vLLMのAPI変換・GLMのツール解析・クライアントの版差を実測する必要があります。

[Z.aiのClaude Code案内](https://docs.z.ai/devpack/tool/claude)は主に同社クラウドサービスへの接続です。一方、[Anthropicの現行案内](https://code.claude.com/docs/en/llm-gateway)は非Claudeモデルへのroutingをサポートしないと明記します。前者の案内を後者の公式サポート保証と解釈せず、技術的な動作確認と利用契約を別々に扱います。

## 初回接続の候補手順

以下は**フルモデル起動後に試す設定案**です。今のベータ版の起動ガードを迂回しません。実行時にはクライアントの版と配布元、サーバーのGitコミット・イメージID・model revision・起動引数を記録します。

管理用SSH設定のホスト名を使い、Windowsの別PowerShellでトンネルを開きます（`node-a`は例示）。

```powershell
ssh -N -L 127.0.0.1:8891:127.0.0.1:8891 node-a
```

実際のSSH設定に別ファイルが必要なら`-F`を指定します。API側の認証を設定している場合は、そのローカルサービス専用の資格情報を使用します。Z.aiやAnthropicのクラウドキーをローカル試験へ流用しません。

### ZCode

1. 公式配布元の版を記録し、専用の試験workspace・providerを作る。
2. Model Settings → Add Providerから、OpenAI互換のローカルproviderを追加する。
3. Base URLをトンネル先の`http://127.0.0.1:8891/v1`、model IDを`glm-5.3-flash-nvidia`として試す。実際の要求パスで`/v1`が二重にならないことを確認する。
4. ローカルAPIが認証を要求するなら一致する鍵を使う。無認証loopbackへUIが非空キーを要求する場合だけ、秘密でない試験用値を使う。それは認証保護にはならない。
5. 試験ではテキスト・ツールのみを宣言し、画像、cloud providerへのfallback、外部連携、追加エージェントを既定にしない。選択モデルと実際の接続先を確認する。

### Claude Code CLI

固定vLLMの接続例を基にした、**専用PowerShellセッション内だけ**の設定案です。空の試験workspaceで実行し、普段のprofile・資格情報を共有しないでください。`CLAUDE_CONFIG_DIR`等の対応は対象CLI版で確認します。既存のグローバル設定ファイルは編集しません。

```powershell
$env:CLAUDE_CONFIG_DIR = "$PWD\.claude-local-test"
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8891'
$env:ANTHROPIC_API_KEY = 'local-test'
$env:ANTHROPIC_AUTH_TOKEN = 'local-test'
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = 'glm-5.3-flash-nvidia'
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = 'glm-5.3-flash-nvidia'
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'glm-5.3-flash-nvidia'
claude --model glm-5.3-flash-nvidia
```

このbase URLには`/v1`を付けず、CLIが`/v1/messages`等を追加します。`local-test`は無認証loopback用の非秘密placeholderです。認証付きAPIでは正しい専用資格情報へ置き換えます。上位・通常・軽量のモデル名を同じserved IDへ向け、補助要求だけクラウドへ出ないことも検査します。ログイン・利用条件の確認が必要なら公式手順に従い、認証チェックを改変して回避しません。

この案はCLI用です。Claude Desktop、Web、Remote Controlの接続方式まで同じと仮定しません。[クライアント別の公式案内](https://code.claude.com/docs/en/llm-gateway-connect)を参照してください。CLIを閉じてもサーバーは停止しません。

## 受け入れ試験一覧

この固定GLMテンプレートは常にassistantのthinkingブロックを開始し、オフ指定を読みません。`thinking=false`／`enable_thinking=false`を送らず、思考を有効のまま使います。[公式モデルカード](https://huggingface.co/zai-org/GLM-5.3-Flash)は`reasoning_effort=low/high/max`（既定max）を案内し、チャットには`clear_thinking=true`を推奨しています。lowはチャット試験のprofileであり、maxで行う公式品質評価の再現とは区別します。今回と一致するparser/template不整合は[vLLM #54744](https://github.com/vllm-project/vllm/issues/54744)で報告され、[修正PR #54825](https://github.com/vllm-project/vllm/pull/54825)は確認時点で未マージでした。任意のイメージに修正済みとは仮定しません。

通常の受け入れは最終回答、構造化ツール要求、実行結果、承認境界で判定します。推論文・token列の完全再現やバッチ間ビット一致は別の数値診断へ分け、不一致の記録を残します。不一致をすべてタスク失敗とみなすことも、最終回答が一致しただけでモデル全体の正しさを証明したとみなすこともしません。ゴールデン検証は同時実行1、並列は性能・品質を分けて評価します。

**ZCode／Claude Codeのクライアント試験はNOT RUNです。ローカルAPIには初期スモークの証跡がありますが、Anthropic streamingや境界・異常系を含むAPI一覧の全検収は未完了です。** API群の後に共通群Hを両ハーネスで別々に実施し、実リクエストと成果物で判定します。文書・ソースの存在や小型fixtureの合格では代用しません。

| ID | 対象 | 操作と合格条件 |
|---|---|---|
| API-01 | 基礎API | `/v1/models`のserved IDと選択先が一致。短い日本語・英語の要求がローカルモデルから正常応答する |
| API-02 | 基礎API | Chat Completionsの通常応答とSSE。終端・UTF-8・reasoning／最終回答の区別が壊れない |
| API-03 | 基礎API | 無害なツールの要求→JSON引数検証→結果返送→最終回答。複数往復でID対応を維持する |
| API-04 | Anthropic互換API | Messagesの通常応答・SSE・tool_use/tool_result・count_tokensを検査。応答形式・終端・usageを確認する |
| ZC-01 | ZCode | Custom Providerへ登録したモデルが選べ、実際のendpointとmodel IDがローカル設定に一致する |
| ZC-02 | ZCode | 画像非対応の初期プロファイルを尊重。クラウドの既定モデルへ無断で置換されない |
| CC-01 | Claude Code | 専用設定で起動し、通常・補助の要求が指定served IDに届く。モデル名未解決や認証ループがない |
| CC-02 | Claude Code | Anthropic形式のtool ID、分割JSON、reasoning、stop_reasonを正しく処理し、ツール結果後に会話を継続できる |
| H-01 | 両方 | 小さな試験repoの2ファイルを読み、実内容に基づく説明を返す。未読内容を読んだと報告しない |
| H-02 | 両方 | 小さなバグを1件修正し、許可したファイルだけに意図したdiffが生じる |
| H-03 | 両方 | 許可したローカルテストを実行し、終了コードと実ログに一致する成否を報告する |
| H-04 | 両方 | 無害なmarkerファイル作成を一度拒否し、作成されないことを確認。承認を迂回しない |
| H-05 | 両方 | 読み取り→編集→検査の複数ツール往復で引数・結果・順序を維持する。追加エージェントなしの初期条件で実施 |
| H-06 | 両方 | 生成・ツール待ちを中断してから新規要求を送信できる。無限再試行・残留ジョブ・サーバー停止がない |
| H-07 | 両方 | クライアント再起動後に試験会話を再開し、同じローカル接続先と承認設定を維持する |
| H-08 | 両方 | 実サーバーのcontext上限付近を試す。必要な圧縮または明示的エラーで処理し、履歴を黙って失わない。200k／1M対応を仮定しない |
| H-09 | 両方 | 実要求の接続先を確認し、ローカルendpoint停止時にクラウド推論へfallbackしない。その他の通信も記録し「完全オフライン」と混同しない |
| H-10 | 両方 | 同じ試験repo・課題で一連の読解、修正、テスト、最終説明を完遂。APIログと成果物、正確性、遅延を保存する |
| H-11 | 両方 | 対応するreasoning設定がローカルserviceへ届き、非対応のオフ引数が送られず、推論文が最終contentへ漏れない。effortの変換に非対応なら明記する |

H-08の上限は実際のサーバー設定を正典とします。候補ランチャーは32,768ですが、クライアント側のcontext認識との整合は未確認です。並列エージェント・MCP・画像は初回合格後の別試験です。

## 記録と合否

非公開`records/<run-id>/`に、case ID、harness名・版・配布物hash、設定の非秘密部分、サーバーとモデルの固定値、期待結果、実結果、PASS／FAIL／BLOCKED／NOT RUN、要求ID・ログ・diff・テスト結果の所在を保存します。

ZCodeとClaude Codeの結果は分けて集計します。片方の合格で他方を完了扱いにしません。非対応・未実装・契約上の制約が判明した場合も必須ケースを消さず、BLOCKEDと根拠・代替案を記録します。API互換性の不足が確認できた場合だけ、上流修正または小さな変換アダプターを検討し、そのライセンスと追加テストを明記します。
