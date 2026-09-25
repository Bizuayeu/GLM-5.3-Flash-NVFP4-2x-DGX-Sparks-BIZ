# ハーネス連携と受け入れ試験

[English](harnesses.md) · [セットアップ](../SETUP.ja.md) · [ライセンス](licensing.ja.md) · [検証範囲](validation.ja.md)

ハーネスは、モデルと会話し、ファイル操作・ツール実行・履歴・承認を管理するクライアントです。GPU上の推論サーバーとは別の部品です。**本書はハーネスの接続方針、受け入れ試験一覧、その実施状態の唯一の正典です。** 他の文書は状態を書き写さず、本書を参照します。全クライアント試験の前提は、フルモデルTP=2の起動と基礎API群の合格です。

## 接続方針

| 経路 | 接続先 | 位置付け・判断 |
|---|---|---|
| 基礎API検査 | ローカルvLLMのChat Completions／Messages API | ハーネス固有の問題とサーバー側の問題を切り分ける |
| ZCode | Custom Provider → SSHトンネル → vLLMのOpenAI互換API | npm `zcode-app-cli` が受け入れた経路（2026-09-22）。DesktopはBLOCKED |
| Claude Code CLI | Anthropic形式 → SSHトンネル → vLLMのMessages API | 追わない（判断で見送り、2026-09-22） |

ZCodeは[公式サイト](https://zcode.z.ai/en)でGLM向けのハーネスとされ、[公式設定ガイド](https://zcode.z.ai/en/docs/configuration#custom-providers-anthropic--openai-compatible)は互換providerの追加を案内していますが、どちらもこのNVFP4・TP=2構成との互換を示すものではありません。固定vLLMは`/v1/messages`と`/v1/messages/count_tokens`を直接提供します（[Claude Codeガイド](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/serving/integrations/claude_code.md)、[APIルーター](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/entrypoints/anthropic/api_router.py)）。[Z.aiのClaude Code案内](https://docs.z.ai/devpack/tool/claude)は同社のクラウドサービス向けで、[Anthropicの案内](https://code.claude.com/docs/en/llm-gateway)は非Claudeモデルへのroutingをサポートしないと明記します。ベンダーのサポート、技術的な互換、利用条件は分けて扱います。

## ZCodeの配布形態

「ZCode」は一つの成果物ではありません。受け入れの対象も外向き通信の事実も配布形態で異なるため、本書では分けて呼びます。

| 配布形態 | 実体 | 本書での役割 |
|---|---|---|
| 公式Desktop GUI | [公式インストーラー](https://zcode.z.ai/en/docs/install)のElectronアプリ。providerはModel Settingsで追加する | 利用不可（BLOCKED、[feedback #270](https://github.com/zai-org/feedback/issues/270)）。受け入れ経路ではない |
| 公式Desktop同梱CLI | Desktopインストール内の`resources/glm/zcode.cjs` | 確認した版では対話起動できない。headlessの`--prompt`は動く（状態欄を参照） |
| npm `zcode-app-cli` | 非公式の[ターミナルラッパー](https://github.com/kingsword09/zcode-cli)。ZCode runtimeを同梱し独自TUIを足す（独自コードはMIT、ZCode本体は上流の条件） | 受け入れた経路（2026-09-22） |

## 接続の設定

以下は受け入れた経路で使った設定です（run `20260915-harness-h-sandbox`）。版に束縛されるため、クライアント更新後は再確認します。`server preflight` を迂回しません。クライアントの版と配布物hash、サーバーのソース・イメージ・revision・起動引数を先に記録します。APIポートは起動TOMLの値を使います（例示テンプレートはloopbackの8893）。

Windowsの別PowerShellでSSHトンネルを開きます（`node-a`は例示。別ファイルが必要なら`-F`を指定）。

```powershell
ssh -N -L 127.0.0.1:8893:127.0.0.1:8893 node-a
```

API側の認証を設定している場合は、そのローカルサービス専用の資格情報を使用します。Z.aiやAnthropicのクラウドキーをローカル試験へ流用しません。

**受け入れ試験中は、ハーネス自身の設定ディレクトリへの書き込みを拒否します**（ユーザープロファイル直下の`.zcode`）。build権限で動くハーネスは、依頼に応じて自分の設定を編集し、成功と報告したうえで次回起動に失敗することがあります。クライアント側のスキーマ拒否は、モデル設定が無いといった無関係なエラーとして現れます。ツールの拒否リストでそのパスを指定し、動作していた設定の複製を残します。`yolo`ではその拒否リストは参照されず、代わりに[既存ファイルガード](#zcodeの権限モードモデル上限既存ファイルガード)が同ディレクトリを守ります。

npm版CLIの設定ファイルは`~/.zcode/cli/config.json`です。OpenAI互換のproviderを`baseURL` `http://127.0.0.1:8893/v1`、model ID `glm-5.3-flash-nvidia`（項目は`provider.<id>.models.<model>`）で追加し、実際の要求パスで`/v1`が二重にならないことを確認します。ローカルAPIが認証を要求するなら一致する鍵を使い、無認証loopbackへ非空の値が要る場合だけ秘密でない試験用値を使います。それは認証保護にはなりません。

入力はサーバに合わせてテキスト・ツール・画像を宣言し（例示profileは二つとも `runtime.vision = true`。ZCodeでは `modalities.input = ["text", "image"]`）、サーバが拒否する動画は宣言しません。cloud providerへのfallback、外部連携、追加エージェントは既定にしません。通常モデルと補助（lite）モデルの両方をローカルのserved IDへ向けます。モデルの`limit.context`をサーバの`max_model_len`に合わせ、`limit.output`は32000に留め、`modelStream.idleTimeoutMs`を想定する最長のprefillより大きくします（[モデル上限](#zcodeの権限モードモデル上限既存ファイルガード)を参照）。選択モデルと実際の接続先を確認します。UIロケールはクライアントが文書化している値だけを受け付け（`zcode --help`に一覧）、非対応の値は設定ファイル全体を無効にします。

## ZCodeの権限モード・モデル上限・既存ファイルガード

以下はDesktop同梱runtime（`resources/glm/zcode.cjs`、Desktop 3.11.2、runtime 0.16.5、2026-09-14）の静的読解に基づきます。npm配布が同梱するruntimeも同じ版（0.16.5）です。版に束縛された事実であり、クライアント更新後は再確認します。受け入れケースを閉じるものではありません。

**モード。** 設定の列挙は`plan`／`build`／`edit`／`yolo`／`auto`です。Claude Code名を写す正規化関数は二つあり、セッション側は`bypassPermissions`／`dontAsk`を`yolo`、`acceptEdits`を`edit`へ、automation側は`acceptEdits`／`autoEdit`／`default`／`auto`を`build`へ写します。`auto`は予約のみで未実装であり、全ツールを拒否します。したがって`permission.allowMediumRiskInAuto`は効きません。headlessの`--prompt`は既定で`yolo`です。

**ツールのリスク記述子。** `Read`はlow・副作用なし。`Write`と`Edit`は一つの権限（`edit`、medium、workspace）を共有します。`Bash`はhigh・system。削除・リネーム専用ツールはなく、削除は`Bash`経由です。記述子のどこにも「新規作成」と「既存変更」の区別はありません。

**判定順。** planモード遷移 → 対話必須ツール → `yolo`は許可 → `auto`は拒否 → `disallowedTools` → プロジェクトの`deny`ルール → 同`ask`ルール → planモード検査 → 同`allow`ルール → `allowedTools` → `edit`モード（`Write`／`Edit`を許可） → `build`モード（読み取り専用は許可。critical、`autoApproveHighRisk`でないhigh、その他の副作用は確認）。帰結として、`build`は全ての書き込みとシェルを確認し、`edit`は`build`にファイル編集の無人化を足したもの、`yolo`はプロジェクトルールと`disallowedTools`より**前**に評価されるため、パスルールでは狭められません。

**hookの合成。** `PreToolUse` hookが返す`hookSpecificOutput.permissionDecision`はpolicyの結果と合成されます。`deny`は常に勝ち、`ask`はpolicyの`allow`（`mode.yolo`を含む）を確認に変え（`hook.PreToolUse.ask`）、`allow`はpolicyの`ask`を許可に変えます。主経路はhookの`ask`をプロジェクトルール限定のフィルタなしで承認brokerへ渡します。`yolo`を狭められる唯一の場所です。hookはstdinのJSONで`tool_name`、`tool_input`（`Write`／`Edit`は絶対パスの`file_path`と`content`、`Bash`は`command`）、`cwd`、`permission_mode`、`session_id`を受け取ります。

**ガード。** [examples/zcode-hooks/exists-guard.cjs](../examples/zcode-hooks/exists-guard.cjs)は「`yolo`で走らせ、既にあるものを変える時だけ確認する」を実装します。`Edit`は常に確認、`Write`は対象が存在すれば確認・新規なら許可、クライアント自身の`.zcode`配下への`Write`／`Edit`は確認、`Bash`は破壊的パターン（`rm`、`mv`、`cp`、`patch`、`Remove-Item`／`Copy-Item`、`git reset`／`clean`／`apply`、`/dev/null`と`NUL`以外へのリダイレクト等）に一致すれば確認・それ以外は許可、他のツールには関与しません。[examples/zcode-hooks/config.hooks.example.json](../examples/zcode-hooks/config.hooks.example.json)が`hooks.events.PreToolUse`の登録形を示します。導入・確認・限界は[hookのREADME](../examples/zcode-hooks/README.ja.md)にあります。

**確認した挙動（`20260914-zcode-exists-guard`、Desktop同梱CLIのheadless `--prompt`、ローカルモデル）。** 新規ファイルへの`Write`は許可され、ファイルが作られました。既存ファイルへの`Write`は`ask`に変わり、headlessには承認クライアントが無いためツール呼び出しが拒否され（"No permission client configured"）、ファイルは無変更でした。非破壊の`Bash`（`ls -la .`）は許可されて応答が返り、`rm doomed.txt`は`ask`に変わって同様に拒否され、ファイルは残りました（この2件の先立つ試行は`ECONNRESET`で失敗し、記録に残しています）。TUIでは同じ`ask`が確認プロンプトになります。

**モデル上限と圧縮予算。** `limit.context`はクライアントのcontext窓になります（既定200000）。`limit.output`は毎リクエストの`max_tokens`として送られます（既定32000。headless一問をloopback tapで採取すると`max_tokens: 32000`でした）。テンプレートが必ず出すthinkingブロックもvLLMはこれに数えるため、小さい値は思考＋本文を`finish_reason: length`で切ります。自動圧縮（`preflight-v1`）は`limit.context` − min(`limit.output`, 21000) − 13000で始まるため、残ったままの小さな値は極端に早く圧縮します（32768／4096なら約15.7Kトークン）。vLLMはprompt＋`max_tokens`が`max_model_len`を超える要求を拒否するので、`limit.context`を`max_model_len`に合わせていれば、サーバの長さによらず`limit.output`は34000未満でなければなりません。32000に留め、モデル公称の64K〜128Kへは上げず、`limit.context`はサーバの`max_model_len`と同時に変えます。既存セッションが作成時のモデル定義を保持するかは未確認なので、上限変更後は新しいセッションで始めます。

**prefix cacheとハーネスのcache表示。** ZCodeのcache表示には二つが要ります。(1) サーバが要求をcacheすること：LPAは例示profileの二つとも無効なので、素の要求はcacheされます。同じサーバでバッチがLPAを有効にすると、近似した要求は共有cacheに何も登録しないため、ハーネスからは`"vllm_xargs": {"glm53_lpa_mode": "off"}`を送ります。ZCodeでは`provider.<id>.models.<model>.options.extra_body.vllm_xargs.glm53_lpa_mode = "off"`で、実リクエストのトップレベル`vllm_xargs`に載ることを確認しています。(2) サーバがhitを返すこと：ZCodeは`usage.prompt_tokens_details.cached_tokens`から率を計算し、vLLMはこの欄を`api.prompt_tokens_details = true`（例示の二つとも有効）の時だけ返します。再利用は4,608 tokenの完全なblock単位で、MTPでは一致した最後のblockを除くため、短いやり取りとセッション最初のターンは0になります（[1.13.0での測定](benchmarks.ja.md#1130での測定)）。`python tools/check_prefix_cache.py --model <served id>` が両条件を確認し、満たさない方を示します。

**ストリームのidle timeout。** `modelStream.idleTimeoutMs`（bundleの既定は600000）は要求送信から最初のSSEイベントまでも数え、ローカルサーバはprefill中に何も流しません。期限切れでクライアントは中断し、同じpromptで再試行し、再試行ごとに30秒を足して最大11回まで繰り返すため、prefillがこの値を超える長文ターンは完了しません。試験設定の700000は、2026-09-14に当時のprofileで256Kへ比例で見積もった606秒から決めた値で、その後に測った最長の要求、256Kの容量要求（[1.6.0での測定](benchmarks.ja.md#160での測定)）と約200Kの2要求の同時実行（[1.10.2での測定](benchmarks.ja.md#1102での測定)）も収まります。`0`はtimerを無効にしますが停止検知を失います。`network.timeout`は補助HTTPクライアントに配線されており、モデルのストリームも縛るかは未確認です。

## ハーネスクライアントの外向き通信

ローカル推論の設定と、テレメトリの停止は別です。下表は各配布物のbundleを静的に読んだ結果であり、実通信の採取ではありません。実通信はH-09で確認します。

| 配布形態（確認した版） | モデル実行のtrace | その他の外向き通信 | 停止手段 |
|---|---|---|---|
| npm `zcode-app-cli`（3.11.2-24、runtime 0.16.5） | 環境変数`OTEL_EXPORTER_OTLP_ENDPOINT`または`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`がある時だけ送信。送信先の直書きなし | 更新確認、hostedモデルカタログ更新、公式plugin marketplaceの取得 | `ZCODE_MODEL_TELEMETRY_ENABLED=0`（`false`／`off`／`disabled`も可）、`ZCODE_DISABLE_UPDATE_CHECK=1`、`ZCODE_DISABLE_MODEL_CATALOG_REFRESH=1`、クライアント設定の`plugins.enabled=false` |
| 公式Desktop（3.11.2） | `*.cn-beijing.log.aliyuncs.com`配下のOTLP trace収集先と、`zcode.z.ai/api/v1/`配下のイベント報告先が直書き。継承した`OTEL_*`と`ZCODE_MODEL_TELEMETRY_ENABLED`を子プロセスの環境から削除してから直書き値を注入する | `cdn-zcode.z.ai`配下の更新フィード | 設定画面・環境変数のいずれにも見つからない。止めるにはDesktopを使わないか、ネットワーク側で該当ホストを遮断する。後者は本リポジトリの範囲外の運用者判断 |

環境変数はクライアントを起動するプロセスで設定し、既存プロセスは再起動します。ネットワーク側の遮断は本リポジトリでは設定しません。適用する場合はクライアントの実行ファイルに範囲を絞り、runの記録に残します。

H-09では、別のPowerShellから起動直後と推論中にクライアントと子プロセスのTCP接続を採取し、配布形態ごとに記録します。

```powershell
$zcodeProcesses = @(Get-CimInstance Win32_Process)
$zcodeIds = @($zcodeProcesses | Where-Object {
    $_.Name -eq 'ZCode.exe' -or
    ($_.Name -eq 'node.exe' -and $_.CommandLine -match 'zcode')
} | Select-Object -ExpandProperty ProcessId)
do {
    $previousCount = $zcodeIds.Count
    $zcodeIds = @($zcodeIds + @($zcodeProcesses | Where-Object {
        $_.ParentProcessId -in $zcodeIds
    } | Select-Object -ExpandProperty ProcessId) | Sort-Object -Unique)
} while ($zcodeIds.Count -gt $previousCount)
Get-NetTCPConnection -ErrorAction SilentlyContinue |
    Where-Object { $_.OwningProcess -in $zcodeIds } |
    Select-Object OwningProcess, State, RemoteAddress, RemotePort
```

一時点の採取は短時間の接続・UDP・DNSを取りこぼします。空の結果は「送信なし」の証明ではなく、アドレスだけではドメインも内容も特定できません。

**記録した採取（2026-09-14、`20260914-zcode-cli-tcp-sample`）:** npm配布を上記の停止手段（OTLP endpointなし、telemetryフラグoff、更新確認・カタログ更新無効、Z.aiログインなし）で起動し、headlessの一問にローカルモデルが回答する間、クライアントのnodeプロセスのTCP接続を0.5秒間隔で20秒間採取した。観測された外部endpointは、ローカルvLLM APIへのloopback SSHトンネルだけだった。この配布形態ではプロンプトとファイル内容がローカル経路に留まることの補助証拠であり、一問・TCPのみ・`plugins.enabled`はtrueのままで、H-09のクラウドfallback部分は試しておらず、公式Desktopについては何も言わない。

## 受け入れ試験で使うreasoning設定

この固定GLMテンプレートは常にassistantのthinkingブロックを開始し、オフ指定を読みません。思考は有効のまま使い、`thinking=false`／`enable_thinking=false`を送りません。[公式モデルカード](https://huggingface.co/zai-org/GLM-5.3-Flash)は`reasoning_effort=low/high/max`（既定max）を案内し、チャットには`clear_thinking=true`を推奨しています。lowはチャット試験のprofileであり、maxで行う公式品質評価の再現とは区別します。ZCodeでは`/effort <level>`（別名`/variant`。`/effort list`が選択肢を表示）でセッションのeffortを切り替えます。`glm-5.3`に一致するIDの選択肢は`low`／`high`／`max`で、`medium`と`xhigh`はGPT・Qwen向けにしかありません。値はproviderに`reasoning_effort`として届き、固定GLM chat templateは`low`と`high`だけを受理して、それ以外（未指定を含む）は`max`として扱います。`max`では思考が実質際限なく続き、通常のコーディングターンでも`max_tokens`と時間の大半をそこに費やすため、日常の作業は`high`以下にします。今回と一致するparser/template不整合は[vLLM #54744](https://github.com/vllm-project/vllm/issues/54744)で報告され、[修正PR #54825](https://github.com/vllm-project/vllm/pull/54825)は2026-09-26時点で未マージなので、任意のイメージに修正済みとは仮定しません。固定したvLLM `385dce36` には、XGrammarと投機デコードの修正[vLLM #53046](https://github.com/vllm-project/vllm/pull/53046)（`c6e19b3`）と[#52805](https://github.com/vllm-project/vllm/pull/52805)（`12f64b3`）が既に入っています（GitHubのcompareで両方に対してbehind 0）。

通常の受け入れは最終回答、構造化ツール要求、実行結果、承認境界で判定します。推論文・token列の完全再現やバッチ間ビット一致は別の数値診断へ分け、不一致の記録を残します。不一致をすべてタスク失敗とみなすことも、最終回答が一致しただけでモデル全体の正しさを証明したとみなすこともしません。ゴールデン検証は同時実行1、並列は性能・品質を分けて評価します。

## 受け入れ試験一覧と実施状態

状態の値はPASS、PARTIAL（一部の条件のみ合格。内訳を記す）、BLOCKED（実行不能。根拠と代替を記録）、NOT RUN。API群を先に行い、共通群Hは実リクエストと成果物で判定します。文書・ソースの存在や小型fixtureの合格では代用しません。状態は2026-09-22時点（runは2026-09-15、判断は2026-09-22）で、run IDは非公開`records/`を指します。

**run `20260915-harness-h-sandbox`（2026-09-15）。** 共通群H-01〜H-11の全11ケースを、npm `zcode-app-cli` 3.11.2-24のTUI（`yolo`＋既存ファイルガードhook、通常・liteともloopbackトンネル先の `glm-5.3-flash-nvidia`、`limit.context` 204800、`limit.output` 32000）で、バグを仕込んだ小さな試験repoとunittestを対象に一巡しました。下表のH行がこのrunで、PARTIALの欄には未試験の部分を記します。

**判断（2026-09-22）。** 受け入れた経路は、運用者が実際に使っているnpm `zcode-app-cli` 配布です。公式Desktop GUIは[feedback #270](https://github.com/zai-org/feedback/issues/270)でBLOCKEDのままで、必須の受け入れ対象から外します。Claude Codeは、同じ機体のサブスクリプション設定と競合するため判断で見送り、CC-01・CC-02とAPI-04の残り項目は「追わない」として閉じます。どちらもHケースは実行していません。PARTIALの行の未試験の小項目は受け入れ範囲の外とします。

| ID | 対象 | 操作と合格条件 | 状態 |
|---|---|---|---|
| API-01 | 基礎API | `/v1/models`のserved IDと選択先が一致。短い日本語・英語の要求がローカルモデルから正常応答する | PASS（`api-acceptance-low-local`） |
| API-02 | 基礎API | Chat Completionsの通常応答とSSE。終端・UTF-8・reasoning／最終回答の区別が壊れない。日本語・韓国語の長い出力は[`server mojibake`](validation.ja.md#フルモデルtp2の実験範囲)で別に監視 | PASS（同run。chatは`reasoning_effort=low`） |
| API-03 | 基礎API | 無害なツールの要求→JSON引数検証→結果返送→最終回答。複数往復でID対応を維持する | PASS（同run） |
| API-04 | Anthropic互換API | Messagesの通常応答・SSE・tool_use/tool_result・count_tokensを検査。応答形式・終端・usageを確認する | PARTIAL：Messagesの通常応答とcount_tokensはモデル既定effortで合格。SSE、tool_use/tool_result、異常系は追わない |
| ZC-01 | ZCode | Custom Providerへ登録したモデルが選べ、実際のendpointとmodel IDがローカル設定に一致する | DesktopはBLOCKED（[feedback #270](https://github.com/zai-org/feedback/issues/270)）：GUIは未実行、同梱CLIは`@zcode/tui`不在で対話起動に失敗、headlessの`--prompt`はローカルserved IDへ届く（`20260914-zcode-exists-guard`）。npm CLI：トンネル先のprovider、served IDの選択、日本語の往復、headless promptへのローカルモデル応答を確認 |
| ZC-02 | ZCode | 宣言した入力（テキスト・ツール・画像、動画なし）を尊重。クラウドの既定モデルへ無断で置換されない | DesktopはZC-01と同じくBLOCKED。npm CLI：通常・liteの両方がローカルserved IDに固定、カタログ更新は無効 |
| CC-01 | Claude Code | 専用設定で起動し、通常・補助の要求が指定served IDに届く。モデル名未解決や認証ループがない | NOT RUN — 判断で見送り（2026-09-22） |
| CC-02 | Claude Code | Anthropic形式のtool ID、分割JSON、reasoning、stop_reasonを正しく処理し、ツール結果後に会話を継続できる | NOT RUN — 判断で見送り（2026-09-22） |
| H-01 | 共通 | 小さな試験repoの2ファイルを読み、実内容に基づく説明を返す。未読内容を読んだと報告しない | PASS（`20260915-harness-h-sandbox`）：試験repoの2ファイルを実内容どおりに説明。未読のテストファイルは未読と明示 |
| H-02 | 共通 | 小さなバグを1件修正し、許可したファイルだけに意図したdiffが生じる | PASS（同run）：`calc.py` のoff-by-oneを除去。`git diff` はその1ファイルのみ |
| H-03 | 共通 | 許可したローカルテストを実行し、終了コードと実ログに一致する成否を報告する | PASS（同run）：red（テスト五件中二件失敗、exit 1）→ green（五件合格、exit 0）。報告は実ログと一致 |
| H-04 | 共通 | 無害なmarkerファイル作成を一度拒否し、作成されないことを確認。承認を迂回しない | PARTIAL（同run）：guardは `.zcode` 配下への `Write` を確認プロンプトに変え、承認の迂回はなし。ただしmarkerはクライアント側の承認後に作成されたため「一度拒否し、作成されない」条件は未達 |
| H-05 | 共通 | 読み取り→編集→検査の複数ツール往復で引数・結果・順序を維持する。追加エージェントなしの初期条件で実施 | PASS（同run）：読み取り・テスト・編集・diff・commit・書き込み・削除・設定読み・curl・bgジョブ起動と停止の約15往復で引数・結果・順序を維持 |
| H-06 | 共通 | 生成・ツール待ちを中断してから新規要求を送信できる。無限再試行・残留ジョブ・サーバー停止がない | PARTIAL（同run）：bgジョブを停止し残留プロセスなし、停止後もサーバーは200応答。生成中の割り込みと無限再試行の確認は未試験 |
| H-07 | 共通 | クライアント再起動後に試験会話を再開し、同じローカル接続先と承認設定を維持する | PASS（同run）：クライアント再起動後に同一セッションIDで再開し、接続先・モデル上限・権限モード・hookが再起動前と一致 |
| H-08 | 共通 | 実サーバーのcontext上限付近を試す。必要な圧縮または明示的エラーで処理し、履歴を黙って失わない。200k／1M対応を仮定しない | PARTIAL（同run）：`limit.context` は当時の200K profileの204800で、サーバーの `/v1/models` の `max_model_len` と一致、`limit.output` は32000。上限付近の挙動と圧縮は未試験。現行profileは262,144で、256Kでは再試験していない |
| H-09 | 共通 | 実要求の接続先を確認し、ローカルendpoint停止時にクラウド推論へfallbackしない。その他の通信も記録し「完全オフライン」と混同しない | PARTIAL（同run）：モデル経路はloopbackトンネルでfallback設定なし。停止時fallbackは未試験。同runの付随通信はユーザー依頼の `curl` 2件（`api.fxtwitter.com`、`pbs.twimg.com`）。クライアントのtelemetryは未採取（上記2026-09-14のTCP採取を参照） |
| H-10 | 共通 | 同じ試験repo・課題で一連の読解、修正、テスト、最終説明を完遂。APIログと成果物、正確性、遅延を保存する | PARTIAL（同run）：読解・修正・テスト・報告の課題を完遂し、成果物と正確性はrun logとGit履歴に保存。APIログと遅延計測は未取得 |
| H-11 | 共通 | 対応するreasoning設定がローカルserviceへ届き、非対応のオフ引数が送られず、推論文が最終contentへ漏れない。effortの変換に非対応なら明記する | PARTIAL（同run）：オフ引数は設定・送信ともになし、最終回答への推論文の漏れなし。リクエスト中の `reasoning_effort` は未採取 |

H-08の上限は起動TOMLの実サーバー設定を正典とします。並列エージェント・MCP・画像は初回合格後の別試験です。

## 記録と合否

非公開`records/<run-id>/`に、case ID、harness名・配布形態・版・hash、設定の非秘密部分、サーバーとモデルの固定値、期待結果、実結果、PASS／PARTIAL／FAIL／BLOCKED／NOT RUN、要求ID・ログ・diff・テスト結果の所在を保存します。ケースの状態が変わったら、同じ変更で上表を更新します。

ハーネスごと、ZCodeは配布形態ごとに分けて集計します。片方の合格で他方を完了扱いにしません。必須ケースは受け入れた経路のケースです。非対応・未実装・契約上の制約が判明した場合もそれを消さず、BLOCKEDと根拠・代替案を記録します。API互換性の不足が確認できた場合だけ、上流修正または小さな変換アダプターを検討し、そのライセンスと追加テストを明記します。
