# ハーネス連携と受け入れ試験

[English](harnesses.md) · [セットアップ](../SETUP.ja.md) · [ライセンス](licensing.ja.md) · [検証範囲](validation.ja.md)

ハーネスは、モデルと会話し、ファイル操作・ツール実行・履歴・承認を管理するクライアントです。GPU上の推論サーバーとは別の部品です。**本書はハーネスの接続方針、受け入れ試験一覧、その実施状態の唯一の正典です。** 他の文書は状態を書き写さず、本書を参照します。全クライアント試験の前提は、フルモデルTP=2の起動と基礎API群の合格です。

## 接続方針

| 経路 | 接続先 | 位置付け・判断 |
|---|---|---|
| 基礎API検査 | ローカルvLLMのChat Completions／Messages API | ハーネス固有の問題とサーバー側の問題を切り分ける |
| ZCode | Custom Provider → SSHトンネル → vLLMのOpenAI互換API | Z.ai公式ハーネス。標準経路として検証する |
| Claude Code CLI | Anthropic形式 → SSHトンネル → vLLMのMessages API | 実験的互換性経路。Anthropicによる非Claudeモデルのサポートとは区別する |

ZCodeは[公式サイト](https://zcode.z.ai/en)でGLM向けの公式ハーネスとされ、[公式設定ガイド](https://zcode.z.ai/en/docs/configuration#custom-providers-anthropic--openai-compatible)はローカル等の互換provider追加を案内しています。「公式ハーネス」であっても、このNVFP4・TP=2構成まで検証済みという意味ではありません。

固定vLLM版の[Claude Codeガイド](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/docs/serving/integrations/claude_code.md)と[APIルーター](https://github.com/vllm-project/vllm/blob/385dce36bcee42309924a5ece951a96db3dce7f2/vllm/entrypoints/anthropic/api_router.py)には`/v1/messages`、`/v1/messages/count_tokens`の実装があります。そのため**追加ゲートウェイなしの直接接続を第一候補**とします。変換部品を減らせる一方、vLLMのAPI変換・GLMのツール解析・クライアントの版差を実測する必要があります。

[Z.aiのClaude Code案内](https://docs.z.ai/devpack/tool/claude)は主に同社クラウドサービスへの接続です。一方、[Anthropicの現行案内](https://code.claude.com/docs/en/llm-gateway)は非Claudeモデルへのroutingをサポートしないと明記します。前者の案内を後者の公式サポート保証と解釈せず、技術的な動作確認と利用契約を別々に扱います。

## ZCodeの配布形態

「ZCode」は一つの成果物ではありません。受け入れの対象も外向き通信の事実も配布形態で異なるため、本書では分けて呼びます。

| 配布形態 | 実体 | 本書での役割 |
|---|---|---|
| 公式Desktop GUI | [公式インストーラー](https://zcode.z.ai/en/docs/install)のElectronアプリ。providerはModel Settingsで追加する | 必須の受け入れ対象 |
| 公式Desktop同梱CLI | Desktopインストール内の`resources/glm/zcode.cjs` | 確認した版では対話起動できない。headlessの`--prompt`は動く（状態欄を参照） |
| npm `zcode-app-cli` | 非公式の[ターミナルラッパー](https://github.com/kingsword09/zcode-cli)。ZCode runtimeを同梱し独自TUIを足す（独自コードはMIT、ZCode本体は上流の条件） | 補助証拠のみ。公式対象のケースを閉じない |

## 初回接続の候補手順

以下は**フルモデル起動後に試す設定案**であり、検収済みの配備手順ではありません。`server preflight` を迂回しません。実行時にはクライアントの版と配布物hash、サーバーのソース・イメージ・revision・起動引数を先に記録します。APIポートは起動TOMLの値を使います（例示テンプレートはloopbackの8893）。

管理用SSH設定のホスト名を使い、Windowsの別PowerShellでトンネルを開きます（`node-a`は例示。別ファイルが必要なら`-F`を指定）。

```powershell
ssh -N -L 127.0.0.1:8893:127.0.0.1:8893 node-a
```

API側の認証を設定している場合は、そのローカルサービス専用の資格情報を使用します。Z.aiやAnthropicのクラウドキーをローカル試験へ流用しません。

**受け入れ試験中は、ハーネス自身のクライアント設定ディレクトリへの書き込みを拒否します**（ZCodeはユーザープロファイル直下の`.zcode`、Claude Codeは`CLAUDE_CONFIG_DIR`で指定した場所）。build権限で動くハーネスは、依頼に応じて自分の設定を編集し、成功と報告したうえで次回起動に失敗することがあります。クライアント側のスキーマ拒否は、モデル設定が無いといった無関係なエラーとして現れます。ツールの拒否リストでそのパスを指定し、動作していた設定の複製を残します。ZCodeの`yolo`ではその拒否リストは参照されないため、[既存ファイルガード](#zcodeの権限モードモデル上限既存ファイルガード)が同ディレクトリへの`Write`／`Edit`を確認に変えます。シェル経由の書き込みは同ガードの破壊的パターン判定に委ねられます。

### ZCode

Model Settings → Add Providerから、専用の試験workspace・providerを作ります。OpenAI互換のローカルproviderを選び、Base URLをトンネル先の`http://127.0.0.1:8893/v1`、model IDを`glm-5.3-flash-nvidia`として試します。実際の要求パスで`/v1`が二重にならないことを確認します。ローカルAPIが認証を要求するなら一致する鍵を使い、無認証loopbackへUIが非空キーを要求する場合だけ秘密でない試験用値を使います。それは認証保護にはなりません。

入力はサーバに合わせてテキスト・ツール・画像を宣言し（配布構成は `runtime.vision = true`。ZCodeでは `modalities.input = ["text", "image"]`）、サーバが拒否する動画は宣言しません。cloud providerへのfallback、外部連携、追加エージェントは既定にしません。通常モデルと補助（lite）モデルの両方をローカルのserved IDへ向けます。モデルの`limit.context`をサーバの`max_model_len`に合わせ、`limit.output`は32000に留め、`modelStream.idleTimeoutMs`を長文prefillの実測より大きくします（[モデル上限](#zcodeの権限モードモデル上限既存ファイルガード)を参照）。選択モデルと実際の接続先を確認します。UIロケールはクライアントが文書化している値だけを受け付け（`zcode --help`に一覧）、非対応の値は設定ファイル全体を無効にします。

### Claude Code CLI

固定vLLMの接続例を基にした、**専用PowerShellセッション内だけ**の設定案です。空の試験workspaceで実行し、普段のprofile・資格情報を共有しません。`CLAUDE_CONFIG_DIR`等の対応は対象CLI版で確認します。既存のグローバル設定ファイルは編集しません。

```powershell
$env:CLAUDE_CONFIG_DIR = "$PWD\.claude-local-test"
$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8893'
$env:ANTHROPIC_API_KEY = 'local-test'
$env:ANTHROPIC_AUTH_TOKEN = 'local-test'
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = 'glm-5.3-flash-nvidia'
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = 'glm-5.3-flash-nvidia'
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'glm-5.3-flash-nvidia'
claude --model glm-5.3-flash-nvidia
```

このbase URLには`/v1`を付けず、CLIが`/v1/messages`等を追加します。`local-test`は無認証loopback用の非秘密placeholderです。認証付きAPIでは正しい専用資格情報へ置き換えます。上位・通常・軽量のモデル名を同じserved IDへ向け、補助要求だけクラウドへ出ないことも検査します。ログイン・利用条件の確認が必要なら公式手順に従い、認証チェックを改変して回避しません。

この案はCLI用です。Claude Desktop、Web、Remote Controlの接続方式まで同じと仮定しません。[クライアント別の公式案内](https://code.claude.com/docs/en/llm-gateway-connect)を参照してください。CLIを閉じてもサーバーは停止しません。

## ZCodeの権限モード・モデル上限・既存ファイルガード

以下はDesktop同梱runtime（`resources/glm/zcode.cjs`、Desktop 3.11.2、runtime 0.16.5、2026-09-14）の静的読解に基づきます。版に束縛された事実であり、クライアント更新後は再確認します。受け入れケースを閉じるものではありません。

**モード。** 設定の列挙は`plan`／`build`／`edit`／`yolo`／`auto`です。Claude Code名を写す正規化関数は二つあり、セッション側は`bypassPermissions`／`dontAsk`を`yolo`、`acceptEdits`を`edit`へ、automation側は`acceptEdits`／`autoEdit`／`default`／`auto`を`build`へ写します。`auto`は予約のみで未実装であり、全ツールを拒否します。したがって`permission.allowMediumRiskInAuto`は効きません。headlessの`--prompt`は既定で`yolo`です。

**ツールのリスク記述子。** `Read`はlow・副作用なし。`Write`と`Edit`は一つの権限（`edit`、medium、workspace）を共有します。`Bash`はhigh・system。削除・リネーム専用ツールはなく、削除は`Bash`経由です。記述子のどこにも「新規作成」と「既存変更」の区別はありません。

**判定順。** planモード遷移 → 対話必須ツール → `yolo`は許可 → `auto`は拒否 → `disallowedTools` → プロジェクトの`deny`ルール → 同`ask`ルール → planモード検査 → 同`allow`ルール → `allowedTools` → `edit`モード（`Write`／`Edit`を許可） → `build`モード（読み取り専用は許可。critical、`autoApproveHighRisk`でないhigh、その他の副作用は確認）。帰結として、`build`は全ての書き込みとシェルを確認し、`edit`は`build`にファイル編集の無人化を足したもの、`yolo`はプロジェクトルールと`disallowedTools`より**前**に評価されるため、パスルールでは狭められません。

**hookの合成。** `PreToolUse` hookが返す`hookSpecificOutput.permissionDecision`はpolicyの結果と合成されます。`deny`は常に勝ち、`ask`はpolicyの`allow`（`mode.yolo`を含む）を確認に変え（`hook.PreToolUse.ask`）、`allow`はpolicyの`ask`を許可に変えます。主経路はhookの`ask`をプロジェクトルール限定のフィルタなしで承認brokerへ渡します。`yolo`を狭められる唯一の場所です。hookはstdinのJSONで`tool_name`、`tool_input`（`Write`／`Edit`は絶対パスの`file_path`と`content`、`Bash`は`command`）、`cwd`、`permission_mode`、`session_id`を受け取ります。

**ガード。** [examples/zcode-hooks/exists-guard.cjs](../examples/zcode-hooks/exists-guard.cjs)は「`yolo`で走らせ、既にあるものを変える時だけ確認する」を実装します。`Edit`は常に確認、`Write`は対象が存在すれば確認・新規なら許可、クライアント自身の`.zcode`配下への`Write`／`Edit`は確認（前述の拒否リストは`yolo`下では効かないため）、`Bash`は破壊的パターン（`rm`、`mv`、`cp`、`patch`、`Remove-Item`／`Copy-Item`、`git reset`／`clean`／`apply`、`/dev/null`と`NUL`以外へのリダイレクト等）に一致すれば確認・それ以外は許可、他のツールには関与しません。[examples/zcode-hooks/config.hooks.example.json](../examples/zcode-hooks/config.hooks.example.json)が`hooks.events.PreToolUse`の登録形（`type: "process"`、`command: "node"`）と`hooks.enabled`の有効化を示します。スクリプトはリポジトリ外に複製し、動作していた設定の複製を残し、スキーマ拒否された項目が設定ファイル全体を無効にすることを前提にします。runtimeはhookの終了コード0を判定、2を拒否として読み、それ以外の終了・起動失敗・timeoutは該当ツール呼び出しの失敗になります。GUIはシェルの`PATH`を継承しないため、適用する設定ではインタプリタを絶対パスで指定し、項目のtimeoutは上位の`hooks.timeoutMs`に委ねます。

**確認した挙動（`20260914-zcode-exists-guard`、Desktop同梱CLIのheadless `--prompt`、ローカルモデル）。** 新規ファイルへの`Write`は許可され、ファイルが作られました。既存ファイルへの`Write`は`ask`に変わり、headlessには承認クライアントが無いためツール呼び出しが拒否され（"No permission client configured"）、ファイルは無変更でした。非破壊の`Bash`（`ls -la .`）は許可されて応答が返り、`rm doomed.txt`は`ask`に変わって同様に拒否され、ファイルは残りました（この2件はローカルAPI復旧後に実施。先立つ試行は`ECONNRESET`で失敗し、記録に残しています）。TUIやDesktopでは同じ`ask`が確認ダイアログになります。Desktop GUIは未実施です。

**モデル上限と圧縮予算。** custom modelの各項目は`limit.context`と`limit.output`を持ちます。`limit.context`はクライアントのcontext窓になり（未設定なら200000）、`limit.output`は`maxOutputTokens`として毎リクエストの`max_tokens`に載ります（未設定なら32000。項目を32000にしてheadless一問をloopback tapで採取すると`/v1/chat/completions`に`max_tokens: 32000`が付いていました）。固定GLMテンプレートは常にthinkingブロックを出し、vLLMはそれも`max_tokens`に数えるため、小さい値は思考＋本文を`finish_reason: length`で切ります。自動圧縮（既定strategy `preflight-v1`）は有効窓＝`limit.context` − min(`limit.output`, 21000)、発火＝有効窓 − 13000です。帰結は二つ。残ったままの小さな値は極端に早く圧縮を始め（32768／4096なら約15.7Kトークン）、また`limit.output`は`max_model_len` −（有効窓 − 13000）より小さく保つ必要があります。vLLMはprompt＋`max_tokens`が`max_model_len`を超える要求を拒否するためで、`limit.context`を`max_model_len`に合わせていれば、サーバの上限が204,800でも262,144でも、その上限は約34000です。`limit.output`は32000に留め、モデル公称の64K〜128Kへは上げません。`limit.context`はサーバの`max_model_len`に合わせ、サーバprofileを変えたら両方を同時に変えます。既存セッションが作成時のモデル定義を保持するかは未確認なので、上限変更後は新しいセッションで始めます。

**prefix cacheとハーネスのcache表示。** ZCodeのcache hit表示は二層で決まります。サーバ側では、LPA優先profile（P22）は近似した要求を共有cacheに登録せず、チャットハーネスのターンは常に`lpa.break_even_tokens`を超える残余を持つため、既定の要求modeでは共有prefixが育ちません。2026-09-14のmetrics採取では`vllm:prefix_cache_queries_total` 952,029に対し`vllm:prefix_cache_hits_total`は0でした。本リポが文書化している逃げ道は要求単位の`"vllm_xargs": {"glm53_lpa_mode": "off"}`で、ZCodeでは`provider.<id>.models.<model>.options.extra_body.vllm_xargs.glm53_lpa_mode = "off"`として設定し、実リクエストのトップレベル`vllm_xargs`に載ることをloopback tapで確認しました。これによりZCodeの要求は通常計算になります。ZCodeをLPAで走らせることとそのcacheを育てることは両立しません。配布テンプレートは `lpa.enabled = false` を既定にしたので素の要求でもcacheは育ちます（同一19,559 tokenの2回目で13,824 token復元を実測）。要求単位のopt-outは、同じサーバでLPAを有効にしたバッチが走る間もハーネス側を通常計算に保つために残しています。クライアント側では、ZCodeは`usage.prompt_tokens_details.cached_tokens`から率を計算しますが、vLLMは`--enable-prompt-tokens-details`（起動TOMLの`api.prompt_tokens_details = true`）がないとこの欄を返さず、cacheが当たっていても表示は0のままです。再利用はblock単位でもあります。この hybrid モデルは attention block を mamba page size に合わせて整列するため、起動時に `kv lcm block sizes 4608` が出て、復元は 4,608 token の完全な block 単位でしか起きません。これより短いやり取りは原理的にヒットしません。両層を直したうえで同一の 16,859 token を 2 回送ると、2 回目の `cached_tokens` は 9,216（`prefix_cache_hits_total` は 0 から 9,216）でした。3 blockでなく 2 block なのは、MTPが一致した末尾blockをhitから除外するためです。セッション最初のターンは常に0なので、100%未満が正常です。`python tools/check_prefix_cache.py --model <served id>` がこの確認を行い、何も復元されない場合はどちらの原因かを示します。ガードのスクリプトと導入手順は [examples/zcode-hooks/](../examples/zcode-hooks/) にあります。

**ストリームのidle timeout。** `modelStream.idleTimeoutMs`（bundleの既定600000、試験設定では60000）は要求送信から最初のSSEイベントまでも数え、ローカルサーバはprefill中に何も流しません。期限切れでクライアントは中断し、同じpromptで再試行し、再試行ごとに30秒を足して最大11回まで繰り返すため、prefillがこの値を超える長文ターンは完了しません。実測の要求全体遅延は200Kで473秒、[画像入力構成](vision.ja.md#200kのテキスト要求)の199,652 token要求で506秒、256Kへの比例見積で約606秒なので、試験設定は700000にしています。`0`はtimerを無効にしますが停止検知を失います。`network.timeout`は補助HTTPクライアントに配線されており、モデルのストリームも縛るかは未確認です。

**限界。** `Bash`の判定は文字列の経験則で漏れがあり、代替は`Bash`を全て確認にすることです。ガードはmatcherに合致するツール（`Write|Edit|Bash`）しか見ないため、将来版が追加する別のファイル書き込みツールはmatcherを広げない限り素通りします。`Edit`／`Write`の保護は存在で決まり内容では決まらないので、承認した上書きは全面上書きのままです。以上は一つのbundleの静的読解と一回のheadless実行であり、受け入れ結果ではありません。

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

この固定GLMテンプレートは常にassistantのthinkingブロックを開始し、オフ指定を読みません。`thinking=false`／`enable_thinking=false`を送らず、思考を有効のまま使います。[公式モデルカード](https://huggingface.co/zai-org/GLM-5.3-Flash)は`reasoning_effort=low/high/max`（既定max）を案内し、チャットには`clear_thinking=true`を推奨しています。lowはチャット試験のprofileであり、maxで行う公式品質評価の再現とは区別します。ZCodeではセッションのeffortを`/effort <level>`（別名`/variant`。`/effort list`が現在値と選択肢を表示）で切り替えます。選択肢はモデルprofileごとで、`glm-5.3`に一致するIDは`low`／`high`／`max`、`medium`と`xhigh`はGPT・Qwen向けprofileにしかありません。値はOpenAI互換providerに`reasoning_effort`として届き、固定GLM chat templateは`low`と`high`だけを受理して、それ以外（`medium`や未指定を含む）は`max`として扱うため、独自の中間値は作れません。運用上の注意：`max`ではこのテンプレートは実質際限なく思考を続け、通常のコーディングターンでも`max_tokens`と時間の大半をthinkingブロックに費やします。日常の作業は`high`以下にし、`max`は意図して一問だけ難問を解かせる時に限ります。今回と一致するparser/template不整合は[vLLM #54744](https://github.com/vllm-project/vllm/issues/54744)で報告され、[修正PR #54825](https://github.com/vllm-project/vllm/pull/54825)は確認時点で未マージでした。任意のイメージに修正済みとは仮定しません。固定したvLLM `385dce36` には、XGrammarと投機デコードの修正[vLLM #53046](https://github.com/vllm-project/vllm/pull/53046)（`c6e19b3`）と[#52805](https://github.com/vllm-project/vllm/pull/52805)（`12f64b3`）が既に入っています（GitHubのcompareで両方に対してbehind 0）。

通常の受け入れは最終回答、構造化ツール要求、実行結果、承認境界で判定します。推論文・token列の完全再現やバッチ間ビット一致は別の数値診断へ分け、不一致の記録を残します。不一致をすべてタスク失敗とみなすことも、最終回答が一致しただけでモデル全体の正しさを証明したとみなすこともしません。ゴールデン検証は同時実行1、並列は性能・品質を分けて評価します。

## 受け入れ試験一覧と実施状態

状態の値はPASS、PARTIAL（一部の条件のみ合格。内訳を記す）、BLOCKED（実行不能。根拠と代替を記録）、NOT RUN。API群を先に行い、共通群Hは両ハーネスで別々に実リクエストと成果物で判定します。文書・ソースの存在や小型fixtureの合格では代用しません。状態は2026-09-15時点で、run IDは非公開`records/`を指します。

**run `20260915-harness-h-sandbox`（2026-09-15）。** 共通群H-01〜H-11の全11ケースを、npm `zcode-app-cli` 3.11.2-24のTUI（`yolo`＋既存ファイルガードhook、通常・liteともloopbackトンネル先の `glm-5.3-flash-nvidia`、`limit.context` 204800、`limit.output` 32000）で、バグを仕込んだ小さな試験repoとunittestを対象に一巡しました。上記の配布形態の規則により、この結果はその配布形態の証拠に限られ、公式Desktop GUIとClaude CodeはHケースを一つも実行していません。PARTIALの欄には未試験の部分を記します。

| ID | 対象 | 操作と合格条件 | 状態 |
|---|---|---|---|
| API-01 | 基礎API | `/v1/models`のserved IDと選択先が一致。短い日本語・英語の要求がローカルモデルから正常応答する | PASS（`api-acceptance-low-local`） |
| API-02 | 基礎API | Chat Completionsの通常応答とSSE。終端・UTF-8・reasoning／最終回答の区別が壊れない。日本語・韓国語の長い出力は[`server mojibake`](validation.ja.md#フルモデルtp2の実験範囲)で別に監視 | PASS（同run。chatは`reasoning_effort=low`） |
| API-03 | 基礎API | 無害なツールの要求→JSON引数検証→結果返送→最終回答。複数往復でID対応を維持する | PASS（同run） |
| API-04 | Anthropic互換API | Messagesの通常応答・SSE・tool_use/tool_result・count_tokensを検査。応答形式・終端・usageを確認する | PARTIAL：Messagesの通常応答とcount_tokensはモデル既定effortで合格。MessagesのSSE、tool_use/tool_result、異常系はNOT RUN |
| ZC-01 | ZCode | Custom Providerへ登録したモデルが選べ、実際のendpointとmodel IDがローカル設定に一致する | 公式Desktop GUIはNOT RUN。Desktop同梱CLIは対話起動がBLOCKED：`@zcode/tui`パッケージ不在で失敗（[zai-org/feedback #270](https://github.com/zai-org/feedback/issues/270)）。headlessの`--prompt`はローカルserved IDへ届く（`20260914-zcode-exists-guard`）が、これはruntimeの証拠でありGUIの証拠ではない。補助証拠としてnpm `zcode-app-cli` 3.11.2-24で、トンネル先のprovider・served IDの選択・日本語の往復・headless promptへのローカルモデル応答を確認 |
| ZC-02 | ZCode | 宣言した入力（テキスト・ツール・画像、動画なし）を尊重。クラウドの既定モデルへ無断で置換されない | ZC-01と同じ：公式はNOT RUN／BLOCKED。npm配布のスモークでは通常・liteの両方がローカルserved IDに固定され、カタログ更新は無効 |
| CC-01 | Claude Code | 専用設定で起動し、通常・補助の要求が指定served IDに届く。モデル名未解決や認証ループがない | NOT RUN |
| CC-02 | Claude Code | Anthropic形式のtool ID、分割JSON、reasoning、stop_reasonを正しく処理し、ツール結果後に会話を継続できる | NOT RUN |
| H-01 | 両方 | 小さな試験repoの2ファイルを読み、実内容に基づく説明を返す。未読内容を読んだと報告しない | npm CLI PASS（`20260915-harness-h-sandbox`）：試験repoの2ファイルを実内容どおりに説明。未読のテストファイルは未読と明示。Desktop／Claude CodeはNOT RUN |
| H-02 | 両方 | 小さなバグを1件修正し、許可したファイルだけに意図したdiffが生じる | npm CLI PASS（同run）：`calc.py` のoff-by-oneを除去。`git diff` はその1ファイルのみ。Desktop／Claude CodeはNOT RUN |
| H-03 | 両方 | 許可したローカルテストを実行し、終了コードと実ログに一致する成否を報告する | npm CLI PASS（同run）：red（テスト五件中二件失敗、exit 1）→ green（五件合格、exit 0）。報告は実ログと一致。Desktop／Claude CodeはNOT RUN |
| H-04 | 両方 | 無害なmarkerファイル作成を一度拒否し、作成されないことを確認。承認を迂回しない | npm CLI PARTIAL（同run）：guardは `.zcode` 配下への `Write` を確認プロンプトに変え、承認の迂回はなし。ただしmarkerはクライアント側の承認後に作成されたため「一度拒否し、作成されない」条件は未達。Desktop／Claude CodeはNOT RUN |
| H-05 | 両方 | 読み取り→編集→検査の複数ツール往復で引数・結果・順序を維持する。追加エージェントなしの初期条件で実施 | npm CLI PASS（同run）：読み取り・テスト・編集・diff・commit・書き込み・削除・設定読み・curl・bgジョブ起動と停止の約15往復で引数・結果・順序を維持。Desktop／Claude CodeはNOT RUN |
| H-06 | 両方 | 生成・ツール待ちを中断してから新規要求を送信できる。無限再試行・残留ジョブ・サーバー停止がない | npm CLI PARTIAL（同run）：bgジョブを停止し残留プロセスなし、停止後もサーバーは200応答。生成中の割り込みと無限再試行の確認は未試験。Desktop／Claude CodeはNOT RUN |
| H-07 | 両方 | クライアント再起動後に試験会話を再開し、同じローカル接続先と承認設定を維持する | npm CLI PASS（同run）：クライアント再起動後に同一セッションIDで再開し、接続先・モデル上限・権限モード・hookが再起動前と一致。Desktop／Claude CodeはNOT RUN |
| H-08 | 両方 | 実サーバーのcontext上限付近を試す。必要な圧縮または明示的エラーで処理し、履歴を黙って失わない。200k／1M対応を仮定しない | npm CLI PARTIAL（同run）：`limit.context` 204800はサーバーの `/v1/models` の `max_model_len` と一致、`limit.output` は32000。上限付近の挙動と圧縮は未試験。Desktop／Claude CodeはNOT RUN |
| H-09 | 両方 | 実要求の接続先を確認し、ローカルendpoint停止時にクラウド推論へfallbackしない。その他の通信も記録し「完全オフライン」と混同しない | npm CLI PARTIAL（同run）：モデル経路はloopbackトンネルでfallback設定なし。停止時fallbackは未試験。同runの付随通信はユーザー依頼の `curl` 2件（`api.fxtwitter.com`、`pbs.twimg.com`）。クライアントのtelemetryは未採取（上記2026-09-14のTCP採取を参照）。Desktop／Claude CodeはNOT RUN |
| H-10 | 両方 | 同じ試験repo・課題で一連の読解、修正、テスト、最終説明を完遂。APIログと成果物、正確性、遅延を保存する | npm CLI PARTIAL（同run）：読解・修正・テスト・報告の課題を完遂し、成果物と正確性はrun logとGit履歴に保存。APIログと遅延計測は未取得。Desktop／Claude CodeはNOT RUN |
| H-11 | 両方 | 対応するreasoning設定がローカルserviceへ届き、非対応のオフ引数が送られず、推論文が最終contentへ漏れない。effortの変換に非対応なら明記する | npm CLI PARTIAL（同run）：オフ引数は設定・送信ともになし、最終回答への推論文の漏れなし。リクエスト中の `reasoning_effort` は未採取。Desktop／Claude CodeはNOT RUN |

H-08の上限は起動TOMLの実サーバー設定を正典とし、クライアント側のcontext認識との整合は未確認です。並列エージェント・MCP・画像は初回合格後の別試験です。

## 記録と合否

非公開`records/<run-id>/`に、case ID、harness名・配布形態・版・hash、設定の非秘密部分、サーバーとモデルの固定値、期待結果、実結果、PASS／PARTIAL／FAIL／BLOCKED／NOT RUN、要求ID・ログ・diff・テスト結果の所在を保存します。ケースの状態が変わったら、同じ変更で上表を更新します。

ZCodeとClaude Codeの結果は分けて集計し、ZCodeの中でも配布形態ごとに分けます。片方の合格で他方を完了扱いにしません。非対応・未実装・契約上の制約が判明した場合も必須ケースを消さず、BLOCKEDと根拠・代替案を記録します。API互換性の不足が確認できた場合だけ、上流修正または小さな変換アダプターを検討し、そのライセンスと追加テストを明記します。
