# GLM起動設定の一括管理

[English](server-configuration.md)

[コメント付きTOML](../examples/server.example.toml)を `state/server.toml` にコピーし、両Linuxノードに同じ内容を置きます。このファイルをランチャーと専用送信コマンドが共通で読みます。

| カテゴリ | 管理するもの |
|---|---|
| `runtime` | 固定イメージID、eager／decode Graph実行、独立EP／PPと層境界、seed、画像入力の切替 |
| `context` | 入出力合計のコンテキスト長、同時シーケンス数、prefillのチャンク予算 |
| `profiling` | 診断用のCUDAカーネル・launch計測。通常の速度測定時は無効 |
| `validation` | CUDA・indexer用、またはexpert実配置用の独立観測worker |
| `cache` | 各ランクのKV容量、要求ブロックサイズ、prefix cache、メモリ使用率、実験用unpack融合 |
| `mtp` | MTP有効化、下書きトークン数、モデルのメタデータview |
| `lpa` | LPA有効化、近似開始層、通常計算を残す末尾、損益分岐の閾値、クエリ省略、projectorとハッシュ |
| `api` | ローカルAPI・ランク間通信ポート、モデル名、パーサー |
| `generation` | 送信コマンドの生成既定値：出力長、temperature、reasoning、タイムアウト |
| `resources` | コンテナ上限、起動前の空き条件、実行中のメモリ余裕、自動停止期限 |
| `nodes` | 両ランクの実測済みfabricアドレス、interface、HCA、GID |

モデルID・revisionとビルドの基底イメージは [runtime.lock.json](../config/runtime.lock.json) が正典です。相対パスはTOML自身の位置が基準です。例外として `mtp.view` はHugging Faceキャッシュからの相対パスで、固定revisionを末尾に自動付加します。秘密鍵やトークンはこのファイルに入れません。

## 配布用の既定設定

配布用TOMLは、[256Kでの画像入力](vision.ja.md)を含む直列の最適化構成を既定にします。これは設定の選定であり、本番・ハーネス検収の完了を意味しません。既存の `state/server.toml` は自動更新されません。

| 項目 | 既定値 |
|---|---|
| 実行 | TP=2、eager、1系列、262,144 token、chunk 2048（[実測](benchmarks.ja.md#200k画像profileでのchunk予算2026-09-17)） |
| 入力 | テキスト・ツール呼び出し・画像（`runtime.vision = true`）。動画は拒否 |
| キャッシュ | FP8、各rank 3 GiB、APC有効、checkpoint保持 `dense`、unpack融合有効、画像前処理キャッシュ0.1 GiB |
| 投機・近似 | MTP k=3。LPA無効（有効時はcut32／tail512／B128、未使用MLA query省略） |
| 検査・並列 | 非同期index検査、EP無効、PP分割なし |
| NCCL | 両rankで `nccl_channels = 8`（NCCLに任せると参照機では64） |
| MoEのtoken順 | `canonical_moe_order = true`：expert内のtoken順を一つに固定し、同一要求の反復を一致させる。この版から作った参照imageが要る（参照機では2026-09-18から配信中） |
| 生成 | temperature=0、max_tokens=4096、reasoning_effort=low、clear_thinking=true |
| 資源 | コンテナ112 GiB、起動前空き108 GiB、実行中余裕3 GiB |
| 実行期限 | `run_seconds=0`：時間による自動停止なし。メモリ監視は継続 |
| 監視 | `stall_seconds=600`：rank 0は要求がrunningのまま `/metrics` の信号が600秒動かなければ停止（`engine-stall`）。`api.dev_endpoints=false` |
| warmup | `warmup=true`、`warmup_long_tokens=0`：readiness後に短文・tool・画像の段を流す。長文段は指定するまで無し |

テキスト専用の代替は `runtime.vision = false` にし、上の長さとKVはそのまま使います。視覚塔を読み込まず、画像前処理キャッシュも持ちません。テキストだけを扱う運用と、メモリの余裕が小さいときの確認用に残しています。その[256K確認](benchmarks.ja.md#256kでの実入力確認)は2026-09-14に保護余裕4 GiB・chunk 512で実施しており、テンプレートの保護3 GiB・chunk 2048は画像なしでは未検証です。

**導入時はimage ID、両機の接続情報、MTP viewを準備してください。LPA projectorとhashはLPAを有効にするときだけ必要です。** 有効な機能のゼロhashは差し替え必須の仮値で、準備不足を理由に機能を黙って無効化しません。[学習済みprojectorの取得](lpa.ja.md#学習済みprojectorの取得)により再学習を省けます。資材の配置は[運用手順](operations.ja.md#資材の保管場所とパス)が正典です。MTP／LPAは個別に無効化でき、基準比較ではAPC・保持・融合・非同期検査も明示的に戻します。

期限は起動時に固定されます。`run_seconds` の変更を稼働中の監視へ反映するには、[両rankの切替手順](launch-safety.ja.md#全レール検査と両rankの切替)で再起動します。設定ファイルの変更だけでは既存の期限は消えません。コンテキストを拡大するときは、以下の容量条件と実要求を別に検証します。

## コマンド

認証クライアント、allocatorの未指定／空文字、全HCA検査、両rankの停止前検査と切替は[起動契約と運用検証](launch-safety.ja.md)を参照してください。P10／P19／P22／E03の追加範囲であり、複数レール実通信などの未検収を機能実装と区別します。

**P22のGPU状態隔離・校正・最終併用・held-out参照評価を確認済みです。** LPAとprefix cachingを両方有効にする場合は `GLM53_APC_LPA_API=1` のimageが必要です。schedulerが全状態を揃えて復元したprefixと `lpa.break_even_tokens` から適用を決めます。テンプレートは[P22の校正](benchmarks.ja.md#apc優先lpaの損益分岐計測p22)に基づく保守的な閾値128を使います。最初の近似以降は、通常計算する末尾・decodeを含めて共有登録を止めます。このモードの `server ask` は判断をサーバーへ任せます。テンプレートは `lpa.enabled = false` を既定とし、バッチ入力に限って用途ごとに有効化します（近似した要求は共有cacheに何も登録しないため）。有効にしている間、要求に `"vllm_xargs": {"glm53_lpa_mode": "off"}` を指定すると通常計算し、通常状態の共有cacheを育てられます。`api.prompt_tokens_details = true`（任意キー、未指定は無効）は `--enable-prompt-tokens-details` を付け、`usage.prompt_tokens_details.cached_tokens` で復元prefix長を返します。無いとvLLMは `null` を返し、cacheが当たっていてもハーネスの表示は0のままです。APCなしの通常の `server ask` はH=0として同じ閾値を使います。[実装契約](apc-lpa-design.ja.md)を参照し、更新するTOMLには新しい閾値キーを明示してください。

`runtime.expert_parallel=false` が既定です。有効にすると両rankへ `--enable-expert-parallel` を追加し、TP=2／DP=1、精度、固定KV予算を維持します。`GLM53_EXPERT_PARALLEL_API=1` を持つイメージが必要ですが、このmarkerは設定対応を表し、EPの検収済み証明ではありません。初期範囲はeager・1／2系列・MTP/LPA/fusion/APCなしです。既存TOMLにも新しいキーを明示し、欠落時の暗黙fallbackは設けません。使用前に[EPの独立評価手順](performance-investigation.ja.md#expert-parallelp21)を参照してください。

`runtime.vision` は任意キーで、未指定はfalse、テンプレートは `true` です。`false` は `--language-model-only` を残し、視覚塔を読み込まずテキスト・ツール専用で動かします。`true` は両rankからこのフラグを外し、`--limit-mm-per-prompt '{"video": 0}'` を付けます。**`vision = true` でも動画入力は無効で、送ると拒否されます。受け付けるのは画像だけです。** 理由は起動時のメモリです。vLLMは最大の入力1件を一度エンコードしてメモリを見積もり、このチェックポイントの動画の上限（30,000 token＝120,000パッチに制限済み）は画像1枚（最大8,000 token）よりはるかに大きいためです。1 promptあたりの画像枚数はvLLMの既定のままです（チャットハーネスは過去の画像を毎ターン送り直すため）。Vision有効時は `cache.mm_processor_cache_gb`（任意キー、未指定は0.1）が `--mm-processor-cache-gb` を決めます。これは前処理済み画像のキャッシュで、vLLMはAPIプロセスとエンジンプロセスの両方に持ち、どちらもrank 0にだけ載ります。vLLMの既定4 GiBのままだと、headのホストRAMを最大8 GiB使い得ます。上限より大きい画像はキャッシュせずに処理し（警告のみ）、拒否はしません。視覚塔はBF16で量子化の対象外です（両方の除外リストに `model.visual*` があり、固定vLLMも量子化設定なしで組み立てる）。実測、headのメモリ余裕、未解決の事項は[200Kでの画像入力](vision.ja.md)にあります。検証fixtureはこのキーに関係なくテキスト専用で読み込みます。

`api.dev_endpoints`（任意キー、未指定はfalse）は両rankに `VLLM_SERVER_DEV_MODE=1` を渡し、loopbackのAPIにvLLMのdev経路（`/reset_prefix_cache`・`/reset_mm_cache`・`/collective_rpc`・`/sleep`・`/wake_up`・`/server_info`）を載せます。LPA・component・expertのprofileは元からこのモードで動きます。このキーは、LPA offの配布profileでもベンチやwarmupの後始末のために再起動なしでprefix cacheを消せるようにするためのものです。これらの経路は無認証です（[起動契約](launch-safety.ja.md#モデルapiクライアント)）。他のローカル利用者がいる機体では off のままにします。

`validation.memory_probe`（任意、未指定はfalse）はworker拡張を一つ載せ、dev経路 `POST /collective_rpc`（`{"method": "allocator_stats"}`）から呼べるようにします。`allocator_stats` はrankごとにtorchのcaching allocatorのreserved・allocated・active・inactive-splitバイト、segment数、確保のretry回数、`mem_get_info` を返します。GB10ではGPUがホストのメモリを共有するので、allocatorの成長はworkerのRSSやコンテナのcgroupが平らなまま `MemAvailable` の減少として現れます。この探針はその二つを切り分けます。ほかのメソッドは、稼働中のworkerの中でしか見えないものを読みます。`host_stats` はworkerの常駐匿名メモリ、glibcの `mallinfo2` の内訳（使用中、保持している空き、mmapしたblock）、torchのpinned host cacheを返し、`{"trim": true}` を付けると二回の読みの間で `malloc_trim(0)` を呼びます。保持している空き・生きているオブジェクト・mallocの外のメモリを切り分けられます。`host_census` は生きているPythonオブジェクトを型ごとに数え、その中のCPU tensorの量を出します。`fa2_stage` はFA2経路の一部だけ（`off`、圧縮の単一操作、`compact`、`plan`、`full`）を走らせて残りを参照経路で答えるので、増加が始まる段を、段ごとの再起動なしに名指しできます。`trace_begin` と `trace_end` は、traceするmodule・draftモデル・sparse NoPE attentionの入出力に1要求ぶんの指紋（呼び出しごとに、byte列をint64の語として読んだwrapする和を二つ、device上に保持。byteをint64へcastして足すとtensorの8倍を確保します）を取り、最初の要求を基準にして、後の同一要求で実行順に最初に食い違った呼び出しを名指しします。`sync` は traceした呼び出しごとに同期します。2次元の整数tensor（候補index）は行ごとに並べ替えてから指紋を取ります。top-kのkernelは同じ候補を呼び出しごとに違う並びで返し、並びはattentionの前で正規化されるので、候補の集合が違う時だけ食い違いとして読めます。基準のrunは、最大の呼び出しの行数とgatherした最大の行数を返します。attentionが触ったcache行のgatherはdecodeの大きさの呼び出しでだけ行います。prefillのchunkではそのgatherがMLA層ごとにGB級になり、GPUがホストと共有するメモリを使うためです。新しい読みを足す前に、本番の形で大きさを測ってください。4層fixtureの形は小さく、この費用は見えません。workerのメソッドが例外を出すとHTTP 500が返り、その次の `/collective_rpc` の呼び出しも、内容に関係なくHTTP 500になります。その後の呼び出しは正常で、各rankが自分の行を返します（基準の2台で2026-09-20に計測：不正な `fa2_stage` を1回、続けて `allocator_stats` を3回）。エラーの直後の1回は捨ててください。LPAや他のworker拡張とは排他（起動につき拡張クラスは一つ）で、起動の他の点は変えません。

`resources.stall_seconds`（任意、未指定は0＝無効、テンプレートは600）と `generation.warmup`／`generation.warmup_long_tokens`（任意、未指定はfalse／0）は[監視・停滞検知・warmup](operations.ja.md#監視停滞検知warmup)で説明します。いずれもprofileのfingerprintを変えるため、次の切替から有効になります。

`runtime.nccl_channels`（任意、未指定はNCCLに任せる、テンプレートは8）は、両rankの `NCCL_MIN_NCHANNELS` と `NCCL_MAX_NCHANNELS` に同じ正の整数を渡します。参照機ではNCCL 2.30.7に任せると64本になります。各rankのエンジンはcommunicatorを2本開き、MTU 1500で8本にすると、実モデルの最小空きメモリが64本に比べてheadで2.8 GiB、peerで3.0 GiB増え、prefillは遅くなりませんでした（1%速い）。decodeは設定の差より計測ごとのぶれの方が大きい値でした。数値は[チャネル数の測定](nccl-validation.ja.md#チャネル数)にあります。1.3.1より前に書いたprofileにはキーが無く、NCCLの選択とfingerprintをそのまま保ちます。テンプレートの値を使うにはキーを足します。Mia PR #200を参考にしました。値を変えるとprofileのfingerprintが変わるため、次の切替から有効になります。

`runtime.canonical_moe_order`（任意。テンプレートは `true`。未指定はimageの既定に従い、この版から作ったimageでは有効）は、両rankに `GLM53_CANONICAL_MOE_ORDER` を渡します。固定版vLLMの `moe_align_block_size` はexpert内のtokenをCUDAスレッドのスケジューリング順に並べ、MarlinのMoEの結果はその順序にわずかに依存し、後段のrouterがそれを増幅するため、同一要求の反復が一致しませんでした（[測定](validation.ja.md#フルモデルtp2の実験範囲)、上流はvLLM issue #52525）。`true` にすると、参照imageがkernelの前に各expertのスロットをtoken id順に並べます。このとき `server preflight` はimageに `GLM53_MOE_ORDER_API` を要求するので、patchの無いimageには指定できません。この版以降のimageは `2` を持ちます。`1` も通します。切替は、動いている対を復旧先として新しいcheckoutで検査するので、要求を後から厳しくすると、古いimageからの切替がすべて拒否されるためです。`false` は比較用のarmで、imageの対応は要りません。expert parallelには手を入れません。8層fixtureでは全反復がbit一致になり、prefillは変わらず、decodeは約1%遅くなりました。参照機では同一要求がbit一致で反復し、decodeは遅くならず、MTPの採択長は上がりました（[検証](validation.ja.md#フルモデルtp2の実験範囲)）。既定で有効にしているのは、今後graphや再量子化のA/Bを読む物差しとして、再現できる基準が要るためです。

`runtime.stable_indexer_topk`（任意。テンプレートは `true`。未指定はimageの既定で、この版以降のimageではon）は、両rankに `GLM53_STABLE_INDEXER_TOPK` を設定します。kpool indexerは4 tokenを1 poolに畳み、query行ごとに512 poolを選びます。固定しているvLLMの `persistent_topk`（decode）と `top_k_per_row_prefill` は、正しいpoolを呼び出しごとに違う並びで返し、512位の境界にpoolの同点があると、同じ入力から違う*集合*を返します。並びはattentionの前で既に正規化していましたが、集合はそうではなく、ある1 stepの同点一つでcompletionが割れていました。場所は決定的で、発生は時々です。GB10では、decodeのkernel単体で同一の呼び出し1,200回に3通りの集合、prefillのkernelで18,000行に4通りが出ました。4層のMTP fixtureでは同一要求12本のうち5本が同じindexerの呼び出しで最初に食い違い、選択を決定的にすると12本とも食い違いませんでした。キーが `true` の時、同点は低いpool indexに決まります。decodeは安定なsort（最大6行、1回0.07〜0.25 ms。kernelは0.01〜0.02 ms。同期なし）、prefillはkernelのまま、512位の値を収まりきらない数のpoolが共有している行だけを選び直します（呼び出しごとに同期1回）。`server preflight` はimageに `GLM53_INDEXER_TOPK_API=1` を要求します。`false` は比較用のarmで、imageの対応は要りません。投機の深さは割れの原因ではなく、completionがどの位置を通るかを決めるだけです。

`runtime.derived_checkpoint`（任意のtable、既定では無し。持たないprofileのfingerprintは変わらない。table内の `enabled = false` はtableを残したまま固定snapshotを配信する）は、固定checkpointを手元で再量子化した複製をA/B用に配信します。`path`（両hostの絶対ディレクトリ。`/derived` に読み取り専用でmount）、`requant_target`（そのディレクトリの `quantization_config.producer.requant_target` と照合）、`overlays`（`{target, source, sha256, base_sha256, marker}` の列。`source` は絶対パスのファイルで、image内のGLMモデルディレクトリの `target` に重ねてmount）を取ります。`server preflight` は、checkpointが `MIXED_PRECISION` とそのtargetを宣言していること、MTP draft層に量子化宣言が無いこと、各overlayが指定のSHA-256でmarkerを含むこと、image内の `target` が `base_sha256` であることを確かめ、どれか一つでも違えば失敗します。別のimage向けに作ったoverlayはmountできません。derived checkpointではMTPのメタデータviewを使いません。`MIXED_PRECISION` では宣言の無いmodule（BF16のdraft層を含む）が無量子化で読まれるためです。テンプレートにこのtableはありません。参照機では[施策台帳](optimization-catalog.ja.md)のP23の比較で一度使いました。

`runtime.pipeline_parallel_size=1` はTP=2を維持し、2にすると同じ2台でTP=1／PP=2を選びます。`GLM53_PIPELINE_API=1` を持つイメージが必要です。`pipeline_split_layer` は前段stageの層数で、既定候補24なら24／21層に分け、この固定モデルでは各stageに21 MoE層ずつを置けます。容量を保証する値ではありません。両stageにMLAが必要なため、境界の許容範囲は4〜43です。初期範囲は1系列・eager・EP/MTP/LPA/fusion/APCなし。[小層の検証結果](component-validation.ja.md#8層ppの観測)は、全モデル速度・長文の数値同値・本番運用の認定ではありません。TOML更新時は両キーを明示してください。

`validation.expert_worker=true` は、実際のexpert配置・kernel・parameter情報を返す型付きRPC `expert_info` を有効にします。独立したeager TP2の基準／EP条件、最大2系列が対象で、他の観測worker・MTP/LPA/APC・PPとは併用しません。層のhash観測は明示的な `pipeline_observe` RPCで初めて開始するため、性能測定中はそのhookを入れません。実験用のローカル制御経路であり、業務利用の認定ではありません。

`runtime.index_checks` は `auto`／`sync`／`async`（配布既定） を選びます。autoはeagerで同期検査、Graphで非同期検査を使い、従来の動作を維持します。asyncを明示すると、独立評価したeagerの非同期検査を選べます（`GLM53_ASYNC_INDEX_CHECK_API=1` が必要）。範囲検査は常に実施します。asyncで不正indexを検出するとCUDA contextが使えなくなる場合があるため、両rankを再起動します。Graphではsyncを拒否します。MTP/LPA/fusionの直列併用はP18／P22で範囲を限定して確認済みです。

コンテキスト長や同時数を変更する前に、[KV容量とRAMの条件](#kv容量とramの条件)も確認してください。

リポジトリ直下で生成される起動条件を確認します。これはWindowsでも実行できます。

```sh
python -m glm53_setup server plan --rank 0
python -m glm53_setup server plan --rank 1
```

各Linuxノードの `server preflight --rank N` でモデル・fabric・イメージID・他のコンテナがGPUを使っていないこと・空きメモリを確認できます（[各検査の範囲](operations.ja.md#フルモデルの起動検査)）。worker側でrank 1を先に、head側でrank 0を後に、それぞれの端末で起動します。

```sh
python -m glm53_setup server start --rank 1
python -m glm53_setup server start --rank 0
```

起動コマンドは前面に残り、自分のコンテナを監視します。端末を維持してください。`resources.run_seconds` の期限には**モデルのロード時間も含まれます**。長時間使う場合は起動前に延ばします。Ctrl+C・期限到達・空きメモリ不足でそのランクを停止します。分散実行に異常が出た場合は両ランクを停止します。コンテナと `records/` のログ・起動設定は残し、自動削除や自動再起動はしません。

APIが準備できたら、headの別端末から送信できます。

```sh
python -m glm53_setup server ask --prompt "GLM-OK とだけ返してください。"
python -m glm53_setup server ask --request request.json
python -m glm53_setup server status --rank 0
python -m glm53_setup server capacity
python -m glm53_setup server warmup
python -m glm53_setup server mojibake
python -m glm53_setup server agreement
python -m glm53_setup server stop --rank 0
```

`agreement` は自作の4文（日本語・英語・コード・数理）を request lock の下で `/v1/completions` に `prompt_logprobs` 付きで2回ずつ送り、実際の次tokenの順位とlog確率を `records/<stamp>-agreement-r0/result.json` に書きます。`--reference <result.json>` を付けると、その過去の実行に対するargmax一致・top-5の重なり・log確率の移動を加えます。配信中の全モデルは、同じ要求を繰り返しても完全には同じ結果になりません。参照機では同じ文を2回流したときのargmax一致が約96%で、log確率が8 nat動いた位置もありました（4層fixtureの反復はbit一致です）。fixtureでは、出どころは呼ぶたびに変わるexpert内のtokenの並び順でした（[検証](validation.ja.md#フルモデルtp2の実験範囲)）。この版から作ったimageでは `runtime.canonical_moe_order` がこれを固定します。そのため記録は、全要求が検査した形で返れば合格とし、この差は物差しとして `self_agreement` に残します。文ごとの平均負log確率は実行間の動きがずっと小さく（0.003〜0.05）、こちらが安定した読みです。

`capacity` は稼働中headの起動ログと `/metrics` を読み、KV poolをそのまま表示します。stockの `GPU KV cache size` 行を `num_gpu_blocks`・最大長要求1本あたりのblock数・group別block幅に分解し、stockの値が `max_concurrency × max_model_len` であることを添えます。会話の保持本数の推定（16K・64K・`max_model_len` でのblock数と本数。dense保持・block整列hit・稼働なしの前提）は、LPA worker extensionを載せたprofileでだけ表示します。`apc_cache_layout` RPCが各groupのspec種別を返すためで、それ以外、または未対応の種別があるときは推測せず withheld と表示します。`warmup` は要求ロックの下でladderを流し、`records/<stamp>-warmup-r0/result.json` に記録します。段が失敗すると非ゼロで終了します。`mojibake` は同じロックの下で稼働中のheadに日本語と韓国語の長い回答を求め、回答とreasoningの化け文字を数えて `records/<stamp>-mojibake-r0/result.json` に記録し、全回答が合格でなければ非ゼロで終了します（[検査の内容](validation.ja.md#フルモデルtp2の実験範囲)）。

workerの状態確認・停止はworker上で `--rank 1` を使います。全コマンドで `--config 設定ファイル.toml` を指定できます。設定を編集したら両ランクを停止・再起動してください。送信時には起動中の設定との一致を検査します。`generation` は専用送信コマンドの既定値で、他のAPIクライアントの生成設定はそのクライアント側で指定します。

## 自動停止と連続稼働

`resources.run_seconds` はロード時間込みの自動停止期限（秒）です。**`0` で時間制限なし**、正の整数で指定秒数後に停止します。負数は受け付けません。どちらの場合も `resources.reserve_gib` によるメモリ保護は有効です。設定は起動時に読み込むため、ファイル編集だけでは起動中の監視プロセスの期限は変わりません。

```toml
[resources]
# 既存のresources節にある、この値を変更します。
run_seconds = 0
```

これは予定時刻で停止しない設定であり、24時間365日の可用性を保証するものではありません。OS起動時の自動起動・障害時の両ランク協調再起動・冗長構成への切り替えは未実装です。前面の監視プロセスを維持する必要があり、本番・ハーネス検収も未完了です。

## KV容量とRAMの条件

**最大長の要求をB本同時に保持するなら、入出力合計の上限Cに対してB×C token分を収容できる容量の確認が必要です。** `max_model_len`は入力と生成の合計上限、`max_num_seqs`は同時実行の上限です。この二つを設定するだけで、最大長×同時数のKVが確保・検収されるわけではありません。受け入れた2系列の範囲は1要求2,112 tokenまでです。Spark 2台の他レシピでは、25〜100Kの要求2本の同時処理が合計約4 tok/sまで落ちたと報告されています（tonyd2wild #14、コードは採用しない）。

起動行 `GPU KV cache size: N tokens, Maximum concurrency for L tokens per request: Cx` は、このhybridモデル（MLA・IndexPool tail・KDA state群・MTP draftが一つのblock poolを共有し、整列した区間ごとにgroup別のidを使う）では `N = C × L` です。`N` は同時実行数をtoken単位で表した値で、prefix cacheが保持できる会話tokenの数ではありません。`server capacity` が分解を表示し、group種別が分かる場合は会話本数の推定も出します。測った画像入力構成の二つでは、KV 1 GiBに4,608 tokenのblockが28個入り、長さLの要求1本はそのうちceil(L / 4608) + 16個を使いました。204,800 tokenと2.5 GiBでは70個のうち61個、262,144と3 GiBでは84個のうち73個で、どちらも1.15倍です。256Kの値は切替前にこの数え方で見積もったものです。他の長さ・KV量・group構成では、それぞれの起動行を確かめてください。prefix cacheもblock単位で働くため、同じN tokenのpromptを繰り返したとき復元されるのは `(floor(N / block) - 1) x block` tokenで、2 block未満では一切復元されません。画像profileのscheduler block 4,608 tokenでの実測は、3,625 tokenで0、14,025 tokenで9,216、28,025 tokenで23,040でした。短い会話はこのprofileでは再利用の恩恵を受けません。

このランチャーの`cache.kv_cache_memory_bytes`は、**各rankで要求間共有する固定KV poolのバイト予算**です。1 GiBを指定したまま同時数を1→2にしても、各rankのKV予算は1 GiBのままです。要求1本あたり1 GiBでも、2台の予算を自由に合算した一つのpoolでもありません。バイト指定時は`gpu_memory_utilization`によるKV容量の自動推定を使わないため、この比率をRAM全体の保護上限として扱いません。[vLLMの設定仕様](https://docs.vllm.ai/en/latest/configuration/engine_args/#kv-cache-memory-bytes)

実行中に必要なcacheは、保持中の各要求の入力＋生成済みtokenに従ってpool内のblockを消費します。最大長を同時に保証したい場合は、出力予算も含む最大条件で検証します。GLMは疎MLA・IndexPool・系列ごとのKDA状態を併用するため、一般的なdense attentionの単純なbytes/token式をそのまま使わず、**固定runtimeのcache spec・block整列・各groupの容量と状態slot数**で見積もります。MTP等の追加状態も別途含めます。

各ノードで、重み＋KV/cache状態＋activation・indexer等の一時領域＋MTP/Graph等の追加領域＋CPU/OS・他負荷＋運用余裕が、利用可能な統合RAMに収まる必要があります。KV poolを固定しても、context・chunk・同時数に依存する別の割当が増えることはあります。コンテナ上限と`reserve_gib`は保護手段であり、容量適合や無停止の保証ではありません。以前の 256K テキスト専用 profile は実測 121 GiB のホストでこの保護に接していました。available は 4.5 GiB 前後で推移し、保護余裕 4 では監視停止（`stop-reason: memory-reserve`）が 2 回発生し（2 回目は 16,859 token の近似要求の最中）、その後 3 へ下げました。画像入力構成の配信中の head の空きは、NCCL の 64 チャネルでは約 4.0〜4.2 GiB（[実測](vision.ja.md#メモリ最終構成)）、8 チャネルでは [1.3.1 の 200K 実入力](benchmarks.ja.md#131での200k実入力)で 6.97 GiB 以上、chunk 2048 で 6.40 GiB 以上でした。256K・KV 3 GiB では 5.82 GiB 以上で（[1.5.0 での測定](benchmarks.ja.md#150での測定)）、テンプレートは 3 です。値は小数でも指定できます。変える前に、この保護が捉えられる範囲を見積もってください。監視は `MemAvailable` を 2 秒ごとに読み、実測では割り込んだ標本からコンテナ終了まで約 9 秒かかりました。実行中のコンパイルで測った 0.15 GiB/秒の下降なら、reserve から約 1.6 GiB 下まで沈み得ます。参照ホストではカーネル・コンテナとも OOM kill の記録はなく、ホスト側のページは 16 GiB の swap が受けるため、reserve を割った先の危険はワーカー内の GPU 確保の失敗です。その場合は監視停止ではなく、エンジンが突然終了します。ドライバの `NV_ERR_NO_MEMORY` カーネルメッセージは空き 4 GiB 以上、主にロード中に出るもので、底の目印にはなりません。

KVが不足すれば起動が拒否される場合があり、実行時は待ちやpreemption・再計算により性能が落ちることがあります。固定KV poolが勝手に必要量まで拡張されるわけではありません。KV以外の割当やRAM予算が不足すればOOMやガード停止も起こり得ます。[vLLMのpreemption説明](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption)

起動時のcache容量・最大並列度は計算上の目安として保存し、**意図する入出力長×同時数の実要求、preemption回数、両rankの空きメモリ最小値、OOM・ガード停止**を確認してから対応範囲を表明します。現行preflightの合格は、最大長×同時数の収容試験の代わりにはなりません。実測範囲は[標準batchingの独立評価](benchmarks.ja.md#標準batchingの独立評価)を参照してください。

## 現行イメージの契約

現行ソースから再ビルドし、`GLM53_LPA_API=2`、worker `glm53_setup.runtime.lpa.LPAWorkerExtension`、RPC `lpa_configure`・`lpa_report`、読み取り専用マウント `/lpa/projector.pt` を使います。preflightはmarkerなしのイメージを拒否します。両ノードの固定イメージIDを確認し、起動設定を更新してから起動します。

旧コマンド・旧設定名・旧イメージへのフォールバックはありません。新ソースをホストへ置くだけでは稼働中のイメージは変わらないため、再ビルドと実機検証を行います。

## LPAとMTP・制約

`runtime.decode_graphs`（任意、テンプレートは `false`、未指定はeager）がdecode Graphの唯一のスイッチです。`true` で `CompilationMode.NONE`・`FULL_DECODE_ONLY` を渡します。capture size は一つで、MTP有効時は `num_speculative_tokens + 1`（固定ランタイムはdecodeのsizeをこの倍数に切り上げ、`[1]` は拒否します）、無効時は `1` です。prefillはcompileせず、イメージには `GLM53_DECODE_GRAPH_API=1` が必要です。同時1シーケンスならMTP・prefix cacheと併用できます。expertのtoken順を固定した4層MTP fixtureで、eagerとgraphは全長さでtokenもlogprobも一致しました（[部品検証](component-validation.ja.md#decode-graphのfixture独立評価)）。LPAは引き続きeagerが必要で、複数系列のGraph設定はbatchingで同じ一致がfixtureで示されるまで起動設定で拒否します。これは全モデルの受入完了を意味せず、全モデルA/Bを読むまで既定はeagerのままです。以前の書き方 `runtime.enforce_eager`（`false`＝Graph）も読むので既存profileのfingerprintは変わりませんが、両方を書く場合は矛盾させないでください。

Graph経路では内部候補indexの範囲検査をGPU上で非同期に行います。不正indexを黙って許容せずdevice assertにしますが、通常のPython例外と異なりCUDA contextが使用不能になり得るため、障害時は両rankを停止して再初期化します。Graph用のメモリ保持・起動時capture時間も比較対象です。[vLLM #53366](https://github.com/vllm-project/vllm/issues/53366) は、compile cacheのhashに投機token数が入っていないと報告しています。compileするGraphをMTPと併せて検収する場合は、kごとにcacheを分けるか、kを変えたときに消してください。上のopt-inは何もcompileせず、MTPとの併用も拒否します。eagerの検査方式も `runtime.index_checks` に従い、配布既定はasyncです。

`validation.component_worker=true` は部品検証専用workerを選び、`GLM53_COMPONENT_API=1` を持つイメージを要求します。eager・同時1シーケンス・LPA/MTP/prefix cacheなしの独立構成です。型を制限したRPCでindexerの候補・時間を採取し、排他的なリクエスト間でunpack融合を切り替えてA/B/Aを検証できます。Reuse/Reindexを本番適用する設定ではありません。制御クライアントは一つに限定します。

`cache.fused_unpack=true` が配布既定です。falseならTorchの参照変換を使い、trueなら656バイトのMLAキャッシュからのFP8変換とFP32スケール乗算を一つのTritonカーネルで処理します。イメージに `GLM53_FUSED_UNPACK_SUPPORTED=1` が必要で、preflightで確認します。部品一致と全モデルA/B／直列併用の実測範囲で使用します。広範な品質・本番検収とは区別します。候補集合の変更や層間のKV共有は行いません。

CUDA融合の実モデルA/Bとindexerの採取結果・検証限界は [部品検証記録](component-validation.ja.md) を参照してください。

`mtp.enabled` と `lpa.enabled` を個別に切り替えます。MTP有効時は [prepare_mtp_view.py](../tools/prepare_mtp_view.py) で作成したviewとBF16 Triton下書きバックエンドを使います。`mtp.num_speculative_tokens` は1〜5を受け、測定済みは1と3（[投機デコード](speculative-decoding.ja.md)）、2・4・5は同じ1層draftで深さを掃引するために起動できます。LPA有効時は `runtime.lpa_image` を選び、projectorを読み取り専用でマウントしてworker拡張を有効にします。併用にはMTP対応を明示したLPA workerを含むイメージが必要で、旧LPAイメージは併用指定を拒否します。

LPAはリクエストごとの入力長が必要です。専用クライアントが実際のテンプレートでトークン数を求め、worker設定→生成→トークン数の一致確認→LPA解除まで行います。入力全体が `lpa.tail` に収まる短文は通常計算です。制御するクライアントは一つに限定してください。専用CLI同士はhead上のロックで直列化しますが、直接APIを呼ぶ他クライアントまでは調停しません。専用送信コマンドは非ストリーミングのテキスト・ツール会話用です。一般ハーネスや本番運用の認定は別です。

範囲はTP=2・テキスト／ツール・Marlin W4A16・FP8 KVです。LPAには同時1シーケンス・eager実行が必要で、prefix cacheは上記のP22経路を使います。LPAなしのthroughput用構成では同時数を増やせますが、[性能調査計画](performance-investigation.ja.md)に従ってタスク品質・状態整合・資源を別途検証します。MTPの下書き数は1と3です。コンテキスト長・チャンク・キャッシュ量・cut・tailを変えた場合は再測定が必要で、設定検査の合格は品質や必要メモリの保証ではありません。[LPA](lpa.ja.md) と [MTP](speculative-decoding.ja.md) に検証範囲を記載しています。
