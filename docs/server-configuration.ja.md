# GLM起動設定の一括管理

[English](server-configuration.md)

[コメント付きTOML](../examples/server.example.toml)を `state/server.toml` にコピーし、両Linuxノードに同じ内容を置きます。このファイルをランチャーと専用送信コマンドが共通で読みます。公開した任意設定には専用の例 [server.axl.example.toml](../examples/server.axl.example.toml) があります：同じprofileに、再パックした重み（NVFP4 BIZ AXL）と同梱のoverlayの `runtime.derived_checkpoint` 表、`runtime.prefix_page_dedup`、同時2系列、rankあたり6 GiBのKVを足したものです。ランチャーは再パックしたcheckpointなしで3 GiBを超えるKVを拒みます：参照対では固定の重みがKV 3 GiBでheadに5.5 GiBを残し（保護は3 GiB）、再パックした重みは10.5 GiBを残します。

| カテゴリ | 管理するもの |
|---|---|
| `runtime` | 固定イメージID、eager／decode Graph実行、独立EP／PPと層境界、seed、画像入力の切替、再現性のスイッチ、derived checkpoint |
| `context` | 入出力合計のコンテキスト長、同時シーケンス数、prefillのチャンク予算 |
| `profiling` | 診断用のTorch/CUDA traceの採取（必要なときだけ）。速度測定ではoff |
| `validation` | CUDA・indexer用、またはexpert実配置用の独立観測worker、メモリ探針 |
| `cache` | 各ランクのKV容量、要求ブロックサイズ、prefix cache、checkpoint保持、メモリ使用率、unpack融合 |
| `mtp` | MTP有効化、下書きトークン数、モデルのメタデータview |
| `lpa` | LPA有効化、近似開始層、通常計算を残す末尾、損益分岐の閾値、クエリ省略、projectorとハッシュ |
| `api` | ローカルAPI・ランク間通信ポート、モデル名、パーサー、dev経路、cache済みtokenの報告 |
| `generation` | 送信コマンドの生成既定値：出力長、temperature、reasoning、タイムアウト。warmupの段 |
| `resources` | コンテナ上限、起動前の空き条件、実行中のメモリ余裕、自動停止期限、停滞検知 |
| `nodes` | 両ランクの実測済みfabricアドレス、interface、HCA、GID。ホスト別Docker CPU setは任意指定 |

モデルID・revisionとビルドの基底イメージは [runtime.lock.json](../config/runtime.lock.json) が正典です。相対パスはTOML自身の位置が基準です。例外として `mtp.view` はHugging Faceキャッシュからの相対パスで、固定revisionを末尾に自動付加します。秘密鍵やトークンはこのファイルに入れません。

## 配布用の既定設定

配布用TOMLは、[256Kでの画像入力](vision.ja.md)を含む直列の最適化構成を既定にします。これは設定の選定であり、通常運用の受け入れは[SETUP手順6](../SETUP.ja.md#6-フルモデルの検証)に別に記録しています。既存の `state/server.toml` は自動更新されません。

| 項目 | 既定値 |
|---|---|
| 実行 | TP=2、eager、1系列、262,144 token、chunk 2048（[実測](benchmarks.ja.md#200k画像profileでのchunk予算2026-09-17)） |
| 入力 | テキスト・ツール呼び出し・画像（`runtime.vision = true`）。動画は拒否 |
| キャッシュ | FP8、各rank 3 GiB、APC有効、checkpoint保持 `dense`、unpack融合有効、画像前処理キャッシュ0.1 GiB |
| prefillのattention | `fa2_attention = true`：query行が6を超えるNoPE attentionの呼び出し（prefillと、2系列が共有するdecodeのstep）をFlashInfer FA2へ、1系列のdecodeは参照経路。LPAとは排他 |
| 投機・近似 | MTP k=3（[深さ1〜5](speculative-decoding.ja.md#深さ152026-09-1920)）。LPA無効（有効時はcut32／tail512／B128、未使用MLA query省略） |
| 検査・並列 | 非同期index検査、EP無効、PP分割なし |
| NCCL | 両rankで `nccl_channels = 8`（NCCLに任せると参照機では64） |
| 再現性 | `canonical_moe_order`・`stable_indexer_topk`・`inductor_deterministic` をすべて `true`：同一要求はbit一致で反復し、どの起動も同じ数値状態で計算する（[再現性のスイッチ](#再現性のスイッチ)） |
| 生成 | temperature=0、max_tokens=4096、reasoning_effort=low、clear_thinking=true |
| 資源 | コンテナ112 GiB、起動前空き108 GiB、実行中余裕3 GiB |
| 実行期限 | `run_seconds=0`：時間による自動停止なし。メモリ監視は継続 |
| 監視 | `stall_seconds=600`：rank 0は要求がrunningのまま `/metrics` の信号が600秒動かなければ停止（`engine-stall`）。`api.dev_endpoints=false` |
| warmup | `warmup=true`、`warmup_long_tokens=0`：readiness後に短文・tool・画像の段を流す。長文段は指定するまで無し |

テキスト専用の代替は `runtime.vision = false` にし、上の長さとKVはそのまま使います。視覚塔を読み込まず、画像前処理キャッシュも持ちません。テキストだけを扱う運用と、メモリの余裕が小さいときの確認用に残しています。その[256K確認](benchmarks.ja.md#256kでの実入力確認)は2026-09-14に保護余裕4 GiB・chunk 512で実施しており、テンプレートの保護3 GiB・chunk 2048は画像なしでは未検証です。

**導入時はimage ID、両機の接続情報、MTP viewを準備してください。LPA projectorとhashはLPAを有効にするときだけ必要です。** 有効な機能のゼロhashは差し替え必須の仮値で、準備不足を理由に機能を黙って無効化しません。[学習済みprojectorの取得](lpa.ja.md#学習済みprojectorの取得)により再学習を省けます。資材の配置は[運用手順](operations.ja.md#資材の保管場所とパス)が正典です。MTP／LPAは個別に無効化でき、基準比較ではAPC・保持・融合・非同期検査も明示的に戻します。

期限は起動時に固定されます。`run_seconds` の変更を稼働中の監視へ反映するには、[両rankの切替手順](launch-safety.ja.md#全レール検査と両rankの切替)で再起動します。設定ファイルの変更だけでは既存の期限は消えません。コンテキストを拡大するときは、容量条件と実要求を別に検証します（[KV容量](#kv容量とramの条件)）。

### CPU配置の任意指定

CPU配置を固定する場合は、各rankの`nodes[].cpuset_cpus`にDockerのCPUリストを指定します（例：`"5-9,15-19"`）。
省略時はDockerの従来の配置を使います。
高性能コアの番号はホストごとの構成と実測から決めてください。
コア番号を別のホストへそのまま転用することはできません。

ランチャーは形式の不正な指定、逆順や重複のある範囲を拒否します。
各ホストの`server preflight`は、指定されたCPUが起動プロセスの利用可能な範囲に含まれるか確認します。
起動後はDockerの`HostConfig.CpusetCpus`を読み戻し、設定と一致しなければ新しいコンテナを停止します。
この設定は配置を制御するもので、速度を保証するものではありません。
両rankに設定してください。
参照対（両ホストとも高性能コアは5〜9と15〜19）では、どちらか一方のrankが高効率コアにいるだけでdecodeが約3分の1になり、固定なしではスケジューラがたまたま両rankを高性能コアに置いていました。
参照対は1.18.0への切替から、両rankに `5-9,15-19` を指定して配信しており、そこでpreflightの確認と読み戻しが通りました（[1.15.0での測定](benchmarks.ja.md#1150での測定)）。

## 公開した任意設定と配布既定の差

二つの例は、一つのprofileの5設定を変えたものです。[`tests/test_axl_example.py`](../tests/test_axl_example.py)が両方のdocker commandと環境変数をrankごとに同じimageを与えて組み立て、差を次の行とその帰結2点に固定します。

| AXLの例の設定 | 起動に加わるもの |
|---|---|
| `runtime.derived_checkpoint`（`path`、`requant_target = "l"`） | mount 1本：再パックしたcheckpointを `/derived` に読み取り専用で |
| `runtime.derived_checkpoint.overlays[0]`（`kda-quant-split.py`） | mount 1本：imageの `kda.py` の上に読み取り専用で。hashと元のhashはpreflightが検査 |
| `runtime.derived_checkpoint.overlays[1]`（`mla-quant-split.py`） | mount 1本：imageの `model.py` の上に読み取り専用で。同じ検査 |
| `runtime.prefix_page_dedup = true` | 環境変数 1つ：`GLM53_PREFIX_PAGE_DEDUP=1` |
| `context.max_num_seqs = 2` | `--max-num-seqs 2`（既定は `1`） |
| `cache.kv_cache_memory_bytes = 6442450944` | `--kv-cache-memory-bytes` 6 GiB（既定は3 GiB） |

1行目から2つの引数が従い、独立した設定ではありません。model引数が `/hf` 配下のMTP metadata viewではなく `/derived` になること（再パックしたcheckpointはBF16のdraft層を自分で宣言する）と、containerのlabelがprofileのfingerprintを持つこと（どのkeyでも変われば変わる）です。command・環境変数のそれ以外はどちらのrankでも同じです。

2026-09-23に、仮値を参照対の値に置き換えたAXLの例は、`server freeze`・`server plan` と両rankの `server preflight` の全検査に合格しました。例外は `startup_memory` だけで、これは稼働中の対の横では成り立ちません。参照対が配信するprofileとの差は `validation.memory_probe` と `api.dev_endpoints`（配信profileでは切替後の検査のためon）、warmupの長文段（配信profileは65,536トークン、例は無し）、使われないLPAの仮値だけです。

## キーの解説

以下の任意キーは、断りが無ければ未指定のとき起動を変えません。キーを足したり変えたりするとprofileのfingerprintが変わるため、次の[切替](launch-safety.ja.md#全レール検査と両rankの切替)から有効になります。imageの対応が要るキーは、[imageの契約](#現行イメージの契約)に挙げたmarkerの無いimageでは `server preflight` が拒否します。

### 再現性のスイッチ

三つのスイッチで、同一要求はbit一致で反復し、どの起動も同じ数値状態で計算します。三つとも両方のexampleで有効です。差の出どころをそれぞれどう見つけて測ったかは[検証](validation.ja.md#フルモデルtp2の実験範囲)にあります。

`runtime.canonical_moe_order`（テンプレートは `true`。未指定はimageの既定に従い、1.6.0から作ったimageでは有効）は、両rankに `GLM53_CANONICAL_MOE_ORDER` を渡します。固定版vLLMの `moe_align_block_size` はexpert内のtokenをCUDAスレッドのスケジューリング順に並べ、MarlinのMoEの結果はその順序にわずかに依存し、後段のrouterがそれを増幅するため、同一要求の反復が一致しませんでした（上流はvLLM issue #52525）。`true` にすると、参照imageがkernelの前に各expertのスロットをtoken id順に並べます。新規の起動には `GLM53_MOE_ORDER_API=2`（1.7.0から作ったimage）が要ります。marker 1は、切替の復旧先として残す稼働中の対にだけ認めます（[起動検査](operations.ja.md#フルモデルの起動検査)）。`false` は比較用のarmで、imageの対応は要りません。expert parallelには手を入れません。参照機では同一要求がbit一致で反復し、decodeは遅くならず、MTPの採択長は上がりました。既定で有効にしているのは、今後のA/Bを読む物差しとして、再現できる基準が要るためです。

`runtime.stable_indexer_topk`（テンプレートは `true`。未指定はimageの既定で、1.6.0から作ったimageではon）は、両rankに `GLM53_STABLE_INDEXER_TOPK` を設定します。kpool indexerは4 tokenを1 poolに畳み、query行ごとに512 poolを選びます。固定の `persistent_topk`（decode）と `top_k_per_row_prefill` は、512位の境界にpoolの同点があると同じ入力から違う*集合*を返し、ある1 stepの同点一つでcompletionが割れます。`true` の時、同点は低いpool indexに決まります。decodeは安定なsort（最大6行、1回0.07〜0.25 ms。kernelは0.01〜0.02 ms。同期なし）、prefillはkernelのまま、512位の値を収まりきらない数のpoolが共有している行だけを選び直します（呼び出しごとに同期1回）。`GLM53_INDEXER_TOPK_API=1` が要ります。`false` は比較用のarmです。

`runtime.inductor_deterministic`（1.12.0からテンプレートは `true`。未指定か `false` はInductorの計測による選択）は、両rankに `TORCHINDUCTOR_DETERMINISTIC=1` を設定し、`TORCHINDUCTOR_CACHE_DIR` を `/root/.cache/torchinductor-deterministic` に移します。複製されたindexerはkeyを、候補configが三つの `torch.compile` のleafで正規化します。以前は各rankが起動のたびに計測で一つを選び、三つのうち一つは行の足し算の順が違うため、両rankが別のclassを引くとpoolが同点になる所でcompletionが分かれました。決定性モードのInductorはreductionのconfigを計測せずに、どのrankでも同じものに決めます。torch 2.13と2.12.1は最初にcompileしたframeの後でモードを切るので（[pytorch/pytorch#198563](https://github.com/pytorch/pytorch/issues/198563)。GB10では [vllm-project/vllm#58636](https://github.com/vllm-project/vllm/issues/58636) に報告）、launcherは `glm53_setup/runtime/inductor_pin.py` と一行の `.pth` をmountし、設定を強制値で保ちます。モードなしでcompileしたgraphは既存のcacheから計測の候補ごと戻ってくるので、モードは専用のcacheにcompileします。keyを付けた最初の起動ではindexerのleafを作り直します（rankごとに約45ファイル）。pointwiseのleafはrankごとに計測を続けますが、要素ごとに同じ命令で計算するのでblockの大きさはbitを変えません。imageの対応は要りません。

これらのスイッチが扱うのは単独の要求です。`max_num_seqs` が2以上だと、他の要求とstepを共有した要求は、なお違うcompletionになりえます。attentionの経路も変わります：MTPの深さ3では相方がいるとdecodeのstepのquery行が8になり、6を超えるのでFA2を通ります（`runtime.fa2_attention`、下記）。ただしこれは主因ではありません：attentionの呼び出しをすべて参照計算に切り替えても、2系列のcompletionの多くは単独のものと違ったままでした（[測定](benchmarks.ja.md#servingでの到達性2026-09-26)）。残る容疑者の先頭は、NVFP4のMarlin MoE（K方向の分け方がstepのexpert block数で決まる）と、prefillと共有したstepのprefill用のkernelです。他に何が走っていても同じcompletionが欲しい場合は `max_num_seqs = 1` で配信します（[同時実行の範囲](validation.ja.md#同時実行の範囲)）。

`runtime.mla_decode_cpb`（1.14.0と1.15.0では両方のexampleで設定）は1.16.0で退役しました。patchが変えるdecodeの呼び出しより前に参照のNoPE attentionがreturnするため、servingでは一度も実行されていません（[servingでの到達性](benchmarks.ja.md#servingでの到達性2026-09-26)）。1.18.0で取り除いたので、keyを持つprofileは、keyを消すよう求めるメッセージとともに拒まれます。

### attentionとcacheとcheckpoint

`runtime.fa2_attention`（未指定はfalse、テンプレートは `true`）は、両rankに `GLM53_FA2_ATTENTION` を設定します。`true` にすると、候補を保持するNoPE attentionのうちquery行が6を超える呼び出しが、参照計算の代わりにFlashInferの `BatchMLAPagedAttentionWrapper`（backendは `fa2`、page sizeは1、各行の候補をその行のKV pageとして渡す）を通ります。packedの `fp8_ds_mla` cacheはそのままで、呼び出しが触る行だけをBF16に展開します。FlashInfer 0.6.18はSM90以外でFP8のMLA KVを受け付けないためです。選ばれた候補はすべて保持するので、この呼び出しより上流のprefix cache・unpack融合・候補の並びは変わりません。6行までの呼び出しは参照経路のままです。`plan()` は各行の長さをhost側に要求し、MLA層ごとに同期が1回入ります。prefillのchunkに対しては安く、decodeのstepに対しては高い費用です。閾値は呼び出し全体の行数で数えます。1系列のdecodeのstep（最大6行＝MTPの深さ5）はすべて参照経路ですが、2系列では深さ3の検証stepが8行になってFA2を通り、ある行の結果が相手の系列の行数と長さにわずかに依存します。参照計算の結果は依存しません（[測定](benchmarks.ja.md#servingでの到達性2026-09-26)）。基準の2台では、38,962 tokenのprefillが約2.2倍速くなりました（[1.6.0での測定](benchmarks.ja.md#160での測定)）。unpack融合は要素数を実行時に受け取るので、256Kの系列で要素数ごとにTritonのkernelを一つcompileすることはもうありません。この経路はLPAと排他です。`server preflight` は、この経路を持つimage（`GLM53_FA2_ATTENTION_API=1`、行 `fa2_attention_support`）を、復旧先も含めて要求します。checkoutは今も、この経路・そのdispatch・上記のunpack融合をimageの上にmountします。

`runtime.prefix_page_dedup`（未指定はoff＝固定vLLMのpoolのまま。AXLの例で設定）は、両rankに `GLM53_PREFIX_PAGE_DEDUP` を設定し、`GLM53_PREFIX_DEDUP_API=1`（1.9.0から作ったimage）を要求します。固定vLLMのblock poolは、同じhashのblockが既にcacheにあっても、fullになったblockをそのhashで登録します。draftがあるとprefix lookupは一致した末尾blockをhitから外して再計算するので、同じ履歴を再送するたびにKV cache groupごとに1 blockが既にあるhashでLRU queueに加わり、その複製が古い履歴を先に追い出します。keyをonにすると、そのblockは登録されません。hashを持たず、要求が終わるとfree queueの先頭に戻り、lookupは先にcacheされた複製にhitし続けます。block idと数値は変わりません（[1.9.0での測定](benchmarks.ja.md#190での測定)）。

`runtime.derived_checkpoint`（任意のtable、既定では無し。table内の `enabled = false` はtableを残したまま固定snapshotを配信する）は、固定checkpointを手元で再量子化した複製を配信します。`path`（両hostの絶対ディレクトリ。`/derived` に読み取り専用でmount）、`requant_target`（そのディレクトリの `quantization_config.producer.requant_target` と照合）、`overlays`（`{target, source, sha256, base_sha256, marker}` の列。`source` は絶対パスのファイルで、image内のGLMモデルディレクトリの `target` に重ねてmount）を取ります。`server preflight` は、checkpointが `MIXED_PRECISION` とそのtargetを宣言していること、MTP draft層に量子化宣言が無いこと、各overlayが指定のSHA-256でmarkerを含むこと、image内の `target` が `base_sha256` であることを確かめ、どれか一つでも違えば失敗します。別のimage向けに作ったoverlayはmountできません。derived checkpointではMTPのメタデータviewを使いません。`MIXED_PRECISION` では宣言の無いmodule（BF16のdraft層を含む）が無量子化で読まれるためです。本リポジトリが配信する2つのoverlayは [`overlays/`](../overlays/README.md) に同梱しています（`kda-quant-split.py` を `kda.py` に、`mla-quant-split.py` を `model.py` に重ねます）。targetは二つあり、`g`（attention projection）と `l`（attention projectionと `lm_head`）です。現行の `l` のcheckpointは分割したKDAのinput projectionを宣言しており（`quantization_config.producer.in_proj_layout = "split-qkv-bfg"`）、この2つのoverlayとしか組めません。以前の融合した対は、このキーを持たないrevisionのものです。基準の2台は `l` をMTP k=3で配信しており、そのcheckpointはoverlayの要件をmodel cardに書いて公開しています（[施策台帳](optimization-catalog.ja.md)のP23、[配信profile](benchmarks.ja.md#基準の2台の配信profile)）。

`cache.prefix_cache_retention_interval`（テンプレートは `"dense"`）と、未指定のときのruntimeの挙動は[APCの履歴検証](launch-safety.ja.md#apcの履歴検証)で説明します。

`runtime.index_checks` は `auto`／`sync`／`async`（配布既定） を選びます。autoはeagerで同期検査、Graphで非同期検査を使い、従来の動作を維持します。asyncを明示すると、独立評価したeagerの非同期検査を選べます（`GLM53_ASYNC_INDEX_CHECK_API=1` が必要）。範囲検査は常に実施します。asyncで不正indexを検出するとCUDA contextが使えなくなる場合があるため、両rankを再起動します。Graphではsyncを拒否します。

### 重みの読み込み

`runtime.safetensors_load_strategy`（未指定では何も渡さずvLLMに任せる。テンプレートには書かない）は、`eager`・`lazy`・`prefetch` のいずれかを両rankのvLLMへ `--safetensors-load-strategy` として渡します。それ以外の値は拒否します。基準の2台では1.18.0で（2026-09-26、4 KiB page、kernel 6.17.0-1032-nvidia）`Loading weights took` がrank 0で532秒、rank 1で192秒、約0.17 GB/sでした。GB10では、CUDA contextの下でのhostからdeviceへのcopyは、fileを割り当てたmappingからだと0.1 GB/s前後、匿名メモリからだと約3.5倍速く走ります。vLLMの既定はshardをfileのmapping越しに読み、`eager` は各shardを `f.read()` で丸ごと匿名メモリへ読んでからtensorをcopyします。代わりにメモリを使います。shardを読む間、少なくともそのshard（18個のうち最大は11.15 GiB）が匿名メモリに載り、unified memoryではそれが重みやKVと同じpoolなので、ほかの割当と同じく `reserve_gib` に効きます。`eager` でこの2台の読み込みが縮むか、実行中に `MemAvailable` がどこまで下がるかは、まだ測っていません。両方を記録してから採用してください。`prefetch` は、mapping越しに読む前にpage cacheを温めるだけです。このkeyを持つprofileは新しいfingerprintになり、持たないprofileは元のままです。

### 並列化と通信

`runtime.nccl_channels`（未指定はNCCLに任せる、テンプレートは8）は、両rankの `NCCL_MIN_NCHANNELS` と `NCCL_MAX_NCHANNELS` に同じ正の整数を渡します。参照機ではNCCL 2.30.7に任せると64本になります。MTU 1500で8本にすると、実モデルの最小空きメモリがheadで2.8 GiB、peerで3.0 GiB増え、prefillは遅くなりませんでした（[チャネル数の測定](nccl-validation.ja.md#チャネル数)）。1.3.1より前に書いたprofileにはキーが無く、NCCLの選択をそのまま保ちます。テンプレートの値を使うにはキーを足します。Mia PR #200を参考にしました。

`runtime.expert_parallel=false` が既定です。有効にすると両rankへ `--enable-expert-parallel` を追加し、TP=2／DP=1、精度、固定KV予算を維持します。`GLM53_EXPERT_PARALLEL_API=1` が必要です。範囲はeager・1／2系列・MTP/LPA/fusion/APCなしです。全モデルで測って不採用としました（[Expert Parallel](performance-investigation.ja.md#expert-parallelp21)）。既存TOMLにもキーを明示し、欠落時のfallbackは設けません。

`runtime.pipeline_parallel_size=1` はTP=2を維持し、2にすると同じ2台でTP=1／PP=2を選びます。`GLM53_PIPELINE_API=1` を持つイメージが必要です。`pipeline_split_layer` は前段stageの層数で、既定候補24なら24／21層に分け、この固定モデルでは各stageに21 MoE層ずつを置けます。容量を保証する値ではありません。両stageにMLAが必要なため、境界の許容範囲は4〜43です。範囲は1系列・eager・EP/MTP/LPA/fusion/APCなしで、PP2は測って不採用としました（[TPとPPの比較](performance-investigation.ja.md#tpとppの比較)）。TOML更新時は両キーを明示してください。

### 画像入力

`runtime.vision` は未指定でfalse、テンプレートは `true` です。`false` は `--language-model-only` を残し、視覚塔を読み込まずテキスト・ツール専用で動かします。`true` は両rankからこのフラグを外し、`--limit-mm-per-prompt '{"video": 0}'` を付けます。**`vision = true` でも動画入力は無効で、送ると拒否されます。受け付けるのは画像だけです。** 理由は起動時のメモリです。vLLMは最大の入力1件を一度エンコードしてメモリを見積もり、このチェックポイントの動画の上限（30,000 token＝120,000パッチに制限済み）は画像1枚（最大8,000 token）よりはるかに大きいためです。1 promptあたりの画像枚数はvLLMの既定のままです（チャットハーネスは過去の画像を毎ターン送り直すため）。Vision有効時は `cache.mm_processor_cache_gb`（未指定は0.1）が `--mm-processor-cache-gb` を決めます。これは前処理済み画像のキャッシュで、vLLMはAPIプロセスとエンジンプロセスの両方に持ち、どちらもrank 0にだけ載ります。vLLMの既定4 GiBのままだと、headのホストRAMを最大8 GiB使い得ます。上限より大きい画像はキャッシュせずに処理し（警告のみ）、拒否はしません。視覚塔はBF16で量子化の対象外です（両方の除外リストに `model.visual*`）。実測、headのメモリ余裕、未解決の事項は[画像入力](vision.ja.md)にあります。検証fixtureはこのキーに関係なくテキスト専用で読み込みます。

### APIと診断

`api.prompt_tokens_details`（未指定は無効、テンプレートは `true`）は `--enable-prompt-tokens-details` を付け、`usage.prompt_tokens_details.cached_tokens` で復元prefix長を返します。無いとvLLMは `null` を返し、cacheが当たっていてもハーネスの表示は0のままです。

`api.dev_endpoints`（未指定はfalse）は両rankに `VLLM_SERVER_DEV_MODE=1` を渡し、loopbackのAPIにvLLMのdev経路（`/reset_prefix_cache`・`/reset_mm_cache`・`/collective_rpc`・`/sleep`・`/wake_up`・`/server_info`）を載せます。LPA・component・expertのprofileは元からこのモードで動きます。このキーは、LPA offのprofileでもベンチやwarmupの後始末のために再起動なしでprefix cacheを消せるようにするためのものです。これらの経路は無認証です（[起動契約](launch-safety.ja.md#モデルapiクライアント)）。他のローカル利用者がいる機体では off のままにします。

`validation.memory_probe`（未指定はfalse）はworker拡張を一つ載せ、dev経路 `POST /collective_rpc`（例：`{"method": "allocator_stats"}`）から呼べるようにします。LPAや他のworker拡張とは排他（起動につき拡張クラスは一つ）で、起動の他の点は変えません。メソッドは次のとおりです。

| メソッド | 返すもの |
|---|---|
| `allocator_stats` | rankごとに、torchのcaching allocatorのreserved・allocated・active・inactive-splitバイト、segment数、確保のretry回数、`mem_get_info`。GB10ではGPUがホストのメモリを共有するので、allocatorの成長はworkerのRSSやcgroupが平らなまま `MemAvailable` の減少として現れる。この二つを切り分ける |
| `host_stats` | workerの常駐匿名メモリ、glibcの `mallinfo2` の内訳、torchのpinned host cache。`{"trim": true}` を付けると二回の読みの間で `malloc_trim(0)` を呼び、保持している空き・生きているオブジェクト・mallocの外のメモリを切り分ける |
| `host_census` | 生きているPythonオブジェクトの型ごとの数と、その中のCPU tensor |
| `weight_digest` | loadされた全parameterとbufferの二つのbyte和（層ごと・全体）。`{"tensors": true}` で行そのものを返す。`tools/weight_digest.py` が切替の後にそれを記録し、前の起動と違うtensorを名指しする |
| `kernel_hashes` | kpool indexerの計算（head gate、gate score、FWHT量子化、pool cacheの書き込み、paged MQA logits、stable top-k）と、1.11.1からは一つの層の実際の重みでの段階（`layer`、既定は19）を、固定入力でworkerの中で走らせたhash。`tools/kernel_hashes.py` が両rankと前の起動を比べる |
| `autotuners`、`inductor_state` | 生きている各Inductor kernelが使っているconfigとその出どころ。Inductorの設定、`TORCHINDUCTOR_*` の環境変数、決定性スイッチへの書き込みのすべて |
| `fa2_stage` | FA2経路の一部だけ（`off`、圧縮の単一操作、`compact`、`plan`、`full`）を走らせて残りを参照経路で答えるので、メモリの増加が始まる段を、段ごとの再起動なしに名指しできる |
| `trace_begin`、`trace_end` | traceするmodule・draft・sparse NoPE attentionが1要求で受け取り返すものの指紋（byte列をint64の語として読んだwrapする和を二つ、device上に保持）。後の同一要求で最初に食い違った呼び出しを名指しする。`sync` はtraceした呼び出しごとに同期する。候補indexのtensorは行ごとに並べ替えてから指紋を取るので、候補の集合が違う時だけ食い違いとして読める。cache行のgatherはdecodeの大きさの呼び出しでだけ行う。`{"keep": true, "export": true}` で行を返し、`trace_differences` が二つの起動をofflineで比べる |

workerのメソッドが例外を出すとHTTP 500が返り、その次の `/collective_rpc` の呼び出しも、内容に関係なくHTTP 500になります。その後の呼び出しは正常です（基準の2台で2026-09-20に計測）。エラーの直後の1回は捨ててください。新しい読みを足す前に、本番の形で費用を測ってください。4層fixtureの形は小さく、その費用は見えません。これらの読みを切替のたびに使う定型は[起動契約](launch-safety.ja.md#切替の後のdecode検査)にあります。

`validation.expert_worker=true` は、実際のexpert配置・kernel・parameter情報を返す型付きRPC `expert_info` を有効にします。独立したeager TP2の基準／EP条件、最大2系列が対象で、他の観測worker・MTP/LPA/APC・PPとは併用しません。層のhash観測は明示的な `pipeline_observe` RPCで初めて開始するため、性能測定中はそのhookを入れません。

`validation.component_worker=true` は、CUDAのA/Bとindexerの観測のための独立した観測workerを選びます（[機能の併用と制約](#lpaとmtp制約)）。

`resources.stall_seconds`（未指定は0＝無効、テンプレートは600）と `generation.warmup`／`generation.warmup_long_tokens`（未指定はfalse／0）は[監視・停滞検知・warmup](operations.ja.md#監視停滞検知warmup)で説明します。認証クライアント、`runtime.cuda_allocator_conf`、`nodes[].additional_rails`、両rankの切替は[起動契約](launch-safety.ja.md)にあります。

### prefix cacheと併用するLPA

P22のGPU状態隔離・校正・最終併用・held-out参照評価を、範囲を限って確認済みです。LPAとprefix cachingを両方有効にする場合は `GLM53_APC_LPA_API=1` のimageが必要です。schedulerが全状態を揃えて復元したprefixと `lpa.break_even_tokens` から適用を決めます。テンプレートは[P22の校正](benchmarks.ja.md#apc優先lpaの損益分岐計測p22)に基づく保守的な閾値128を使います。最初の近似以降は、通常計算する末尾・decodeを含めて共有登録を止めます。このモードの `server ask` は判断をサーバーへ任せ、APCなしの通常の `server ask` はH=0として同じ閾値を使います。テンプレートは `lpa.enabled = false` を既定とし、バッチ入力に限って用途ごとに有効化します（近似した要求は共有cacheに何も登録しないため）。有効にしている間、要求に `"vllm_xargs": {"glm53_lpa_mode": "off"}` を指定すると通常計算し、通常状態の共有cacheを育てられます。更新するTOMLには閾値キーを明示してください。[実装契約](apc-lpa-design.ja.md)と[LPA](lpa.ja.md)を参照してください。

## コマンド

コンテキスト長や同時数を変更する前に、[KV容量とRAMの条件](#kv容量とramの条件)も確認してください。

リポジトリ直下で生成される起動条件を確認します。これはWindowsでも実行できます。

```sh
python -m glm53_setup server plan --rank 0
python -m glm53_setup server plan --rank 1
```

各Linuxノードの `server preflight --rank N` でモデル・fabric・イメージID・他のコンテナがGPUを使っていないこと・空きメモリを確認できます（[各検査の範囲](operations.ja.md#フルモデルの起動検査)）。動いていない対を起動するには、worker側でrank 1を先に、head側でrank 0を後に、それぞれの端末で起動します。稼働中の対を置き換えるには[両rankの切替](launch-safety.ja.md#全レール検査と両rankの切替)を使います。

```sh
python -m glm53_setup server start --rank 1
python -m glm53_setup server start --rank 0
```

起動コマンドは前面に残り、自分のコンテナを監視します。端末を維持してください。`resources.run_seconds` の期限には**モデルのロード時間も含まれます**。Ctrl+C・期限到達・空きメモリ不足でそのランクを停止します。分散実行に異常が出た場合は両ランクを停止します。コンテナと `records/` のログ・起動設定は残し、自動削除や自動再起動はしません。

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

`agreement` は自作の4文（日本語・英語・コード・数理）を request lock の下で `/v1/completions` に `prompt_logprobs` 付きで2回ずつ送り、実際の次tokenの順位とlog確率を `records/<stamp>-agreement-r0/result.json` に書きます。`--reference <result.json>` を付けると、その過去の実行に対するargmax一致・top-5の重なり・log確率の移動を加えます。`self_agreement` は実行内の差を持ちます。再現性のスイッチが有効なら同じ文の2回はbit一致し、`runtime.canonical_moe_order` 以前はargmax一致が約96%で、log確率が8 nat動いた位置もありました。記録は、全要求が検査した形で返れば合格とします。文ごとの平均負log確率が、実行間で安定した読みです。実行内の一致は、起動を跨いだ一致の証拠ではありません（[候補の比較](validation.ja.md#候補と無改変対照の比較)）。

`capacity` は稼働中headの起動ログと `/metrics` を読み、KV poolをそのまま表示します。stockの `GPU KV cache size` 行を `num_gpu_blocks`・最大長要求1本あたりのblock数・group別block幅に分解します。会話の保持本数の推定（16K・64K・`max_model_len` でのblock数と本数）は、LPA worker extensionを載せたprofileでだけ表示します。`apc_cache_layout` RPCが各groupのspec種別を返すためで、それ以外では推測せず withheld と表示します。`warmup` は要求ロックの下でladderを流し、`records/<stamp>-warmup-r0/result.json` に記録します。段が失敗すると非ゼロで終了します。`mojibake` は同じロックの下で稼働中のheadに日本語と韓国語の長い回答を求め、回答とreasoningの化け文字を数えて `records/<stamp>-mojibake-r0/result.json` に記録し、全回答が合格でなければ非ゼロで終了します（[検査の内容](validation.ja.md#フルモデルtp2の実験範囲)）。

workerの状態確認・停止はworker上で `--rank 1` を使います。全コマンドで `--config 設定ファイル.toml` を指定できます。設定を編集したら両ランクを停止・再起動してください。送信時には起動中の設定との一致を検査します。`generation` は専用送信コマンドの既定値で、他のAPIクライアントの生成設定はそのクライアント側で指定します。

## 自動停止と連続稼働

`resources.run_seconds` はロード時間込みの自動停止期限（秒）です。**`0` で時間制限なし**、正の整数で指定秒数後に停止します。負数は受け付けません。どちらの場合も `resources.reserve_gib` によるメモリ保護は有効です。設定は起動時に読み込むため、ファイル編集だけでは起動中の監視プロセスの期限は変わりません。

```toml
[resources]
# 既存のresources節にある、この値を変更します。
run_seconds = 0
```

これは予定時刻で停止しない設定であり、24時間365日の可用性を保証するものではありません。OS起動時の自動起動・切替の外での両ランク協調復旧・冗長構成への切り替えは未実装で、前面の監視プロセスを維持する必要があります。

## KV容量とRAMの条件

**最大長の要求をB本同時に保持するなら、入出力合計の上限Cに対してB×C token分を収容できる容量の確認が必要です。** `max_model_len`は入力と生成の合計上限、`max_num_seqs`は同時実行の上限です。この二つを設定するだけで、最大長×同時数のKVが確保・検収されるわけではありません。2026-09-12の同時2系列の評価は1要求2,112 tokenまでで、公開した任意設定では2026-09-23に約200Kの要求2本を同時に配信しました（[1.10.2での測定](benchmarks.ja.md#1102での測定)）。Spark 2台の他レシピでは、25〜100Kの要求2本の同時処理が合計約4 tok/sまで落ちたと報告されています（tonyd2wild #14、コードは採用しない）。

このランチャーの`cache.kv_cache_memory_bytes`は、**各rankで要求間共有する固定KV poolのバイト予算**です。1 GiBを指定したまま同時数を1→2にしても、各rankのKV予算は1 GiBのままです。要求1本あたり1 GiBでも、2台の予算を自由に合算した一つのpoolでもありません。バイト指定時は`gpu_memory_utilization`によるKV容量の自動推定を使わないため、この比率をRAM全体の保護上限として扱いません。[vLLMの設定仕様](https://docs.vllm.ai/en/latest/configuration/engine_args/#kv-cache-memory-bytes)

実行中に必要なcacheは、保持中の各要求の入力＋生成済みtokenに従ってpool内のblockを消費します。最大長を同時に保証したい場合は、出力予算も含む最大条件で検証します。GLMは疎MLA・IndexPool・系列ごとのKDA状態を併用するため、一般的なdense attentionの単純なbytes/token式をそのまま使わず、**固定runtimeのcache spec・block整列・各groupの容量と状態slot数**で見積もります。MTP等の追加状態も別途含めます。

各ノードで、重み＋KV/cache状態＋activation・indexer等の一時領域＋MTP/Graph等の追加領域＋CPU/OS・他負荷＋運用余裕が、利用可能な統合RAMに収まる必要があります。KV poolを固定しても、context・chunk・同時数に依存する別の割当が増えることはあります。コンテナ上限と`reserve_gib`は保護手段であり、容量適合や無停止の保証ではありません。以前の 256K テキスト専用 profile は実測 121 GiB のホストでこの保護に接していました。available は 4.5 GiB 前後で推移し、保護余裕 4 では監視停止（`stop-reason: memory-reserve`）が 2 回発生し（2 回目は 16,859 token の近似要求の最中）、その後 3 へ下げました。画像入力構成の配信中の head の空きは、NCCL の 64 チャネルでは約 4.0〜4.2 GiB（[実測](vision.ja.md#メモリ最終構成)）、8 チャネルでは [1.3.1 の 200K 実入力](benchmarks.ja.md#131での200k実入力)で 6.97 GiB 以上、chunk 2048 で 6.40 GiB 以上でした。256K・KV 3 GiB では 5.82 GiB 以上で（[1.5.0 での測定](benchmarks.ja.md#150での測定)）、テンプレートは 3 です。値は小数でも指定できます。変える前に、この保護が捉えられる範囲を見積もってください。監視は `MemAvailable` を 2 秒ごとに読み、実測では割り込んだ標本からコンテナ終了まで約 9 秒かかりました。実行中のコンパイルで測った 0.15 GiB/秒の下降なら、reserve から約 1.6 GiB 下まで沈み得ます。参照ホストではカーネル・コンテナとも OOM kill の記録はなく、ホスト側のページは 16 GiB の swap が受けるため、reserve を割った先の危険はワーカー内の GPU 確保の失敗です。その場合は監視停止ではなく、エンジンが突然終了します。ドライバの `NV_ERR_NO_MEMORY` カーネルメッセージは空き 4 GiB 以上、主にロード中に出るもので、底の目印にはなりません。

KVが不足すれば起動が拒否される場合があり、実行時は待ちやpreemption・再計算により性能が落ちることがあります。固定KV poolが勝手に必要量まで拡張されるわけではありません。KV以外の割当やRAM予算が不足すればOOMやガード停止も起こり得ます。[vLLMのpreemption説明](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption)

起動行 `GPU KV cache size: N tokens, Maximum concurrency for L tokens per request: Cx` は、このhybridモデル（MLA・IndexPool tail・KDA state群・MTP draftが一つのblock poolを共有し、整列した区間ごとにgroup別のidを使う）では `N = C × L` です。`N` は同時実行数をtoken単位で表した値で、prefix cacheが保持できる会話tokenの数ではありません。`server capacity` が分解を表示し、group種別が分かる場合は会話本数の推定も出します。測った画像入力構成の二つでは、KV 1 GiBに4,608 tokenのblockが28個入り、長さLの要求1本はそのうちceil(L / 4608) + 16個を使いました。204,800 tokenと2.5 GiBでは70個のうち61個、262,144と3 GiBでは84個のうち73個で、どちらも1.15倍です。256Kの値は切替前にこの数え方で見積もったものです。他の長さ・KV量・group構成では、それぞれの起動行を確かめてください。`GLM53_KPOOL_RING=1` を持つimageでは、IndexPool tail groupのblockがMTPの深さで変わります。MTPなしは4、k = 1〜4は8、k = 5は16です（それ以前のimageはどのkでも4）。実行中の要求1本につき1 blockです。起動行の `kv cache group sizes` に表れます。上の実測値はpatchのないimageのもので、整列したblockと要求1本あたりのblock数が一緒に動くかどうかは、起動から記録するfingerprintと同じく、作り直したimageの起動行で読みます。prefix cacheもblock単位で働くため、同じN tokenのpromptを繰り返したとき復元されるのは `(floor(N / block) - 1) x block` tokenで、2 block未満では一切復元されません。画像profileのscheduler block 4,608 tokenでの実測は、3,625 tokenで0、14,025 tokenで9,216、28,025 tokenで23,040でした。短い会話はこのprofileでは再利用の恩恵を受けません。起動時のcache容量・最大並列度は計算上の目安として保存し、**意図する入出力長×同時数の実要求、preemption回数、両rankの空きメモリ最小値、OOM・ガード停止**を確認してから対応範囲を表明します。現行preflightの合格は、最大長×同時数の収容試験の代わりにはなりません。実測範囲は[標準batchingの独立評価](benchmarks.ja.md#標準batchingの独立評価)を参照してください。

## 現行イメージの契約

起動するcheckoutから参照imageをビルドし（`python -m glm53_setup build-reference`）、両ホストでIDを確認して `reference_image`（LPAでは `runtime.lpa_image` も）に設定します。旧コマンド・旧イメージへのフォールバックはありません。現行ソースから作ったimageは下のmarkerをすべて持ちます。表は、どの設定のときに `server preflight` が各markerを要求するかと、どの版から作ったimageがそれを持つかを示します（—：CHANGELOGがmarkerを記録する以前から）。

| marker | preflightが要求する条件 | 持ち始めた版 |
|---|---|---|
| `GLM53_REFERENCE_ATTENTION=1` | 常に（`reference_attention`） | — |
| `GLM53_LPA_API=2` | `lpa.enabled`（`lpa_worker`） | — |
| `GLM53_APC_LPA_API=1` | LPAとprefix cachingの併用（`apc_lpa_support`） | — |
| `GLM53_FUSED_UNPACK_SUPPORTED=1` | `cache.fused_unpack`（`fused_unpack_support`） | — |
| `GLM53_ASYNC_INDEX_CHECK_API=1` | 非同期index検査（`async_index_check_support`） | — |
| `GLM53_DECODE_GRAPH_API=1` | `runtime.decode_graphs`（`decode_graph_support`） | — |
| `GLM53_EXPERT_PARALLEL_API=1` | `runtime.expert_parallel` または `validation.expert_worker`（`expert_parallel_support`） | — |
| `GLM53_PIPELINE_API=1` | `runtime.pipeline_parallel_size = 2`（`pipeline_support`） | — |
| `GLM53_COMPONENT_API=1` | `validation.component_worker`（`component_worker`） | — |
| `GLM53_MOE_ORDER_API=2` | `runtime.canonical_moe_order`（`moe_order_support`。marker 1は復旧先に限る） | 1.7.0（1は1.6.0から） |
| `GLM53_INDEXER_TOPK_API=1` | `runtime.stable_indexer_topk`（`indexer_topk_support`） | 1.6.0 |
| `GLM53_FA2_ATTENTION_API=1` | `runtime.fa2_attention = true`（`fa2_attention_support`） | 1.6.0 |
| `GLM53_SLOT_MAPPING_GUARD=1` | 要求しない。無いと公開した任意設定で約25万tokenを超える要求が失敗する（[運用手順](operations.ja.md#フルモデルの起動検査)） | 1.7.0 |
| `GLM53_PREFIX_DEDUP_API=1` | `runtime.prefix_page_dedup`（`prefix_dedup_support`） | 1.9.0 |
| `GLM53_KPOOL_SEED_STRIDE=1` | 要求しない（[運用手順](operations.ja.md#フルモデルの起動検査)） | 1.13.0 |
| `GLM53_KPOOL_RING=1` | 要求しない（[運用手順](operations.ja.md#フルモデルの起動検査)） | 未リリース（1.19.0のbranch） |

1.14.0から1.17.0までに作ったimageは、取り除いた `runtime.mla_decode_cpb` の `GLM53_MLA_DECODE_CPB_API=1` と、届かないpatchも持ちます。どの検査もそれを読まず、害はありません。

imageが持つmarkerは `docker image inspect IMAGE --format '{{json .Config.Env}}'` で確かめられます。

## LPAとMTP・制約

`runtime.decode_graphs`（テンプレートは `false`、未指定はeager）がdecode Graphの唯一のスイッチです。`true` で `CompilationMode.NONE`・`FULL_DECODE_ONLY` を渡します。capture size は一つで、MTP有効時は `num_speculative_tokens + 1`（固定ランタイムはdecodeのsizeをこの倍数に切り上げ、`[1]` は拒否します）、無効時は `1` です。prefillはcompileしません。同時1シーケンスならMTP・prefix cacheと併用できます。expertのtoken順を固定した4層MTP fixtureで、eagerとgraphは全長さでtokenもlogprobも一致しました（[部品検証](component-validation.ja.md#decode-graphのfixture独立評価)）。LPAはeagerが必要で、複数系列のGraph設定は起動設定で拒否します。全モデルではGraphのdecodeがeagerより遅く、採用していません（[全モデルでのdecode Graphs](benchmarks.ja.md#全モデルでのdecode-graphs)）。以前の書き方 `runtime.enforce_eager`（`false`＝Graph）も読むので既存profileのfingerprintは変わりませんが、両方を書く場合は矛盾させないでください。

Graph経路では内部候補indexの範囲検査をGPU上で非同期に行います。不正indexを黙って許容せずdevice assertにしますが、通常のPython例外と異なりCUDA contextが使用不能になり得るため、障害時は両rankを停止して再初期化します。Graph用のメモリ保持・起動時capture時間も比較対象です。[vLLM #53366](https://github.com/vllm-project/vllm/issues/53366) は、compile cacheのhashに投機token数が入っていないと報告しています。compileするGraphをMTPと併せて使う場合は、kごとにcacheを分けるか、kを変えたときに消してください。

`validation.component_worker=true` は部品検証専用workerを選び、`GLM53_COMPONENT_API=1` を持つイメージを要求します。eager・同時1シーケンス・LPA/MTP/prefix cacheなしの独立構成です。型を制限したRPCでindexerの候補・時間を採取し、排他的なリクエスト間でunpack融合を切り替えてA/B/Aを検証できます。Reuse/Reindexを本番適用する設定ではありません。制御クライアントは一つに限定します。CUDA融合の実モデルA/Bとindexerの採取結果は[部品検証記録](component-validation.ja.md)にあります。

`cache.fused_unpack=true` が配布既定です。falseならTorchの参照変換を使い、trueなら656バイトのMLAキャッシュからのFP8変換とFP32スケール乗算を一つのTritonカーネルで処理し、`GLM53_FUSED_UNPACK_SUPPORTED=1` が必要です。候補集合の変更や層間のKV共有は行いません。

`mtp.enabled` と `lpa.enabled` を個別に切り替えます。MTP有効時は [prepare_mtp_view.py](../tools/prepare_mtp_view.py) で作成したviewとBF16 Triton下書きバックエンドを使います。`mtp.num_speculative_tokens` は1〜5を受けます。五つとも再量子化したcheckpointで、1・3・4は固定のcheckpointで測定済みで、両方のexampleは3を使います（[投機デコード](speculative-decoding.ja.md#両方のcheckpointで深さ32026-09-21)）。LPA有効時は `runtime.lpa_image` を選び、projectorを読み取り専用でマウントしてworker拡張を有効にします。LPAは `runtime.fa2_attention` と排他で、MTPと併用するときは深さ1・2・3を受け（4と5は検証で拒否します）、MTP対応を明示したLPA workerを含むイメージが必要です。深さ2のLPAは4層fixtureでだけ確かめています（[部品検証](component-validation.ja.md#apc優先lpaのcache隔離p22)）。`lpa.py` と `apc_worker.py` はマウントではなくイメージに焼き込まれるため、深さ2を許す前に作ったイメージは要求のたびにこれを拒否します。

LPAはリクエストごとの入力長が必要です。専用クライアントが実際のテンプレートでトークン数を求め、worker設定→生成→トークン数の一致確認→LPA解除まで行います。入力全体が `lpa.tail` に収まる短文は通常計算です。制御するクライアントは一つに限定してください。専用CLI同士はhead上のロックで直列化しますが、直接APIを呼ぶ他クライアントまでは調停しません。専用送信コマンドは非ストリーミングのテキスト・ツール会話用です。

範囲はTP=2・Marlin W4A16・FP8 KV、テキスト・ツール呼び出し・画像です。LPAには同時1シーケンス・eager実行が必要で、LPAとprefix cacheの併用は上記のP22経路を使います。同時2系列以上を受け入れているのは、公開した任意設定の同時2系列profileだけです（[同時実行の範囲](validation.ja.md#同時実行の範囲)）。コンテキスト長・チャンク・キャッシュ量・cut・tailを変えた場合は再測定が必要で、設定検査の合格は品質や必要メモリの保証ではありません。[LPA](lpa.ja.md) と [MTP](speculative-decoding.ja.md) に検証範囲を記載しています。
