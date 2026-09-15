# LPA：後段Prefill近似 — 実験機能

[English](lpa.md) · [基準ベンチ](benchmarks.ja.md)

前段の表現から後段各層の正規化済みAttention入力を予測し、GLM本来のAttention処理でキャッシュと再帰状態を作る。過去tokenのMLP計算を省略し、生成時は全層を実行する。本体のcheckpointは変更せず、小さな補助器だけを学習する。

機能名は **LPA（Late-prefill approximation）**。起動設定は `[lpa]`、CLIは `lpa-fixture`・`lpa-corpus`・`lpa-train` と表記する。[現行イメージの契約](startup-configuration.ja.md#現行イメージの契約)に合わせて再ビルドし、旧インターフェースは残さない。

着想は[きしだ氏のQwen3実験](https://nowokay.hatenablog.com/entry/2026/09/11/120001)から。GLMへの適用には、独立した状態・品質の検査を設けている。

疎MLAにはlatent cache・indexer・未完poolのtail、KDAには畳み込み履歴と再帰状態が必要になる。既存の更新経路を使い、各tokenを一度だけ処理する。`skip_mla_queries=true`では、固定した参照MLA backendで未使用の過去query計算だけを省く。計算するqueryの候補は全て保持し、過去tokenのmHC・KDA出力計算は残す。

補助器は「学習した対角scale＋低rank残差」。層間で入力スケールが異なるため、identityを前提とした残差だけではKDAからMLAへの境界を近似しにくい。成果物にはcut・層数・形式版とtensorを保存し、読込時に形状・精度・有限値を検査する。教師と同じ固定checkpoint・演算条件で使い、読込済みの補助器は要求間で再利用する。

## 使用範囲

**LPAとprefix cacheの再利用は両立しないため、配布テンプレートは `lpa.enabled = false` を既定とし、バッチ用のopt-inとして扱う。** 近似した要求は共有prefix cacheに何も登録せず、抑止は通常計算の末尾とdecodeまで続く（近似区間より後ろのblockも近似履歴に依存するため）。伸びるプロンプトを毎回送り直す用途（チャット・コーディングハーネス全般）では、LPAを有効にしている限り再利用可能なprefixが育たない。`break_even_tokens` は単一要求のprefill費用しか比較しておらず、以後の全要求が失う再利用を勘定に入れられないからである。LPAは「一度だけ処理する長い入力」に使い、会話には使わない。

配布用TOMLではMTP併用を有効にしている。修正済みの四層MTP3／unpack融合／async fixtureは、`integration-fixture-v36` で3〜8,192 tokenの8条件と使用中のKDA状態比較を通過した。その後の[直列フルモデル比較](benchmarks.ja.md#直列併用の評価p18)に、限定した課題・速度・容量・切断復帰の受入結果を記録している。部品のtoken一致を一般的なフルモデル品質保証とはしない。

- eager、テキスト専用、TP=2、同時実行1、制御クライアント1つ。sequence-parallel MoE・複数の制御クライアントは未対応。APCなしのMTP k=1/k=3併用は `allow_mtp=true` を明示し、[起動設定TOML](startup-configuration.ja.md)では自動設定する。APCは以下のscheduler統合済みP22経路を使う。
- 入力末尾の指定範囲は通常計算する。全入力が保護範囲に収まる場合は、補助器の読込・実行をせず通常経路へ戻す。
- 過去の状態は近似になる。Decodeを全層で行っても、元モデルの確率分布が厳密に復元されるわけではない。
- 実験用worker extensionであり、通常ランチャーや各コーディングハーネスの常用検収とは別。開発用RPCはloopbackに限定する。

### APCと共有状態の由来

LPAはKVとKDA状態を近似する。近似で作ったキャッシュをLPA off要求や別のprojector設定へ混ぜることはできない。手動の `lpa_configure` RPCではschedulerの共有登録境界を確定できないため、prefix cachingとの併用を拒否する。

P22は[APC優先・未処理部分へのLPA方式](apc-lpa-design.ja.md)を実装する。schedulerが全状態を揃えた通常計算由来のprefix Hを復元し、N/H/Tと損益分岐の閾値から残余の近似区間を決める。最初の近似以降は、通常計算する末尾・decodeまで共有登録を抑止する。共通資料を通常計算してcacheを作る要求指定を設け、近似状態は要求内だけで使う。

対応imageのmarkerと起動設定が必要になる。CPU契約、4層GPUの共有状態隔離試験、全モデルでの校正、MTP／融合／非同期検査との併用、held-out評価まで完了しており、その状態は[設計契約](apc-lpa-design.ja.md)が正典。APCとのcapture／oracle RPC併用は有効にしていない。改善率は単体結果から足し合わせず、実際の組合せで測る。

## 学習済みprojectorの取得

検証済みcut32 projectorを、**Apache-2.0**の独立した[GitHub Release添付物](https://github.com/Bizuayeu/GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ/releases/tag/lpa-cut32-v1)として配布します。取得URL、正確な容量・hash、形式、教師モデル、学習来歴の正典は[config/lpa-projector.lock.json](../config/lpa-projector.lock.json)です。NVIDIAの本体checkpointは別途取得します。このprojectorを使うための再学習は不要です。

**両方のLinuxホスト**のcheckoutで実行します。

```sh
mkdir -p state/lpa
curl --fail --location --output state/lpa/glm53-lpa-cut32-v1.tar.gz https://github.com/Bizuayeu/GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ/releases/download/lpa-cut32-v1/glm53-lpa-cut32-v1.tar.gz
curl --fail --location --output state/lpa/SHA256SUMS https://github.com/Bizuayeu/GLM-5.3-Flash-NVFP4-2x-DGX-Sparks-BIZ/releases/download/lpa-cut32-v1/SHA256SUMS
(cd state/lpa && sha256sum --check SHA256SUMS)
```

アーカイブのchecksumが合格した場合だけ、次へ進みます。

```sh
tar -xzf state/lpa/glm53-lpa-cut32-v1.tar.gz -C state/lpa
python -c 'import hashlib,json,pathlib; p=pathlib.Path; m=json.loads(p("config/lpa-projector.lock.json").read_text()); f=p("state/lpa")/m["artifact"]/m["file"]; assert f.stat().st_size == m["bytes"]; assert hashlib.sha256(f.read_bytes()).hexdigest() == m["sha256"]; print("Projector verified:", f)'
```

`state/startup.toml`の`[lpa].projector`を`"lpa/glm53-lpa-cut32-v1/projector.pt"`にし、lockの`sha256`を`[lpa].projector_sha256`へ転記します。`cut=32`を維持し、tailと損益分岐値はテンプレートの実測設定を使います。対応するLPA imageを指定し、[対応する用途](#使用範囲)でのみ`lpa.enabled=true`にします。LPAの実測範囲はテキスト専用なので、そのprofileは`runtime.vision=false`にしてください。稼働中profileの変更には通常の[両rank切替](launch-safety.ja.md#全レール検査と両rankの切替)が必要です。取得だけではLPAの有効化もサーバー再起動も行いません。

### モデルカードと学習来歴

凍結した教師のAttention入力にridge回帰で合わせた、対角scale・共有低rank基底・層別の残差写像です。cut層は自身の入力を使い、学習した写像は33〜44層を担当します。重みファイルに含むのは学習済みtensorとscalarメタデータで、コーパス本文・教師活性の採取物・認証情報・実機設定は含みません。配布前にメタデータ、tensor形状、有限値、SHA-256を検査し、下記の実測に使用したprojectorとバイト一致を確認しました。

学習元はlockに記載した固定LLM-jp Corpus v3の日本語・英語Wikipediaと、許容ライセンスで絞ったC++です。学習splitのみでfitし、validationで候補を選び、test文書は分離しています。上限付きの標本取得のため、元shard全体のchecksumは未検証です。学習データのライセンス・帰属は[projectorのライセンス](licensing.ja.md#lpa-projector)と分けて保持し、データ本体の再許諾・同梱は行いません。下記の小規模な評価は、一般品質、記憶再現の不存在、他のcheckpoint・精度への適合を証明するものではありません。

## 部品の再現

GPUコマンドは教師と同じ固定reference image内で実行する。helpとコーパス採取にはTorchは不要。

```sh
python -m glm53_setup lpa-corpus --output records/corpus-ja --documents 512
python -m glm53_setup lpa-corpus --subset en-wiki --output records/corpus-en --documents 128
python -m glm53_setup lpa-corpus --subset code --shard 300 --output records/corpus-code --documents 128
python -m glm53_setup lpa-fixture --fixture /fixture --output /out/oracle --cut 0 --skip-mla-queries --lengths 3 4 5 127 128 129 511 512 513 8705
python -m glm53_setup lpa-train --captures /out/teacher --output /out/projector --cut 40 --rank 256 --ridge 0.001
```

文書数・rank・ridgeはpilot用の値で、最適値ではない。採取器はLLM-jp corpusのrevisionを固定し、出典metaを保持して転送量に上限を置く。正規化した全文hashでtrain/validation/testを分ける。コードはdatasetの元リポジトリ別ライセンス情報からMIT/Apache/BSD/ISC表記のものを選ぶ。コーパス全体を本リポジトリのApacheライセンスとして扱わず、出力した出典・各subsetの条件を保持する。[LLM-jp corpusの説明](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3)を参照。

## 教師採取と実験の制御

reference imageのサーバーに`--worker-extension-cls glm53_setup.runtime.lpa.LPAWorkerExtension`と`VLLM_SERVER_DEV_MODE=1`を指定する。私用の`/collective_rpc`から`lpa_configure`と`lpa_report`を呼ぶ。設定と対象要求の間に別の要求を入れず、サーバーと同じtokenizer・template設定で実際の入力長を求める。

`lpa_configure`の引数は`mode`・`cut`・`prompt_length`・`tail`、予測時は`predictor_path`。modeは`off`・`capture`・`oracle`・`identity`・`predict`で、返り値には要求modeと実効modeを分けて載せる。`capture`は教師入力を採取し、`oracle`は同じ入力の採取値で状態注入を検査する。`identity`は診断用であり、学習済み近似器ではない。

`lpa_report(output=...)`は新しい出力先へ`rank-N/layer-L.pt`を保存する。学習側は、`cut`と`cases`を持つ採取完了の`result.json`を読む。各caseに`id`・`split`・`prompt_tokens`、tensorは`<id>/rank-0/layer-L.pt`として置く。コードの途中を切り出しても文書単位のsplitを維持する。広く採った後段の教師データは、後ろ寄りのcutにも再利用できる。trainで学習し、validationで候補を選び、testは最終評価まで使わない。

`profile=true`で層全体とAttention/MLPのCUDA event区間を計測する。Prefillを分離する場合は出力1tokenにし、最終の実時間比較では計測hookを無効にする。モジュール内の通信時間は各区間に含まれ、独立した通信計測ではない。

query省略はcall内に限定し、既定では無効。固定した参照backendを必要とし、layout不一致・想定外のbackend呼出しは明示的に失敗させる。KV作成はread-onlyのquery計算より前で維持する。oracle fixtureでは通常処理への復帰と、実際に省略されたquery数も検査する。

## 検証の現在地

4層fixtureではpool・conv・chunk・cache block境界を検査し、全モデルのoracle試験も8,192 tokenまで実施した。生成列／文字列の一致と数値差は分けて記録している。1 tokenのfixtureは近似なしの対照でも不安定で、検収対象には含めていない。

LLM-jpの日本語・英語Wikipediaと、許容したライセンス表記のC++サンプルから補助器を学習した。選定した実験profileはcut=32、後段13層、末尾512 tokenを通常計算し、参照MLAの不要queryを省く構成。cut=24の21層版は長文照合値を誤答したため不採用とした。

| 事前tokenizeした入力 | 通常経路：A/復帰Aの中央値平均 | 近似経路中央値 | 所要時間の短縮 |
|---:|---:|---:|---:|
| 2,048 token | 6.174秒 | 5.118秒 | 17.1% |
| 8,192 token | 24.763秒 | 19.409秒 | 21.6% |

単一client・TP=2の参照profileで、出力1 token、warmup別に各条件5回測定した。入力はvalidation文書から構成し、設定RPCとtokenizeは計測区間外。Prefill／最初のtoken応答の比較であり、Decode高速化や一般的な半減を示すものではない。

近似が有効な長文参照・長文コード6件と、長文からのtool往復は合格。未使用文書8件の照合値は全件正答したが、1件に説明追加があり初回の厳密形式は7/8だった。その同じ課題の再確認は通常・近似とも4/4合格。SSE、生成中に接続を閉じた後の復帰、cold restart後の長文問い合わせも通過した。通常モデルにも書式揺れがあり、小標本による統計的非劣性保証ではない。

生データ・教師活性の採取物・実機固有の制御コードは非公開の`records/`に保持し、選定した補助器だけを上記の方法で別途配布する。通常ランチャー／ハーネスの資格ゲートは変更していない。
