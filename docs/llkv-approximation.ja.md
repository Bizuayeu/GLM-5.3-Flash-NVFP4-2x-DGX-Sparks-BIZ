# LPA：後段Prefill近似 — 実験機能

[English](llkv-approximation.md) · [基準ベンチ](benchmarks.ja.md)

前段の表現から後段各層の正規化済みAttention入力を予測し、GLM本来のAttention処理でキャッシュと再帰状態を作る。過去tokenのMLP計算を省略し、生成時は全層を実行する。本体のcheckpointは変更せず、小さな補助器だけを学習する。

機能名は **LPA（Late-prefill approximation）**。起動設定は `[lpa]`、CLIは `lpa-fixture`・`lpa-corpus`・`lpa-train` と表記する。既存の `llkv-*` コマンド・RPC名・文書URLも互換性を保つ。旧イメージでは再ビルドまで従来のCLI名を使う。

着想は[きしだ氏のQwen3実験](https://nowokay.hatenablog.com/entry/2026/09/11/120001)から。GLMへの適用には、独立した状態・品質の検査を設けている。

疎MLAにはlatent cache・indexer・未完poolのtail、KDAには畳み込み履歴と再帰状態が必要になる。既存の更新経路を使い、各tokenを一度だけ処理する。`skip_mla_queries=true`では、固定した参照MLA backendで未使用の過去query計算だけを省く。計算するqueryの候補は全て保持し、過去tokenのmHC・KDA出力計算は残す。

補助器は「学習した対角scale＋低rank残差」。層間で入力スケールが異なるため、identityを前提とした残差だけではKDAからMLAへの境界を近似しにくい。成果物にはcut・層数・形式版とtensorを保存し、読込時に形状・精度・有限値を検査する。教師と同じ固定checkpoint・演算条件で使い、読込済みの補助器は要求間で再利用する。

## 使用範囲

MTP併用は明示的に有効化する。四層fixture＋MTP k=3では3〜8,192トークンの8条件でcapture・通常・oracle・復帰後の出力tokenが一致した。初期の状態診断は未使用の畳み込み巻き戻し領域も含めていたため、prefillで使う範囲だけを比較するよう修正した。修正した診断の再確認と、フルモデル併用時の品質・速度評価は未実施。部品のtoken一致をフルモデルの品質保証とはしない。

- eager、テキスト専用、TP=2、同時実行1、制御クライアント1つ。APC・sequence-parallel MoE・複数の制御クライアントは未対応。MTP k=1/k=3との併用は `allow_mtp=true` を明示する。[起動設定TOML](startup-configuration.ja.md)では両機能の有効化から自動設定する。
- 入力末尾の指定範囲は通常計算する。全入力が保護範囲に収まる場合は、補助器の読込・実行をせず通常経路へ戻す。
- 過去の状態は近似になる。Decodeを全層で行っても、元モデルの確率分布が厳密に復元されるわけではない。
- 実験用worker extensionであり、通常ランチャーや各コーディングハーネスの常用検収とは別。開発用RPCはloopbackに限定する。

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

reference imageのサーバーに`--worker-extension-cls glm53_setup.runtime.llkv.LLKVWorkerExtension`と`VLLM_SERVER_DEV_MODE=1`を指定する。私用の`/collective_rpc`から`llkv_configure`と`llkv_report`を呼ぶ。設定と対象要求の間に別の要求を入れず、サーバーと同じtokenizer・template設定で実際の入力長を求める。

`llkv_configure`の引数は`mode`・`cut`・`prompt_length`・`tail`、予測時は`predictor_path`。modeは`off`・`capture`・`oracle`・`identity`・`predict`で、返り値には要求modeと実効modeを分けて載せる。`capture`は教師入力を採取し、`oracle`は同じ入力の採取値で状態注入を検査する。`identity`は診断用であり、学習済み近似器ではない。

`llkv_report(output=...)`は新しい出力先へ`rank-N/layer-L.pt`を保存する。学習側は、`cut`と`cases`を持つ採取完了の`result.json`を読む。各caseに`id`・`split`・`prompt_tokens`、tensorは`<id>/rank-0/layer-L.pt`として置く。コードの途中を切り出しても文書単位のsplitを維持する。広く採った後段の教師データは、後ろ寄りのcutにも再利用できる。trainで学習し、validationで候補を選び、testは最終評価まで使わない。

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

生データ・補助器・実機固有の制御コードは非公開の`records/`に保持する。通常ランチャー／ハーネスの資格ゲートは変更していない。
