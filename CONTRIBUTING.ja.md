# コントリビューション

[English](CONTRIBUTING.md)

本プロジェクトはベータ版のエンジニアリング作業です。主張は、実際に試験したハードウェア、イメージ、重み、精度、負荷の範囲に限定してください。

Python 3.11以降を使い、checkoutしたディレクトリで実行します。

```sh
python -m unittest discover -s tests -t . -v
ruff check glm53_setup tests tools
ruff format --check glm53_setup tests tools
python tools/check_publication.py
```

GPUの検査はCPUテストとは別です。固定版イメージを使い、実効引数、出力の完全性、数値差、失敗を記録します。検証手順は[docs/validation.ja.md](docs/validation.ja.md)にあります。

- 認証情報、実機固有の設定、モデルの重み、生ログ、非公開の実験記録をコミットしません。
- 推論中はモデル・cacheの成果物を読み取り専用に保ち、失敗した実行も残します。
- 合格結果を得るためだけに、実行時のガードや数値基準を緩めません。
- 複製・改変したコードは元の通知を保持します。本プロジェクトへの新規の貢献はApache-2.0で提出し、第三者部分にはそれぞれの通知が引き続き適用されます。
- 改変した上流ファイルは目立つ形で明示し、実際の配布範囲は[ライセンス整理](docs/licensing.ja.md)で確認します。
- [ハーネス受け入れ表](docs/harnesses.ja.md)ではZCodeとClaude Codeの結果を分けて記録し、未実行の項目を合格と記しません。
- 利用者に見える手順が変わる場合は、英語と日本語のREADMEを一緒に更新します。
