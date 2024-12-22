# モデル変換と量子化スクリプト

このリポジトリには、Hugging FaceのモデルをONNXおよびORT形式に変換し、量子化を行い、変換されたモデルのREADMEファイルを生成するためのPythonスクリプトが含まれています。このスクリプトは、モデルの最適化プロセスを自動化し、さまざまな環境でモデルを簡単に使用できるようにします。

[Click here for the English README](README.md)

<!-- TOC -->
* [モデル変換と量子化スクリプト](#モデル変換と量子化スクリプト)
  * [特徴](#特徴)
  * [必要要件](#必要要件)
  * [使い方](#使い方)
  * [使用例](#使用例)
  * [注意事項](#注意事項)
  * [ライセンス](#ライセンス)
  * [貢献](#貢献)
<!-- TOC -->

## 特徴

- **モデル変換**: Hugging FaceのモデルをONNXおよびORT形式に変換。
- **モデル最適化**: ONNXモデルを最適化してパフォーマンスを向上。
- **量子化**: モデルに対してFP16、INT8、UINT8の量子化を実施。
- **README生成**: 変換されたモデルの英語と日本語のREADMEファイルを自動生成。
- **Hugging Face統合**: オプションで変換されたモデルをHugging Face Hubにアップロード。

## 必要要件

- Python 3.11以上
- `requirements.txt`を使用して必要なパッケージをインストールします：

  ```bash
  pip install -r requirements.txt
  ```

または、個別にパッケージをインストールします：

```bash
pip install torch transformers onnx onnxruntime onnxconverter-common onnxruntime-tools onnxruntime-transformers huggingface_hub
```

## 使い方

1. **リポジトリのクローン**

   ```bash
   git clone https://github.com/yourusername/model_conversion.git
   cd model_conversion
   ```

2. **依存関係のインストール**

   **Python 3.11以上**がインストールされていることを確認してください。`requirements.txt`を使用して必要なパッケージをインストールします：

   ```bash
   pip install -r requirements.txt
   ```

3. **変換スクリプトの実行**

   `convert_model.py` スクリプトはモデルの変換と量子化を行います。

   ```bash
   python convert_model.py --model あなたのモデル名 --output_dir 出力ディレクトリ
   ```

   - `あなたのモデル名` を変換したいHugging Faceのモデル名またはパスに置き換えてください。
   - `--output_dir` 引数は出力ディレクトリを指定します。指定しない場合、デフォルトでモデル名になります。

   **例:**

   ```bash
   python convert_model.py --model bert-base-japanese --output_dir bert_onnx
   ```

4. **Hugging Faceへのアップロード（オプション）**

   変換されたモデルをHugging Face Hubにアップロードするには、`--upload` フラグを追加します。

   ```bash
   python convert_model.py --model あなたのモデル名 --output_dir 出力ディレクトリ --upload
   ```

   Hugging Face CLIにログインしていることを確認してください。

   ```bash
   huggingface-cli login
   ```

## 使用例

変換スクリプトを実行した後、以下のように変換されたモデルを使用できます。

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

# トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained('あなたのモデル名')

# 入力の準備
text = 'ここに入力テキストを置き換えてください。'
inputs = tokenizer(text, return_tensors='np')

# 使用するモデルのパスを指定
# ONNXモデルとORTモデルの両方をテストする
model_paths = [
    'onnx_models/model_opt.onnx',    # ONNXモデル
    'ort_models/model.ort'           # ORTフォーマットのモデル
]

# モデルごとに推論を実行
for model_path in model_paths:
    print(f'\n===== Using model: {model_path} =====')
    # モデルの拡張子を取得
    model_extension = os.path.splitext(model_path)[1]

    # モデルの読み込み
    if model_extension == '.ort':
        # ORTフォーマットのモデルをロード
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    else:
        # ONNXモデルをロード
        session = ort.InferenceSession(model_path)

    # 推論の実行
    outputs = session.run(None, dict(inputs))

    # 出力の形状を表示
    for idx, output in enumerate(outputs):
        print(f'Output {idx} shape: {output.shape}')

    # 結果の表示（必要に応じて処理を追加）
    print(outputs)
```

## 注意事項

- ORT形式のモデルをPythonで使用する場合、ONNX Runtimeのバージョンが**1.15.0以上**である必要があります。
- ハードウェアに応じて、`providers` パラメータを適切に設定してください（例えば、GPUを使用する場合は `'CUDAExecutionProvider'`）。

## ライセンス

このプロジェクトは Apache License 2.0 の下でライセンスされています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 貢献

貢献は歓迎します！改善点があれば、Issueを立てるかプルリクエストを送ってください。
