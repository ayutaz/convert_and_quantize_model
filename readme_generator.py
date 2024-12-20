# readme_generator.py
from pathlib import Path
from huggingface_hub import HfApi

class ReadmeGenerator:
    def __init__(self, output_dir, model_name_or_path):
        self.output_dir = output_dir
        self.model_name_or_path = model_name_or_path
        self.original_model_url = f"https://huggingface.co/{self.model_name_or_path}"
        self.api = HfApi()
        self.license_info = self.get_license_info()

        # サンプルコードの定義（英語と日本語）
        self.sample_code_en = self.generate_sample_code(language='English')
        self.sample_code_ja = self.generate_sample_code(language='Japanese')

    def get_license_info(self):
        # モデル情報の取得
        try:
            model_info = self.api.model_info(repo_id=self.model_name_or_path)
            # ライセンス情報の取得
            license_info = model_info.cardData.get('license', 'apache-2.0')
        except Exception as e:
            print(f"Failed to get model info. Using default license. Error: {e}")
            license_info = "apache-2.0"
        return license_info

    def generate_sample_code(self, language='English'):
        # サンプルコードの作成
        optimized_model_filename = "model_opt.onnx"
        ort_optimized_model_filename = "model.ort"

        if language == 'English':
            sample_code = f"""# Example code
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('{self.model_name_or_path}')

# Prepare inputs
text = 'Replace this text with your input.'
inputs = tokenizer(text, return_tensors='np')

# Specify the model paths
# Test both the ONNX model and the ORT model
model_paths = [
    'onnx_models/{optimized_model_filename}',    # ONNX model
    'ort_models/{ort_optimized_model_filename}'  # ORT format model
]

# Run inference with each model
for model_path in model_paths:
    print(f'\\n===== Using model: {{model_path}} =====')
    # Get the model extension
    model_extension = os.path.splitext(model_path)[1]

    # Load the model
    if model_extension == '.ort':
        # Load the ORT format model
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    else:
        # Load the ONNX model
        session = ort.InferenceSession(model_path)

    # Run inference
    outputs = session.run(None, dict(inputs))

    # Display the output shapes
    for idx, output in enumerate(outputs):
        print(f'Output {{idx}} shape: {{output.shape}}')

    # Display the results (add further processing if needed)
    print(outputs)
"""
        else:
            sample_code = f"""# サンプルコード
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

# トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained('{self.model_name_or_path}')

# 入力の準備
text = 'ここに入力テキストを置き換えてください。'
inputs = tokenizer(text, return_tensors='np')

# 使用するモデルのパスを指定
# ONNXモデルとORTモデルの両方をテストする
model_paths = [
    'onnx_models/{optimized_model_filename}',    # ONNXモデル
    'ort_models/{ort_optimized_model_filename}'  # ORTフォーマットのモデル
]

# モデルごとに推論を実行
for model_path in model_paths:
    print(f'\\n===== Using model: {{model_path}} =====')
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
        print(f'Output {{idx}} shape: {{output.shape}}')

    # 結果の表示（必要に応じて処理を追加）
    print(outputs)
"""
        return sample_code

    def create_readme_files(self):
        # README.md と README_ja.md を作成
        self.create_readme('README.md', 'English')
        self.create_readme('README_ja.md', 'Japanese')

    def create_readme(self, filename, language):
        # 共通の情報
        optimized_model_filename = "model_opt.onnx"
        ort_optimized_model_filename = "model.ort"
        readme_path = Path(self.output_dir) / filename

        with open(readme_path, "w", encoding="utf-8") as f:
            # ヘッダーの追加
            f.write("---\n")
            f.write(f"license: {self.license_info}\n")
            f.write("tags:\n")
            f.write("- onnx\n")
            f.write("- ort\n")
            f.write("---\n\n")

            if language == 'English':
                # 英語版 README.md の内容
                f.write(f"# ONNX and ORT models with quantization of [{self.model_name_or_path}]({self.original_model_url})\n")
                f.write("\n")
                f.write("[日本語READMEはこちら](README_ja.md)\n")
                f.write("\n")
                f.write(f"This repository contains the ONNX and ORT formats of the model [{self.model_name_or_path}]({self.original_model_url}), along with quantized versions.\n")
                f.write("\n")
                f.write("## License\n")
                f.write(f"The license for this model is \"{self.license_info}\". For details, please refer to the original model page: [{self.model_name_or_path}]({self.original_model_url}).\n")
                f.write("\n")
                f.write("## Usage\n")
                f.write("To use this model, install ONNX Runtime and perform inference as shown below.\n")
                f.write("```python\n")
                f.write(self.sample_code_en)
                f.write("```\n")
                f.write("\n")
                f.write("## Contents of the Model\n")
                f.write("This repository includes the following models:\n")
                f.write("\n")
                f.write("### ONNX Models\n")
                f.write(f"- `onnx_models/model.onnx`: Original ONNX model converted from [{self.model_name_or_path}]({self.original_model_url})\n")
                f.write(f"- `onnx_models/{optimized_model_filename}`: Optimized ONNX model\n")
                f.write("- `onnx_models/model_fp16.onnx`: FP16 quantized model\n")
                f.write("- `onnx_models/model_int8.onnx`: INT8 quantized model\n")
                f.write("- `onnx_models/model_uint8.onnx`: UINT8 quantized model\n")
                f.write("\n")
                f.write("### ORT Models\n")
                f.write(f"- `ort_models/{ort_optimized_model_filename}`: ORT model using the optimized ONNX model\n")
                f.write("- `ort_models/model_fp16.ort`: ORT model using the FP16 quantized model\n")
                f.write("- `ort_models/model_int8.ort`: ORT model using the INT8 quantized model\n")
                f.write("- `ort_models/model_uint8.ort`: ORT model using the UINT8 quantized model\n")
                f.write("\n")
                f.write("## Notes\n")
                f.write(f"Please adhere to the license and usage conditions of the original model [{self.model_name_or_path}]({self.original_model_url}).\n")
                f.write("\n")
                f.write("## Contribution\n")
                f.write("If you find any issues or have improvements, please create an issue or submit a pull request.\n")
                print(f"English README.md created at: {readme_path}")
            else:
                # 日本語版 README_ja.md の内容
                f.write(f"# [{self.model_name_or_path}]({self.original_model_url}) のONNXおよびORTモデルと量子化モデル\n")
                f.write("\n")
                f.write("[Click here for the English README](README.md)\n")
                f.write("\n")
                f.write(f"このリポジトリは、元のモデル [{self.model_name_or_path}]({self.original_model_url}) をONNXおよびORT形式に変換し、さらに量子化したものです。\n")
                f.write("\n")
                f.write("## ライセンス\n")
                f.write(f"このモデルのライセンスは「{self.license_info}」です。詳細は元のモデルページ（[{self.model_name_or_path}]({self.original_model_url})）を参照してください。\n")
                f.write("\n")
                f.write("## 使い方\n")
                f.write("このモデルを使用するには、ONNX Runtimeをインストールし、以下のように推論を行います。\n")
                f.write("```python\n")
                f.write(self.sample_code_ja)
                f.write("```\n")
                f.write("\n")
                f.write("## モデルの内容\n")
                f.write("このリポジトリには、以下のモデルが含まれています。\n")
                f.write("\n")
                f.write("### ONNXモデル\n")
                f.write(f"- `onnx_models/model.onnx`: [{self.model_name_or_path}]({self.original_model_url}) から変換された元のONNXモデル\n")
                f.write(f"- `onnx_models/{optimized_model_filename}`: 最適化されたONNXモデル\n")
                f.write("- `onnx_models/model_fp16.onnx`: FP16による量子化モデル\n")
                f.write("- `onnx_models/model_int8.onnx`: INT8による量子化モデル\n")
                f.write("- `onnx_models/model_uint8.onnx`: UINT8による量子化モデル\n")
                f.write("\n")
                f.write("### ORTモデル\n")
                f.write(f"- `ort_models/{ort_optimized_model_filename}`: 最適化されたONNXモデルを使用したORTモデル\n")
                f.write("- `ort_models/model_fp16.ort`: FP16量子化モデルを使用したORTモデル\n")
                f.write("- `ort_models/model_int8.ort`: INT8量子化モデルを使用したORTモデル\n")
                f.write("- `ort_models/model_uint8.ort`: UINT8量子化モデルを使用したORTモデル\n")
                f.write("\n")
                f.write("## 注意事項\n")
                f.write(f"元のモデル [{self.model_name_or_path}]({self.original_model_url}) のライセンスおよび使用条件を遵守してください。\n")
                f.write("\n")
                f.write("## 貢献\n")
                f.write("問題や改善点があれば、Issueを作成するかプルリクエストを送ってください。\n")
                print(f"Japanese README_ja.md created at: {readme_path}")