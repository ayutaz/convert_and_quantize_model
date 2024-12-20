import argparse
import torch
import os
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import onnx
import onnxruntime
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
import shutil
from huggingface_hub import HfApi, Repository, HfFolder

def convert_model(model_name_or_path, output_dir, opset_version=14):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model and tokenizer
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.eval()
    model_config = model.config
    model_type = model_config.model_type  # Get the model type

    # Create dummy input
    sample_text = "Hello, world!"
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    input_names = list(inputs.keys())
    output_names = ['output']

    # Prepare output directories
    output_dir = Path(output_dir)
    onnx_models_dir = output_dir / "onnx_models"
    ort_models_dir = output_dir / "ort_models"
    onnx_models_dir.mkdir(parents=True, exist_ok=True)
    ort_models_dir.mkdir(parents=True, exist_ok=True)

    model_onnx_path = onnx_models_dir / "model.onnx"

    # Export the model to ONNX
    torch.onnx.export(
        model,
        args=tuple(inputs.values()),
        f=str(model_onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: 'batch_size'} for name in input_names},
        opset_version=opset_version,
        do_constant_folding=True
    )

    print(f"Model converted to ONNX format at: {model_onnx_path}")

    # Optimize the model
    optimized_model_path = onnx_models_dir / "model_opt.onnx"
    optimize_model(model_onnx_path, optimized_model_path, opset_version, model_type, model_config)

    # Perform quantization
    quantize_model(optimized_model_path, onnx_models_dir, ort_models_dir)

    # Create both English and Japanese README files
    create_readme(output_dir, model_name_or_path)

def optimize_model(onnx_model_path, optimized_model_path, opset_version, model_type, model_config):
    print("Optimizing the model...")

    # Set optimization options
    optimization_options = FusionOptions(model_type)

    # Get num_heads and hidden_size from model config, default to 0 if not available
    num_heads = getattr(model_config, 'num_attention_heads', 0)
    hidden_size = getattr(model_config, 'hidden_size', 0)

    # Call optimizer.optimize_model
    opt_model = optimizer.optimize_model(
        input=str(onnx_model_path),
        model_type=model_type,
        num_heads=num_heads,
        hidden_size=hidden_size,
        optimization_options=optimization_options,
        use_gpu=False,
        opt_level=1,
        only_onnxruntime=True,
    )
    opt_model.save_model_to_file(str(optimized_model_path))
    print(f"Model optimized and saved at: {optimized_model_path}")

def quantize_model(optimized_model_path, onnx_models_dir, ort_models_dir):
    print("Preparing for quantization...")

    # FP16 quantization
    from onnxconverter_common import float16
    fp16_model_path = onnx_models_dir / "model_fp16.onnx"

    # Load the model
    model_fp32 = onnx.load_model(str(optimized_model_path))

    # Convert to FP16
    model_fp16 = float16.convert_float_to_float16(model_fp32)

    # Save the model
    onnx.save_model(model_fp16, str(fp16_model_path))
    print(f"FP16 quantization completed: {fp16_model_path}")

    # INT8 quantization
    int8_model_path = onnx_models_dir / "model_int8.onnx"
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(optimized_model_path),
        model_output=str(int8_model_path),
        weight_type=QuantType.QInt8
    )
    print(f"INT8 quantization completed: {int8_model_path}")

    # UINT8 quantization
    uint8_model_path = onnx_models_dir / "model_uint8.onnx"
    quantize_dynamic(
        model_input=str(optimized_model_path),
        model_output=str(uint8_model_path),
        weight_type=QuantType.QUInt8
    )
    print(f"UINT8 quantization completed: {uint8_model_path}")

    # Convert to ORT format
    print("Converting to ORT format...")

    # Convert optimized model to ORT
    model_ort_path = ort_models_dir / "model.ort"
    convert_to_ort(optimized_model_path, model_ort_path)

    # Convert FP16 model to ORT
    model_fp16_ort_path = ort_models_dir / "model_fp16.ort"
    convert_to_ort(fp16_model_path, model_fp16_ort_path)

    # Convert INT8 model to ORT
    model_int8_ort_path = ort_models_dir / "model_int8.ort"
    convert_to_ort(int8_model_path, model_int8_ort_path)

    # Convert UINT8 model to ORT
    model_uint8_ort_path = ort_models_dir / "model_uint8.ort"
    convert_to_ort(uint8_model_path, model_uint8_ort_path)

def convert_to_ort(model_onnx_path, ort_model_path):
    import onnxruntime as ort
    # Create session options
    sess_options = ort.SessionOptions()
    # Set optimization level
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # Specify optimized model file path
    sess_options.optimized_model_filepath = str(ort_model_path)
    # Create session to optimize and save the model
    _ = ort.InferenceSession(str(model_onnx_path), sess_options)
    print(f"Converted to ORT format: {ort_model_path}")

def create_readme(output_dir, model_name_or_path):
    from huggingface_hub import HfApi

    # Create HfApi instance
    api = HfApi()

    # Get model info
    try:
        model_info = api.model_info(repo_id=model_name_or_path)
        license_info = model_info.license if model_info.license else "apache-2.0"
    except Exception as e:
        print(f"Failed to get model info. Using default license. Error: {e}")
        license_info = "apache-2.0"

    # Create original model URL
    original_model_url = f"https://huggingface.co/{model_name_or_path}"

    # Generate sanitized model name for file names
    if '/' in model_name_or_path:
        repo_model_name = model_name_or_path.split('/', 1)[1]
    else:
        repo_model_name = model_name_or_path

    # English README.md
    readme_path = Path(output_dir) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        # Add header
        f.write("---\n")
        f.write(f"license: {license_info}\n")
        f.write("tags:\n")
        f.write("- onnx\n")
        f.write("- ort\n")
        f.write("---\n\n")
        f.write(f"# ONNX and ORT models with quantization of {model_name_or_path}\n")
        f.write("\n")
        # Link to Japanese README
        f.write("[日本語READMEはこちら](README_ja.md)\n")
        f.write("\n")
        # Original model link
        f.write(f"This repository contains the ONNX and ORT formats of the model [{model_name_or_path}]({original_model_url}), along with quantized versions.\n")
        f.write("\n")
        f.write("## License\n")
        f.write(f"The license for this model is \"{license_info}\". For details, please refer to the original model page: [{model_name_or_path}]({original_model_url}).\n")
        f.write("\n")
        f.write("## Usage\n")
        f.write("To use this model, install ONNX Runtime and perform inference as shown below.\n")
        f.write("```python\n")
        f.write("# Example code\n")
        f.write("import onnxruntime as ort\n")
        f.write("session = ort.InferenceSession('path_to_model.onnx')\n")
        f.write("# Prepare inputs and run inference\n")
        f.write("```\n")
        f.write("\n")
        f.write("## Contents of the Model\n")
        f.write("This repository includes the following models:\n")
        f.write("\n")
        f.write("### ONNX Models\n")
        f.write("- `onnx_models/model.onnx`: Original ONNX model\n")
        f.write("- `onnx_models/model_opt.onnx`: Optimized ONNX model\n")
        f.write("- `onnx_models/model_fp16.onnx`: FP16 quantized model\n")
        f.write("- `onnx_models/model_int8.onnx`: INT8 quantized model\n")
        f.write("- `onnx_models/model_uint8.onnx`: UINT8 quantized model\n")
        f.write("\n")
        f.write("### ORT Models\n")
        f.write("- `ort_models/model.ort`: ORT model using the optimized ONNX model\n")
        f.write("- `ort_models/model_fp16.ort`: ORT model using the FP16 quantized model\n")
        f.write("- `ort_models/model_int8.ort`: ORT model using the INT8 quantized model\n")
        f.write("- `ort_models/model_uint8.ort`: ORT model using the UINT8 quantized model\n")
        f.write("\n")
        f.write("## Notes\n")
        f.write(f"Please adhere to the license and usage conditions of the original model [{model_name_or_path}]({original_model_url}).\n")
        f.write("\n")
        f.write("## Contribution\n")
        f.write("If you find any issues or have improvements, please create an issue or submit a pull request.\n")
    print(f"English README.md created at: {readme_path}")

    # Japanese README_ja.md
    readme_ja_path = Path(output_dir) / "README_ja.md"
    with open(readme_ja_path, "w", encoding="utf-8") as f:
        # Add header
        f.write("---\n")
        f.write(f"license: {license_info}\n")
        f.write("tags:\n")
        f.write("- onnx\n")
        f.write("- ort\n")
        f.write("---\n\n")
        f.write(f"# {model_name_or_path}のONNXおよびORTモデルと量子化モデル\n")
        f.write("\n")
        # Link to English README
        f.write("[Click here for the English README](README.md)\n")
        f.write("\n")
        # Original model link
        f.write(f"このリポジトリは、元のモデル [{model_name_or_path}]({original_model_url}) をONNXおよびORT形式に変換し、さらに量子化したものです。\n")
        f.write("\n")
        f.write("## ライセンス\n")
        f.write(f"このモデルのライセンスは「{license_info}」です。詳細は元のモデルページ（[{model_name_or_path}]({original_model_url})）を参照してください。\n")
        f.write("\n")
        f.write("## 使い方\n")
        f.write("このモデルを使用するには、ONNX Runtimeをインストールし、以下のように推論を行います。\n")
        f.write("```python\n")
        f.write("# コード例\n")
        f.write("import onnxruntime as ort\n")
        f.write("session = ort.InferenceSession('path_to_model.onnx')\n")
        f.write("# 入力の準備と推論の実行\n")
        f.write("```\n")
        f.write("\n")
        f.write("## モデルの内容\n")
        f.write("このリポジトリには、以下のモデルが含まれています。\n")
        f.write("\n")
        f.write("### ONNXモデル\n")
        f.write("- `onnx_models/model.onnx`: 元のONNXモデル\n")
        f.write("- `onnx_models/model_opt.onnx`: 最適化されたONNXモデル\n")
        f.write("- `onnx_models/model_fp16.onnx`: FP16による量子化モデル\n")
        f.write("- `onnx_models/model_int8.onnx`: INT8による量子化モデル\n")
        f.write("- `onnx_models/model_uint8.onnx`: UINT8による量子化モデル\n")
        f.write("\n")
        f.write("### ORTモデル\n")
        f.write("- `ort_models/model.ort`: 最適化されたONNXモデルを使用したORTモデル\n")
        f.write("- `ort_models/model_fp16.ort`: FP16量子化モデルを使用したORTモデル\n")
        f.write("- `ort_models/model_int8.ort`: INT8量子化モデルを使用したORTモデル\n")
        f.write("- `ort_models/model_uint8.ort`: UINT8量子化モデルを使用したORTモデル\n")
        f.write("\n")
        f.write("## 注意事項\n")
        f.write(f"元のモデル [{model_name_or_path}]({original_model_url}) のライセンスおよび使用条件を遵守してください。\n")
        f.write("\n")
        f.write("## 貢献\n")
        f.write("問題や改善点があれば、Issueを作成するかプルリクエストを送ってください。\n")
    print(f"Japanese README_ja.md created at: {readme_ja_path}")

def upload_to_huggingface(output_dir, model_name_or_path):
    # Import necessary modules
    from huggingface_hub import create_repo, upload_folder, whoami, get_full_repo_name
    import urllib.parse
    from huggingface_hub.utils import HfHubHTTPError

    # Get the logged-in token
    token = HfFolder.get_token()
    if token is None:
        print("Hugging Face token not found. Please login using 'huggingface-cli login'.")
        return

    # Create repository name
    if '/' in model_name_or_path:
        repo_model_name = model_name_or_path.split('/', 1)[1]
    else:
        repo_model_name = model_name_or_path

    safe_model_name = repo_model_name.replace('/', '_')  # Replace slashes with underscores
    repo_name = f"{safe_model_name}-ONNX-ORT"
    full_repo_name = f"ort-community/{repo_name}"

    # Create repository (if it doesn't exist)
    try:
        create_repo(repo_id=full_repo_name, exist_ok=True, token=token)
    except HfHubHTTPError as e:
        print(f"Failed to create repository: {e}")
        return

    # Upload the model
    print(f"Uploading the model to: https://huggingface.co/{full_repo_name}")
    upload_folder(
        folder_path=str(output_dir),
        path_in_repo="",
        repo_id=full_repo_name,
        token=token,
        commit_message="Add ONNX and ORT models with quantization",
    )
    print(f"Model upload completed: https://huggingface.co/{full_repo_name}")

def main():
    parser = argparse.ArgumentParser(description='Script to convert Hugging Face models to ONNX and ORT formats, quantize them, and upload to Hugging Face.')
    parser.add_argument('--model', type=str, required=True, help='Specify the model name on Hugging Face')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name (default is the model name)')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version (default is 14)')
    parser.add_argument('--upload', action='store_true', help='Upload the model to Hugging Face if specified')
    args = parser.parse_args()

    # Set output directory name
    if args.output_dir is None:
        if '/' in args.model:
            sanitized_model_name = args.model.split('/', 1)[1].replace('/', '_')
        else:
            sanitized_model_name = args.model.replace('/', '_')
        args.output_dir = sanitized_model_name

    # Run conversion and quantization
    convert_model(args.model, args.output_dir, args.opset)

    # Upload the model
    if args.upload:
        upload_to_huggingface(args.output_dir, args.model)
    else:
        print("Skipping upload to Hugging Face.")

if __name__ == '__main__':
    main()