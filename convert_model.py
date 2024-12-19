import argparse
import torch
import os
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import shutil
from huggingface_hub import HfApi, Repository

def convert_model(model_name_or_path, output_dir, opset_version=14):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # モデルとトークナイザーのロード
    model = AutoModel.from_pretrained(model_name_or_path, force_download=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, force_download=True)
    model.eval()

    # ダミー入力の作成
    sample_text = "Hello, world!"
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    input_names = list(inputs.keys())
    output_names = ['output']

    # ONNXへのエクスポートの準備
    output_dir = Path(output_dir)
    onnx_models_dir = output_dir / "onnx"
    onnx_models_dir.mkdir(parents=True, exist_ok=True)

    model_onnx_path = onnx_models_dir / "model.onnx"

    # モデルのONNXへのエクスポート
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

    print(f"モデルをONNX形式に変換しました：{model_onnx_path}")

    # 量子化処理の実行
    quantize_model(model_onnx_path, onnx_models_dir)

    # READMEファイルの作成
    create_readme(output_dir, model_name_or_path)

def quantize_model(model_onnx_path, onnx_models_dir):
    print("量子化のための準備をしています...")

    # FP16量子化
    fp16_model_path = onnx_models_dir / "model_fp16.onnx"
    from onnxconverter_common import float16
    import onnx

    # モデルのロード
    model_fp32 = onnx.load_model(str(model_onnx_path))

    # FP16への変換
    model_fp16 = float16.convert_float_to_float16(model_fp32)

    # モデルの保存
    onnx.save_model(model_fp16, str(fp16_model_path))
    print(f"FP16量子化を完了しました：{fp16_model_path}")

    # INT8量子化
    int8_model_path = onnx_models_dir / "model_int8.onnx"
    quantize_dynamic(
        model_input=str(model_onnx_path),
        model_output=str(int8_model_path),
        weight_type=QuantType.QInt8
    )
    print(f"INT8量子化を完了しました：{int8_model_path}")

    # UINT8量子化
    uint8_model_path = onnx_models_dir / "model_uint8.onnx"
    quantize_dynamic(
        model_input=str(model_onnx_path),
        model_output=str(uint8_model_path),
        weight_type=QuantType.QUInt8
    )
    print(f"UINT8量子化を完了しました：{uint8_model_path}")

def create_readme(output_dir, model_name_or_path):
    readme_path = Path(output_dir) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# {model_name_or_path}のONNXモデルと量子化モデル\n")
        f.write("\n")
        f.write("このフォルダには、以下のONNXモデルが含まれています。\n")
        f.write("- onnx/model.onnx: 元のONNXモデル\n")
        f.write("- onnx/model_fp16.onnx: FP16による量子化モデル\n")
        f.write("- onnx/model_int8.onnx: INT8による量子化モデル\n")
        f.write("- onnx/model_uint8.onnx: UINT8による量子化モデル\n")
    print(f"README.mdを作成しました：{readme_path}")

def upload_to_huggingface(output_dir, model_name_or_path, hf_token):
    api = HfApi()
    user = api.whoami(token=hf_token)["name"]
    repo_name = f"{user}/{model_name_or_path.replace('/', '_')}_onnx"
    repo_url = api.create_repo(name=repo_name, token=hf_token, exist_ok=True)
    print(f"モデルをアップロードしています：{repo_url}")

    repo = Repository(local_dir=output_dir, clone_from=repo_url, use_auth_token=hf_token)
    repo.git_add()
    repo.git_commit("Add ONNX model and quantized models")
    repo.git_push()
    print(f"モデルのアップロードが完了しました：{repo_url}")

def main():
    parser = argparse.ArgumentParser(description='Hugging FaceモデルをONNX形式に変換し、量子化するスクリプト')
    parser.add_argument('--model', type=str, required=True, help='Hugging Face上のモデル名を指定')
    parser.add_argument('--output_dir', type=str, default=None, help='出力ディレクトリ名（デフォルトはモデル名）')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version（デフォルトは14）')
    parser.add_argument('--hf_token', type=str, default='', help='Hugging Faceのアクセストークン。モデルをアップロードする場合に必要です')
    args = parser.parse_args()

    # 出力ディレクトリ名の設定
    if args.output_dir is None:
        sanitized_model_name = args.model.replace('/', '_')
        args.output_dir = sanitized_model_name

    # モデルの変換と量子化の実行
    convert_model(args.model, args.output_dir, args.opset)

    # モデルのアップロード（アクセストークンが指定された場合）
    if args.hf_token:
        upload_to_huggingface(args.output_dir, args.model, args.hf_token)

if __name__ == '__main__':
    main()