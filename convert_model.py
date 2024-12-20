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
    print(f"使用デバイス: {device}")

    # モデルとトークナイザーのロード
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.eval()
    model_config = model.config
    model_type = model_config.model_type  # モデルタイプを取得

    # ダミー入力の作成
    sample_text = "Hello, world!"
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    input_names = list(inputs.keys())
    output_names = ['output']

    # 出力ディレクトリの準備
    output_dir = Path(output_dir)
    onnx_models_dir = output_dir / "onnx_models"
    ort_models_dir = output_dir / "ort_models"
    onnx_models_dir.mkdir(parents=True, exist_ok=True)
    ort_models_dir.mkdir(parents=True, exist_ok=True)

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

    # モデルの最適化
    optimized_model_path = onnx_models_dir / "model_opt.onnx"
    optimize_model(model_onnx_path, optimized_model_path, opset_version, model_type, model_config)

    # 量子化処理の実行
    quantize_model(optimized_model_path, onnx_models_dir, ort_models_dir)

    # READMEファイルの作成
    create_readme(output_dir, model_name_or_path)

def optimize_model(onnx_model_path, optimized_model_path, opset_version, model_type, model_config):
    print("モデルの最適化を行っています...")

    # 最適化オプションの設定
    optimization_options = FusionOptions(model_type)

    # モデルの設定から num_heads と hidden_size を取得
    # 取得できない場合は 0 を設定
    num_heads = getattr(model_config, 'num_attention_heads', 0)
    hidden_size = getattr(model_config, 'hidden_size', 0)

    # optimizer.optimize_model 関数の呼び出し
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
    print(f"モデルを最適化しました：{optimized_model_path}")

def quantize_model(optimized_model_path, onnx_models_dir, ort_models_dir):
    print("量子化のための準備をしています...")

    # FP16量子化
    from onnxconverter_common import float16
    fp16_model_path = onnx_models_dir / "model_fp16.onnx"

    # モデルのロード
    model_fp32 = onnx.load_model(str(optimized_model_path))

    # FP16への変換
    model_fp16 = float16.convert_float_to_float16(model_fp32)

    # モデルの保存
    onnx.save_model(model_fp16, str(fp16_model_path))
    print(f"FP16量子化を完了しました：{fp16_model_path}")

    # INT8量子化
    int8_model_path = onnx_models_dir / "model_int8.onnx"
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(optimized_model_path),
        model_output=str(int8_model_path),
        weight_type=QuantType.QInt8
    )
    print(f"INT8量子化を完了しました：{int8_model_path}")

    # UINT8量子化
    uint8_model_path = onnx_models_dir / "model_uint8.onnx"
    quantize_dynamic(
        model_input=str(optimized_model_path),
        model_output=str(uint8_model_path),
        weight_type=QuantType.QUInt8
    )
    print(f"UINT8量子化を完了しました：{uint8_model_path}")

    # ORT形式への変換
    print("ORT形式への変換を行います...")

    # 元の最適化モデルのORT変換
    model_ort_path = ort_models_dir / "model.ort"
    convert_to_ort(optimized_model_path, model_ort_path)

    # FP16モデルのORT変換
    model_fp16_ort_path = ort_models_dir / "model_fp16.ort"
    convert_to_ort(fp16_model_path, model_fp16_ort_path)

    # INT8モデルのORT変換
    model_int8_ort_path = ort_models_dir / "model_int8.ort"
    convert_to_ort(int8_model_path, model_int8_ort_path)

    # UINT8モデルのORT変換
    model_uint8_ort_path = ort_models_dir / "model_uint8.ort"
    convert_to_ort(uint8_model_path, model_uint8_ort_path)

def convert_to_ort(model_onnx_path, ort_model_path):
    import onnxruntime as ort
    # セッションオプションの作成
    sess_options = ort.SessionOptions()
    # 最適化レベルの設定（ORT_ENABLE_EXTENDEDに変更）
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # 最適化されたモデルの保存先を指定
    sess_options.optimized_model_filepath = str(ort_model_path)
    # セッションを作成し、モデルを最適化して保存
    _ = ort.InferenceSession(str(model_onnx_path), sess_options)
    print(f"ORT形式への変換を完了しました：{ort_model_path}")

def create_readme(output_dir, model_name_or_path):
    from huggingface_hub import HfApi
    
    # HfApiインスタンスの作成
    api = HfApi()
    
    # モデル情報の取得
    try:
        model_info = api.model_info(repo_id=model_name_or_path)
        license_info = model_info.license if model_info.license else "apache-2.0"
    except Exception as e:
        print(f"モデル情報の取得に失敗しました。デフォルトのライセンスを使用します。エラー：{e}")
        license_info = "apache-2.0"
    
    # 元のモデルのURLを作成
    original_model_url = f"https://huggingface.co/{model_name_or_path}"
    
    readme_path = Path(output_dir) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        # READMEのヘッダー部分を追加
        f.write("---\n")
        f.write(f"license: {license_info}\n")
        f.write("tags:\n")
        f.write("- onnx\n")
        f.write("- ort\n")
        f.write("---\n\n")
        f.write(f"# {model_name_or_path}のONNXおよびORTモデルと量子化モデル\n")
        f.write("\n")
        # 元のモデルへのハイパーリンクをREADMEに追加
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
    print(f"README.mdを作成しました：{readme_path}")

def upload_to_huggingface(output_dir, model_name_or_path):
    # 必要なモジュールをインポート
    from huggingface_hub import create_repo, upload_folder, whoami,get_full_repo_name
    import urllib.parse
    from huggingface_hub.utils import HfHubHTTPError

    # ログイン済みのトークンを取得
    token = HfFolder.get_token()
    if token is None:
        print("Hugging Faceのトークンが見つかりません。'huggingface-cli login'でログインしてください。")
        return

    # リポジトリ名の作成
    if '/' in model_name_or_path:
        repo_model_name = model_name_or_path.split('/', 1)[1]
    else:
        repo_model_name = model_name_or_path

    safe_model_name = repo_model_name.replace('/', '_')  # スラッシュをアンダースコアに置換
    repo_name = f"{safe_model_name}-ONNX-ORT"
    full_repo_name = f"ort-community/{repo_name}"

    # リポジトリを作成（既存でない場合）
    try:
        create_repo(repo_id=full_repo_name, exist_ok=True, token=token)
    except HfHubHTTPError as e:
        print(f"リポジトリの作成に失敗しました: {e}")
        return

    # モデルをアップロード
    print(f"モデルをアップロードしています：https://huggingface.co/{full_repo_name}")
    upload_folder(
        folder_path=str(output_dir),
        path_in_repo="",
        repo_id=full_repo_name,
        token=token,
        commit_message="Add ONNX and ORT models with quantization",
    )
    print(f"モデルのアップロードが完了しました：https://huggingface.co/{full_repo_name}")

def main():
    parser = argparse.ArgumentParser(description='Hugging FaceモデルをONNXおよびORT形式に変換し、量子化してアップロードするスクリプト')
    parser.add_argument('--model', type=str, required=True, help='Hugging Face上のモデル名を指定')
    parser.add_argument('--output_dir', type=str, default=None, help='出力ディレクトリ名（デフォルトはモデル名）')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version（デフォルトは14）')
    args = parser.parse_args()

    # 出力ディレクトリ名の設定
    if args.output_dir is None:
        if '/' in args.model:
            sanitized_model_name = args.model.split('/', 1)[1].replace('/', '_')
        else:
            sanitized_model_name = args.model.replace('/', '_')
        args.output_dir = sanitized_model_name

    # モデルの変換と量子化の実行
    convert_model(args.model, args.output_dir, args.opset)

    # モデルのアップロード
    upload_to_huggingface(args.output_dir, args.model)

if __name__ == '__main__':
    main()