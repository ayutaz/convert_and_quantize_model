import argparse
from pathlib import Path

import onnx
import torch
from huggingface_hub import HfFolder
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from transformers import AutoModel, AutoTokenizer

from readme_generator import ReadmeGenerator


def convert_model(model_name_or_path, output_dir, opset_version=20):
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

    # Export the model to ONNX with dynamic axes for batch size and sequence length
    torch.onnx.export(
        model,
        args=tuple(inputs.values()),
        f=str(model_onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: 'batch_size', 1: 'sequence_length'} for name in input_names},
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
    readme_generator = ReadmeGenerator(output_dir, model_name_or_path)
    readme_generator.create_readme_files()

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

def upload_to_huggingface(output_dir, model_name_or_path):
    # Import necessary modules
    from huggingface_hub import create_repo, upload_folder
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
    parser.add_argument('--opset', type=int, default=20, help='ONNX opset version (default is 21)')
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