# Model Conversion and Quantization Script

This repository contains a Python script to convert Hugging Face models to ONNX and ORT formats, perform quantization, and generate README files for the converted models. The script automates the process of optimizing models for deployment, making it easier to use models in different environments.

[日本語版READMEはこちら](README_ja.md)

## Features

- **Model Conversion**: Convert Hugging Face models to ONNX and ORT formats.
- **Model Optimization**: Optimize the ONNX models for better performance.
- **Quantization**: Perform FP16, INT8, and UINT8 quantization on the models.
- **README Generation**: Automatically generate English and Japanese README files for the converted models.
- **Hugging Face Integration**: Optionally upload the converted models to Hugging Face Hub.

## Requirements

- Python 3.11 or higher
- Install required packages using `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

Alternatively, you can install the packages individually:

```bash
pip install torch transformers onnx onnxruntime onnxconverter-common onnxruntime-tools onnxruntime-transformers huggingface_hub
```

## Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/model_conversion.git
   cd model_conversion
   ```

2. **Install Dependencies**

   Ensure that you have **Python 3.11 or higher** installed. Install the required packages using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Conversion Script**

   The script `convert_model.py` converts and quantizes the model.

   ```bash
   python convert_model.py --model_name_or_path your-model-name --output_dir output_directory
   ```

   - Replace `your-model-name` with the name or path of the Hugging Face model you want to convert.
   - The `--output_dir` argument specifies the output directory. If not provided, it defaults to the model name.

   **Example:**

   ```bash
   python convert_model.py --model_name_or_path bert-base-uncased --output_dir bert_onnx
   ```

4. **Upload to Hugging Face (Optional)**

   To upload the converted models to Hugging Face Hub, add the `--upload` flag.

   ```bash
   python convert_model.py --model_name_or_path your-model-name --output_dir output_directory --upload
   ```

   Make sure you are logged in to Hugging Face CLI:

   ```bash
   huggingface-cli login
   ```

## Script Overview

### `convert_model.py`

This script performs the following steps:

- Loads the specified Hugging Face model and tokenizer.
- Exports the model to ONNX format with dynamic axes.
- Optimizes the ONNX model using `onnxruntime-tools`.
- Performs quantization (FP16, INT8, UINT8) on the optimized model.
- Converts the optimized and quantized models to ORT format.
- Generates README files (`README.md` and `README_ja.md`) with sample usage code.
- Optionally uploads the models and README files to Hugging Face Hub.

### Key Functions

- `convert_model(model_name_or_path, output_dir, opset_version)`: Main function to convert and optimize the model.
- `optimize_model(onnx_model_path, optimized_model_path, opset_version, model_type, model_config)`: Optimizes the ONNX model.
- `quantize_model(optimized_model_path, onnx_models_dir, ort_models_dir)`: Performs quantization on the optimized model.
- `convert_to_ort(model_onnx_path, ort_model_path)`: Converts an ONNX model to ORT format.
- `upload_to_huggingface(output_dir, model_name_or_path)`: Uploads the models and README files to Hugging Face Hub.
- `main()`: Parses command-line arguments and initiates the conversion process.

### Code Structure

- The code is modularized for better readability and maintenance.
- The `ReadmeGenerator` class in `readme_generator.py` handles the creation of README files in both English and Japanese.
- The script uses `argparse` for command-line argument parsing.

## Example Usage

After running the conversion script, you can use the converted models as shown below:

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('your-model-name')

# Prepare inputs
text = 'Replace this text with your input.'
inputs = tokenizer(text, return_tensors='np')

# Specify the model paths
# Test both the ONNX model and the ORT model
model_paths = [
    'onnx_models/model_opt.onnx',    # ONNX model
    'ort_models/model.ort'           # ORT format model
]

# Run inference with each model
for model_path in model_paths:
    print(f'\n===== Using model: {model_path} =====')
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
        print(f'Output {idx} shape: {output.shape}')

    # Display the results (add further processing if needed)
    print(outputs)
```

## Notes

- Ensure that your ONNX Runtime version is **1.15.0 or higher** to use ORT format models.
- Adjust the `providers` parameter based on your hardware (e.g., `'CUDAExecutionProvider'` for GPUs).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contribution

Contributions are welcome! Please open an issue or submit a pull request for improvements.