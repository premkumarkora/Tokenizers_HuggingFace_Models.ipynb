# Tokenizers & HuggingFace Models Colab

![GitHub Repo Size](https://img.shields.io/github/repo-size/premkumarkora/Tokenizers_HuggingFace_Models)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## Overview

This Google Colab notebook, **Tokenizers & HuggingFace Models**, demonstrates how to leverage the Hugging Face Transformers library to:

- **Load** and configure tokenizers for various pre-trained models.
- **Apply** chat-style templates for prompt engineering.
- **Quantize** and optimize model memory usage with BitsAndBytes.
- **Stream** token-by-token generation for interactive applications.
- **Integrate** end-to-end into your Git version control workflow.

Whether you are building chatbots, text generators, or experimenting with quantized inference, this notebook provides a clear, step-by-step guide.

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Setup](#setup)  
3. [Notebook Structure](#notebook-structure)  
4. [Key Sections](#key-sections)  
5. [Running the Notebook](#running-the-notebook)  
6. [Git Integration](#git-integration)  
7. [Troubleshooting](#troubleshooting)  
8. [License](#license)  

## Prerequisites

- A **Google account** for Colab.  
- **Git** installed locally (optional, for version control).  
- Basic familiarity with Python and Jupyter notebooks.

## Setup

1. **Clone this repository** (if you haven’t already):
   ```bash
   git clone https://github.com/premkumarkora/Tokenizers_HuggingFace_Models.git
   cd Tokenizers_HuggingFace_Models
   ```
2. **Open in Colab**:  
   - Visit [Google Colab](https://colab.research.google.com).  
   - Select **File → Open notebook → GitHub**.  
   - Paste the repository URL and open `Tokenizers_HuggingFace_Models.ipynb`.
3. **Install dependencies** (in the first notebook cell):
   ```bash
   !pip install -q transformers torch bitsandbytes sentencepiece accelerate
   ```

## Notebook Structure

- **Section 1: Environment Setup**  
  Installs required packages and configures GPU/quantization settings.
- **Section 2: Tokenizer Loading**  
  Demonstrates loading different tokenizers and configuring chat templates.
- **Section 3: Model Quantization**  
  Shows how to define `BitsAndBytesConfig` for optimal 4-bit NF4 quantization.
- **Section 4: Generation Function**  
  Implements a reusable `generate()` function for streaming inference.
- **Section 5: Memory Profiling**  
  Explains how to measure the model’s GPU memory footprint.
- **Section 6: Example Runs**  
  Runs sample prompts for chat, translation, and summarization.
- **Section 7: Git & Colab Integration**  
  Provides commands for syncing notebook changes back to GitHub.

## Key Sections

### Quantization Config
```python
from bitsandbytes import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```

### Generate Function
```python
def generate(model_name, messages):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to("cuda")
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=quant_config
    )
    model.generate(inputs, max_new_tokens=80, streamer=streamer)
```

### Memory Footprint Example
```python
memory_mb = model.get_memory_footprint() / 1e6
print(f"Memory footprint: {memory_mb:,.1f} MB")
```

## Running the Notebook

1. **Select GPU**: Runtime → Change runtime type → GPU.  
2. **Run all cells**: Runtime → Run all.  
3. **Interact**: Modify prompts or model names, then re-run relevant cells.

## Git Integration

```bash
!git config --global user.name "Your Name"
!git config --global user.email "you@example.com"
!git add Tokenizers_HuggingFace_Models.ipynb
!git commit -m "Update notebook"
!git push origin main
```

## Troubleshooting

- **TOKENIZERS_PARALLELISM warning**: Add at top of notebook:
  ```python
  import os
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  ```
- **CUDA OOM**: Lower `max_new_tokens` or offload layers via `device_map`.

## License

This notebook and associated code are released under the [MIT License](LICENSE).
