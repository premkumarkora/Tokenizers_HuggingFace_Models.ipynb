# Kora-2-2B-IT

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-blue.svg)](https://huggingface.co/premkumarkora/kora-2-2b-it)

**Kora-2-2B-IT** is a highly memory-efficient, instruction-tuned variant of Google’s Gemma-2 2B model.  
Quantized to 4-bit NF4 with double quantization and leveraging bfloat16 for compute, it delivers state-of-the-art text generation performance in just ~2.2 GB of GPU memory.

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Downloading the Model](#downloading-the-model)
- [Usage Example](#usage-example)
- [Quantization Configuration](#quantization-configuration)
- [Model Card](#model-card)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Key Features

- **2.61 B parameters** quantized to 4-bit NF4  
- **~2.2 GB** GPU footprint with `device_map="auto"`  
- Optimized for chatbots, summarization, Q&A, translation, and general text generation  
- Streaming-friendly inference via `bitsandbytes` and `transformers`

## Quick Start

1. **Clone the repository**  
   ```bash
   git clone https://github.com/premkumarkora/kora-2-2b-it.git
   cd kora-2-2b-it
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the example**  
   ```bash
   python example_generate.py
   ```

## Installation

Install core libraries using pip:

```bash
pip install transformers torch bitsandbytes sentencepiece accelerate huggingface_hub
```

## Downloading the Model

- **Via Hugging Face CLI**  
  ```bash
  huggingface-cli login
  huggingface-cli repo clone premkumarkora/kora-2-2b-it
  ```
- **Via Transformers API**  
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM

  tokenizer = AutoTokenizer.from_pretrained("premkumarkora/kora-2-2b-it")
  model     = AutoModelForCausalLM.from_pretrained("premkumarkora/kora-2-2b-it")
  ```

## Usage Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bitsandbytes import BitsAndBytesConfig

# Quantization setup
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4"
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("premkumarkora/kora-2-2b-it")
model     = AutoModelForCausalLM.from_pretrained(
    "premkumarkora/kora-2-2b-it",
    quantization_config=quant_cfg,
    device_map="auto"
)

# Generate text
prompt = "Translate to Shakespearean English: Hello, friend!"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=60)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Quantization Configuration

```python
from bitsandbytes import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```

- **Scheme:** NF4 (4-bit NormalFloat) with double quantization  
- **Compute dtype:** bfloat16  
- **Memory footprint:** ~2,192 MB on GPU  

## Model Card

Detailed information is available on the [Hugging Face model page](https://huggingface.co/premkumarkora/kora-2-2b-it).

## License

Licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).

## Citation

```bibtex
@misc{kora-2-2b-it,
  title        = {Kora-2-2B-IT: A 4-bit Quantized Instruction-Tuned Variant of Gemma-2},
  author       = {Google Research and premkumarkora},
  year         = {2024},
  howpublished = {\url{https://huggingface.co/premkumarkora/kora-2-2b-it}}
}
```

## Contact

For questions or feedback, please open an issue in this repository or connect via LinkedIn.
