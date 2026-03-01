# Building Large Language Models From Scratch

A hands-on implementation of Large Language Models (LLMs) from scratch in PyTorch, following the approach from **Build a Large Language Model From Scratch** by Sebastian Raschka.

- **Book**: [Manning Publications](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- **Reference Code**: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

## Overview

This repository walks through building a GPT-style model step by step—from tokenization and attention to pretraining and fine-tuning—without relying on high-level LLM frameworks. The goal is to understand how transformer-based language models work internally.

## Project Structure

| Path | Description |
|------|-------------|
| `Chapter 2 - Working-with-text-data.ipynb` | Tokenization, BPE (tiktoken), text preprocessing |
| `Chapter 3 - Coding attention mechanims.ipynb` | Self-attention and multi-head attention |
| `Chapter 4 - Implementing GPT From Scratch.ipynb` | Full GPT architecture: embeddings, transformer blocks, generation |
| `Chapter 5 - Pretraining On Unlabled Data.ipynb` | Pretraining on raw text, loss curves |
| `Chapter 6 - Fine Tuning For Classification.ipynb` | Fine-tuning for sentiment/classification |
| `Chapter 7 - Fine Tuning For Instructions.ipynb` | Instruction tuning with JSON data |
| `previous_chapters.py` | Shared code (GPT model, attention, training utilities) |
| `gpt.py` | Core GPT components (attention, layer norm, feedforward) |
| `gpt_download.py` | Script to download GPT-2 weights from OpenAI |
| `chapter6/` | Classification fine-tuning data and outputs |
| `chapter7/` | Instruction-tuning data (`instruction-data.json`) |
| `Math-For-ML/` | Math foundations (e.g., linear regression) |
| `pytorch-*.ipynb` | PyTorch basics and training exercises |

## Prerequisites

- Python 3.8+
- PyTorch 2.x (CPU, CUDA, or MPS on Apple Silicon)
- Jupyter or compatible notebook environment

### Key Dependencies

```bash
pip install torch tiktoken numpy matplotlib scikit-learn tqdm
```

Or install from the provided requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Building-LLM-From-Scratch-SR
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run notebooks in order**  
   Start with Chapter 2 and work through Chapter 7. Later notebooks import helpers from `previous_chapters.py`, so run them in sequence.

4. **Optional: Download GPT-2 weights** (for Chapter 4 weight loading)
   ```bash
   python gpt_download.py
   ```
   This downloads GPT-2 checkpoints (124M, 355M, 774M, 1558M) to the `gpt2/` directory.

## Data

- `the-verdict.txt` — Sample text (Edith Wharton) for tokenization and pretraining demos
- `chapter7/instruction-data.json` — Instruction tuning examples (downloaded in Chapter 7 if missing)

## Outputs

- `chapter6/review_classifier.pth` — Trained classification model
- `saved-model/` — Checkpoints from pretraining
- `accuracy-plot.pdf`, `loss-plot.pdf` — Training metrics

## Device Support

The notebooks detect hardware automatically:

- **Apple Silicon**: MPS (Metal Performance Shaders)
- **NVIDIA**: CUDA
- **Otherwise**: CPU

## License

This project follows the approach from Sebastian Raschka's book. The original reference implementation is under [Apache License 2.0](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE).
