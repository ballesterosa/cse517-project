# Reproducibility Project: Modular Cross-Lingual Transfer in LLMs

**Team:** Saan Popović, Mariana Shuman, and Antonio Ballesteros

This repository contains the code for reproducing the experiments from the paper *"The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs"* (Bandarkar & Peng, 2025) for CSE 517. [https://aclanthology.org/2025.mrl-main.10/]

Our pipeline is entirely self-contained within a single Python script (`517finalproject.py`) and is engineered with fault-tolerance to run on a single NVIDIA A100 GPU.

## Overview of Implementation

To accommodate severe computational constraints while fulfilling rigorous reproducibility standards, our codebase is unified into a single executable script. By default, the script executes our **Full-Scale Swahili Reproduction** (80,000 samples) and a custom Layer-Swapping ablation study.

**🚨 Compute Requirements Warning 🚨**
Executing the default script (which runs the Full-Scale Swahili reproduction and the 50/50 layer ablation) takes approximately **50 hours on a single NVIDIA A100 GPU**.

The code to execute the Bengali and Telugu pipelines is fully implemented but commented out by default. If an evaluator uncomments these lines to run the end-to-end pipeline across all three languages, total compute time will exceed **65 hours**. The script will begin training immediately upon execution, but we do not recommend running it to completion.

## Ablation Study: Parameter Allocation in Layer-Swapping

The original paper relies heavily on "Partition [C]" for Layer-Swapping, which allocates only the top two and bottom six transformer layers to language capabilities, reserving the vast majority of the middle layers for mathematical reasoning. The authors theorize that language syntax processing is heavily concentrated at the very input and output layers of dense LLMs.

To empirically validate this structural assumption, we introduce a novel ablation testing a balanced 50/50 parameter split. We merged our separately trained Qwen 2.5 experts by allocating the bottom 14 layers strictly to language syntax and the top 14 layers to mathematical reasoning. Contrary to the authors' hypothesis, our balanced 50/50 split significantly outperformed the original Partition [C] for Swahili, achieving an Exact Match score of 18.0% compared to the 14.0% achieved by Partition [C] and the Base Model. This newly generated ablation indicates that the original authors' "edge-only" language allocation may be too restrictive for non-English reasoning in modern 7B models.

---

## Reproducibility Checklist

### 1. Dependencies & Installation

This code was developed and tested on an A100 GPU environment.
To install and run on department Linux machines, execute the following bash commands:

```bash
git clone https://github.com/ballesterosa/cse517-project.git
cd cse517-project
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers peft datasets<4.0.0 trl bitsandbytes accelerate safetensors tqdm matplotlib numpy
# log into huggingface for the datasets/models (you'll need an account with a token)
hf auth login
```

You should use a virtual environment for the python packages for this project with `python3 -m venv .venv` and `source .venv/bin/activate`.
Then you can run the `pip install ...` command.

### 2. Data Download Instructions

No manual data downloading is required. All evaluation datasets (MGSM) and SFT datasets are fetched programmatically via the HuggingFace datasets library at runtime. The script automatically handles caching to local directories to prevent redundant API calls.

### 3. Preprocessing Code & Commands

Preprocessing, tokenization (ChatML format), and dynamic left-padding for batched evaluation are handled natively within the training scripts. No separate preprocessing commands are needed.

### 4. Training Code & Commands

To execute the end-to-end training pipelines (Math Expert, Language Expert, Data-Mixing, Simultaneous SFT, and Layer-Swapping), run the script directly from the bash terminal.

```bash
python3 517finalproject.py
```

Note: To evaluate Bengali or Telugu, uncomment the respective lines in the language_configs list inside the main execution block of the script.

### 5. Evaluation Code & Commands

Evaluation is performed automatically at the end of the training script. It utilizes a highly optimized, fast-batched (Batch Size = 8) 2-shot Chain-of-Thought (CoT) inference loop to drastically reduce evaluation time.

Results are incrementally saved to a local smart_reproduction_results.json file, and a publication-ready .png chart is generated automatically upon completion.

### 6. Pretrained Model

We utilize the Qwen 2.5 (7B) base model (Qwen/Qwen2.5-7B-Instruct), loaded dynamically from HuggingFace in 4-bit quantization using BitsAndBytes.
