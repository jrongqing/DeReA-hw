# DeReA

[中文](README.md) | **English**

DeReA is a translation enhancement system for unconventional expressions, including slang, idioms, proverbs, metaphors, and other expressions whose intended meanings differ from their literal meanings. It improves translation by combining expression detection, knowledge-base retrieval, and retrieval-augmented translation.

## Overview

The main DeReA pipeline has three stages:

1. **Detect**: identify whether an English sentence contains unconventional expressions and extract candidate expressions.
2. **Retrieval**: retrieve relevant explanations from `data/udKB.json` with an embedding model and a FAISS index.
3. **Translate**: generate the final translation using the source sentence, detection analysis, RAG knowledge, and direct translation.

The repository also includes training scripts, evaluation scripts, and a web demo.

## Repository Structure

```text
DeReA-hw/
|-- benchmark/
|   `-- emerging_slang.json
|-- data/
|   |-- LoML.json
|   `-- udKB.json
|-- demo/
|   `-- app.py
|-- detect/
|   |-- infer/
|   |   `-- detect_infer.py
|   |-- data_construct/
|   `-- train/
|       |-- sft/
|       `-- dpo/
|-- embedding/
|   |-- data/
|   |-- train/
|   `-- retrieve/
|-- eval/
|   |-- detect/
|   |-- retrieval/
|   |-- translate/
|   |-- metric/
|   `-- eval_infer.py
|-- model/
|   |-- base/
|   |-- detect/
|   `-- retrieval/
`-- translate/
    `-- main.py
```

## Data and Models

### Data Files

| Path | Description |
| :--- | :--- |
| `data/LoML.json` | Default input data used by the demo and scripts |
| `data/udKB.json` | Knowledge base for unconventional expressions |
| `benchmark/emerging_slang.json` | The currently retained benchmark file |
| `embedding/data/pie_train_data_cleaned.json` | Training data for the embedding model |

### Model Paths

| Purpose | Default path |
| :--- | :--- |
| Detection model | `model/detect/qwen_dpo` |
| Translation model | `model/base/Qwen3-8B` |
| Retrieval model | `model/retrieval/qwen3-0.6b-finetuned-new/final/final` |
| FAISS index | `model/retrieval/index/udKB_index_0.6b.faiss` |

Large model weights are usually not committed to Git. Before running the project, make sure the required models and indexes are available under `model/`.

## Installation

Python 3.10 or later is recommended.

```bash
pip install torch transformers vllm sentence-transformers faiss-cpu tqdm numpy datasets
```

For GPU FAISS or automatic evaluation:

```bash
pip install faiss-gpu unbabel-comet openai
```

Detection model training requires LLaMA-Factory:

```bash
pip install llamafactory
```

## Quick Start: Web Demo

The recommended entry point is `demo/app.py`.

```bash
python demo/app.py
```

Then open:

```text
http://127.0.0.1:7860
```

The demo runs:

1. Detect: loads `model/detect/qwen_dpo` and writes detection outputs.
2. Retrieval: loads the retrieval model, FAISS index, and `data/udKB.json`.
3. Translate: loads `model/base/Qwen3-8B` and writes final translations.

Default inputs and paths:

| Parameter | Default value |
| :--- | :--- |
| Input JSON Path | `data/LoML.json` |
| Output Directory | `demo/outputs` |
| Detect Model | `model/detect/qwen_dpo` |
| Translate Model | `model/base/Qwen3-8B` |
| Retrieval Model | `model/retrieval/qwen3-0.6b-finetuned-new/final/final` |
| FAISS Index | `model/retrieval/index/udKB_index_0.6b.faiss` |
| Idiom Dictionary JSON | `data/udKB.json` |
| Max Items | `5` |
| Preview Items | `3` |

After a run, the demo saves:

```text
demo/outputs/
|-- 01_detect_result.json
|-- 02_retrieval.json
`-- 03_translate_result.json
```

## Input and Output Format

The input file should be a JSON list. Each item must contain at least `sentence`:

```json
[
  {
    "sentence": "When the share market crashed his fingers were burnt.",
    "direct_8b": "股市崩盘时，他的手指被烧伤了。",
    "label": "Figurative",
    "reference_zh": "股市崩盘时，他因投资失误遭受了损失。"
  }
]
```

Common input fields:

| Field | Required | Description |
| :--- | :--- | :--- |
| `sentence` | Yes | Original English sentence |
| `direct_8b` / `direct` | No | Direct translation used as Translation 0 |
| `label` | No | Detection label, commonly `Figurative` or `Without_Idiom` |
| `reference_zh` | No | Chinese reference translation for evaluation |

Pipeline outputs add the following fields:

| Field | Stage | Description |
| :--- | :--- | :--- |
| `detect_label` | Detect | Whether an unconventional expression is detected |
| `dpo_eval_8B` | Detect | Raw detection analysis from the detection model |
| `detect_idiom` | Detect | Parsed list of detected expressions |
| `rag_info` | Retrieval | Retrieved knowledge-base entries |
| `retrieval_top3` | Retrieval | Top-3 retrieval matches for each detected expression |
| `cot_result` | Translate | Final translation output |

## Scripted Evaluation Pipeline

You can also run the scripts under `eval/` stage by stage.

### 1. Detect

```bash
CUDA_VISIBLE_DEVICES=0 python eval/detect/detect.py
```

Default configuration:

```python
input_file_path = "data/LoML.json"
output_file_path = "eval/detect/result/LoML-result.json"
model_path = "model/detect/qwen_dpo"
```

The script writes detection labels and detection analysis. If the input contains `label`, it also reports Precision, Recall, and F1.

### 2. Retrieval

```bash
python eval/retrieval/faiss_load.py
```

Default assets:

```text
model/retrieval/qwen3-0.6b-finetuned-new/final
model/retrieval/index/udKB_index_0.6b.faiss
data/udKB.json
```

The script reads `detect_idiom` from detection outputs and writes `rag_info`.

### 3. Translate

```bash
python eval/translate/translate.py
```

Default configuration:

```python
input_file_path = "data/LoML.json"
output_file_path = "eval/translate/Cot_eval_new.json"
model_path = "model/base/Qwen3-8B"
```

Translation behavior:

1. If no unconventional expression is detected, translate directly.
2. If detected expressions and `rag_info` are available, use RAG knowledge and detection analysis for enhanced translation.
3. If expressions are detected but retrieval results are missing, use a fallback prompt that asks the model to explain and translate.

### 4. Multi-strategy Inference

`eval/eval_infer.py` compares multiple translation strategies:

| Output file | Field | Description |
| :--- | :--- | :--- |
| `direct.json` | `direct` | Direct translation |
| `slangdit.json` | `slangdit` | Detect and explain unconventional expressions before translation |
| `srag.json` | `srag` | Translation enhanced only with retrieved knowledge |
| `cot.json` | `cot` | DeReA-style integrated reasoning translation |

Before running it, update the input, output, model, and target-language settings at the top of the script.

## Metrics

### COMET

```bash
python eval/metric/comet/main.py
```

or:

```bash
python eval/metric/comet/main_ud.py
```

Before running, update:

```python
input_file_path = "..."
output_file = "..."
SOURCE_KEY = "sentence"
TARGET_KEY = "cot_result"
REFERENCE_KEY = "reference_zh"
```

The scripts report COMET scores for all samples, figurative samples, and non-idiomatic samples.

### LLM-as-Judge

```bash
python eval/metric/LAJ/claude_all.py
```

Configure an OpenAI-compatible API before running:

```python
client = OpenAI(
    api_key="...",
    base_url="..."
)
```

This evaluator focuses on:

1. whether the intended meaning of the unconventional expression is translated correctly;
2. whether the expression is naturally integrated into the target-language context.

## Detection Model Training

### SFT

Configuration:

```text
detect/train/sft/sft/qwen3_full_sft.yaml
```

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train detect/train/sft/sft/qwen3_full_sft.yaml
```

Dataset registry:

```text
detect/train/sft/data/dataset_info.json
```

### DPO

Configuration:

```text
detect/train/dpo/dpo/qwen3_lora_dpo.yaml
```

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train detect/train/dpo/dpo/qwen3_lora_dpo.yaml
```

Dataset registry:

```text
detect/train/dpo/data/dataset_info.json
```

After training, place the final model under:

```text
model/detect/qwen_dpo
```

## Retrieval Model and FAISS Index

### Data Processing

```bash
python embedding/train/data.py
```

This script converts training data into the `anchor` / `positive` format used by SentenceTransformer and saves it as a HuggingFace Dataset.

### Train the Embedding Model

```bash
python embedding/train/train.py
```

After training, save or copy the model to:

```text
model/retrieval/qwen3-0.6b-finetuned-new/final/final
```

### Build the FAISS Index

```bash
python embedding/retrieve/faiss_set.py
```

The recommended output path is:

```text
model/retrieval/index/udKB_index_0.6b.faiss
```

The FAISS index must use the same item order as `data/udKB.json`.

## Recommended Workflow

Quick demo:

```bash
python demo/app.py
```

Batch experiments:

```bash
python eval/detect/detect.py
python eval/retrieval/faiss_load.py
python eval/translate/translate.py
```

Evaluation:

```bash
python eval/metric/comet/main.py
python eval/metric/LAJ/claude_all.py
```

## Notes

* Project paths are relative. Run commands from the repository root.
* `benchmark/` currently keeps only `emerging_slang.json`.
* Large models and index files must be placed manually under `model/`.
* If GPU memory is limited, reduce `Max Items` in the demo or switch to smaller models.
* On Windows PowerShell, running commands from the repository root is recommended to avoid relative-path issues.

