# DeReA

**中文** | [English](README_EN.md)

DeReA 是一个面向非常规表达（unconventional expressions）的翻译增强系统，覆盖 slang、idiom、proverb、metaphor 等字面含义和真实语义不一致的表达。系统通过检测、知识库检索和翻译增强，让模型在处理文化性、习语性表达时得到更可靠的翻译结果。

## 项目概览

DeReA 的主流程包含三个阶段：

1. **Detect**：检测英文句子中是否存在非常规表达，并抽取候选表达。
2. **Retrieval**：使用检索模型和 FAISS 索引，从 `data/udKB.json` 中检索相关释义。
3. **Translate**：结合原句、检测分析、RAG 释义和直译结果，生成最终翻译。

项目同时提供训练、评测和可视化 Demo 入口。

## 目录结构

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

## 数据与模型

### 数据文件

| 路径 | 说明 |
| :--- | :--- |
| `data/LoML.json` | Demo 和默认脚本使用的输入数据 |
| `data/udKB.json` | 非常规表达知识库 |
| `benchmark/emerging_slang.json` | 当前保留的 benchmark 数据 |
| `embedding/data/pie_train_data_cleaned.json` | embedding 训练数据 |

### 模型目录

| 用途 | 默认路径 |
| :--- | :--- |
| 检测模型 | `model/detect/qwen_dpo` |
| 翻译模型 | `model/base/Qwen3-8B` |
| 检索模型 | `model/retrieval/qwen3-0.6b-finetuned-new/final/final` |
| FAISS 索引 | `model/retrieval/index/udKB_index_0.6b.faiss` |

模型权重文件较大，通常不会直接提交到 Git。运行前请确认本地 `model/` 目录下已经放置对应模型和索引文件。

## 环境依赖

推荐使用 Python 3.10 或以上版本。

```bash
pip install torch transformers vllm sentence-transformers faiss-cpu tqdm numpy datasets
```

如果需要 GPU 版 FAISS 或自动评测：

```bash
pip install faiss-gpu unbabel-comet openai
```

检测模型训练依赖 LLaMA-Factory：

```bash
pip install llamafactory
```

## 快速开始：Web Demo

推荐优先使用 `demo/app.py` 运行完整流程。

```bash
python demo/app.py
```

启动后访问：

```text
http://127.0.0.1:7860
```

Demo 页面会依次执行：

1. Detect：加载 `model/detect/qwen_dpo`，输出检测结果；
2. Retrieval：加载检索模型、FAISS 索引和 `data/udKB.json`，输出 RAG 释义；
3. Translate：加载 `model/base/Qwen3-8B`，输出最终翻译。

默认输入输出：

| 参数 | 默认值 |
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

运行完成后会保存：

```text
demo/outputs/
|-- 01_detect_result.json
|-- 02_retrieval.json
`-- 03_translate_result.json
```

## 输入与输出格式

输入文件应为 JSON list，每条数据至少需要包含 `sentence` 字段：

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

常用字段说明：

| 字段 | 必需 | 说明 |
| :--- | :--- | :--- |
| `sentence` | 是 | 原始英文句子 |
| `direct_8b` / `direct` | 否 | 基座模型直译结果，可作为翻译增强的 Translation 0 |
| `label` | 否 | 检测评估标签，常用值为 `Figurative` / `Without_Idiom` |
| `reference_zh` | 否 | 中文参考译文，用于 COMET 等评测 |

流程输出会逐步增加以下字段：

| 字段 | 阶段 | 说明 |
| :--- | :--- | :--- |
| `detect_label` | Detect | 是否检测到非常规表达 |
| `dpo_eval_8B` | Detect | 检测模型的原始分析输出 |
| `detect_idiom` | Detect | 从检测输出中解析出的表达列表 |
| `rag_info` | Retrieval | 检索到的知识库释义 |
| `retrieval_top3` | Retrieval | 每个检测表达的 top-3 检索结果 |
| `cot_result` | Translate | 最终翻译结果 |

## 脚本化评测流程

除 Web Demo 外，也可以手动分阶段运行 `eval/` 下的脚本。

### 1. Detect

```bash
CUDA_VISIBLE_DEVICES=0 python eval/detect/detect.py
```

默认配置：

```python
input_file_path = "data/LoML.json"
output_file_path = "eval/detect/result/LoML-result.json"
model_path = "model/detect/qwen_dpo"
```

该脚本会输出检测标签、检测分析，并在数据包含 `label` 字段时计算 Precision、Recall 和 F1。

### 2. Retrieval

```bash
python eval/retrieval/faiss_load.py
```

默认使用：

```text
model/retrieval/qwen3-0.6b-finetuned-new/final
model/retrieval/index/udKB_index_0.6b.faiss
data/udKB.json
```

脚本读取检测结果中的 `detect_idiom`，并写入 `rag_info`。

### 3. Translate

```bash
python eval/translate/translate.py
```

默认配置：

```python
input_file_path = "data/LoML.json"
output_file_path = "eval/translate/Cot_eval_new.json"
model_path = "model/base/Qwen3-8B"
```

翻译逻辑：

1. 未检测到非常规表达时，直接翻译；
2. 检测到表达且存在 `rag_info` 时，使用 RAG 释义和检测分析进行翻译增强；
3. 检测到表达但没有检索结果时，使用 fallback prompt 让模型自行解释并翻译。

### 4. Multi-strategy Inference

`eval/eval_infer.py` 用于对比不同翻译策略：

| 输出文件 | 字段 | 说明 |
| :--- | :--- | :--- |
| `direct.json` | `direct` | 直接翻译 |
| `slangdit.json` | `slangdit` | 先解释非常规表达再翻译 |
| `srag.json` | `srag` | 仅使用检索知识增强翻译 |
| `cot.json` | `cot` | 使用 DeReA 综合推理翻译 |

运行前请根据实验任务修改脚本顶部的输入、输出、模型和目标语言配置。

## 评测指标

### COMET

```bash
python eval/metric/comet/main.py
```

或：

```bash
python eval/metric/comet/main_ud.py
```

运行前需要根据输出文件修改：

```python
input_file_path = "..."
output_file = "..."
SOURCE_KEY = "sentence"
TARGET_KEY = "cot_result"
REFERENCE_KEY = "reference_zh"
```

脚本会分别统计整体样本、非常规表达样本和普通样本的 COMET 分数。

### LLM-as-Judge

```bash
python eval/metric/LAJ/claude_all.py
```

运行前需要配置 OpenAI-compatible API：

```python
client = OpenAI(
    api_key="...",
    base_url="..."
)
```

该脚本主要评估：

1. 非常规表达真实含义是否翻译正确；
2. 表达是否自然融入目标语言上下文。

## 检测模型训练

### SFT

配置文件：

```text
detect/train/sft/sft/qwen3_full_sft.yaml
```

运行示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train detect/train/sft/sft/qwen3_full_sft.yaml
```

数据注册文件：

```text
detect/train/sft/data/dataset_info.json
```

### DPO

配置文件：

```text
detect/train/dpo/dpo/qwen3_lora_dpo.yaml
```

运行示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train detect/train/dpo/dpo/qwen3_lora_dpo.yaml
```

数据注册文件：

```text
detect/train/dpo/data/dataset_info.json
```

训练完成后，可将最终模型放到：

```text
model/detect/qwen_dpo
```

## 检索模型与 FAISS 索引

### 数据处理

```bash
python embedding/train/data.py
```

该脚本将训练数据处理为 SentenceTransformer 使用的 `anchor` / `positive` 格式，并保存为 HuggingFace Dataset。

### 训练 embedding 模型

```bash
python embedding/train/train.py
```

训练完成后，建议将模型保存或复制到：

```text
model/retrieval/qwen3-0.6b-finetuned-new/final/final
```

### 构建 FAISS 索引

```bash
python embedding/retrieve/faiss_set.py
```

建议将最终索引保存到：

```text
model/retrieval/index/udKB_index_0.6b.faiss
```

FAISS 索引必须与 `data/udKB.json` 的条目顺序保持一致。

## 推荐工作流

快速体验：

```bash
python demo/app.py
```

批量实验：

```bash
python eval/detect/detect.py
python eval/retrieval/faiss_load.py
python eval/translate/translate.py
```

结果评测：

```bash
python eval/metric/comet/main.py
python eval/metric/LAJ/claude_all.py
```

## 注意事项

* 当前项目路径已统一为相对路径，运行脚本时请在项目根目录执行命令。
* `benchmark/` 目录当前只保留 `emerging_slang.json`。
* 大模型和索引文件需要手动放入 `model/` 对应目录。
* 如果显存不足，可以先在 Demo 中调小 `Max Items`，或替换为更小的模型。
* Windows PowerShell 下建议从项目根目录运行命令，以避免相对路径解析错误。

