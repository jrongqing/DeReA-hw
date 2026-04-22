# DeReA

该项目由三个核心组件组成：**检测模型 (Detect)**、**嵌入模型 (Embedding)** 以及 **端到端推理与评估 (Eval)**。

---

## 1. 检测模型 (Detect Model)

该模块负责识别非规范表达（如俚语），涵盖了训练（SFT/DPO）和推理阶段。所有模型相关文件均存储在 `detect/` 目录下。

### 1.1 训练目录结构
* **根目录**: `detect/train`
* **子目录**:
    * `detect/train/sft`: Qwen3 8B 模型的 SFT 训练
    * `detect/train/dpo`: Qwen3 8B 模型的 DPO 训练

### 1.2 训练数据路径
实际训练数据存储在相应的候选数据目录中：
* `/dft/data` (用于 SFT 任务)
* `/dpo/data` (用于 DPO 任务)

### 1.3 训练执行 (以 Qwen3 为例)
* **训练配置文件**: `detect/train/sft/sft/qwen3_full_sft.yaml`
* **执行命令**:
    ```bash
    nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train sft/qwen3_full_sft.yaml" > sft/train.log 2>&1 &
    ```

### 1.4 推理与评估
* **推理脚本**: `/detect/infer/detect_infer.py`
* **源数据集**: `/benchmark/LoMI.json`
* **执行命令**:
    ```bash
    CUDA_VISIBLE_DEVICES=2 python eval/data/dpo_detect.py
    ```

---

## 2. 嵌入模型与知识库 (Embedding & Link)

该模块负责训练嵌入模型以及构建/加载 Faiss 向量索引。相关代码位于 `embedding/` 目录下。

### 2.1 嵌入模型训练
* **训练程序目录**: `embedding/train/sentence_transformer`
* **核心文件**:
    * `data.py`: 训练数据构建
    * `train.py`: 主训练脚本
* **模型输出目录**: `embedding/train/sentence_transformer/models`

### 2.2 Faiss 索引管理
代码位于 `embedding/train/retrieve/` 目录下：
* **索引构建**: `faiss_set.py`
* **索引加载**: `faiss_load.py`

---

## 3. 端到端流水线执行与评估 (Eval)

该模块统筹整个翻译工作流，包括数据输入、检测、基于嵌入的检索、推理及评分。相关代码均位于 `eval/` 目录下。

### 3.1 核心数据与流水线 (Pipeline)
| 阶段 | 文件 / 脚本路径 | 描述 |
| :--- | :--- | :--- |
| **LoMI 测试集** | `/benchmark/LoMI.json` | 原始输入数据 |
| **新锐俚语测试集** | `/benchmark/emerging_slang.json` | 原始输入数据 |
| **俚语检测** | `/detect/infer/detect_infer.py` | 运行检测模型 |
| **俚语嵌入检索** | `embedding/retrieve/faiss_load.py` | 执行检索并评估检索质量 |

### 3.2 翻译推理
* **俚语翻译智能体推理**: `/eval/eval_infer.py`

### 3.3 结果评估
* **COMET 评分**: `eval/comet/main.py`
* **基于 LLM 的评估**: `eval/LAJ/all.py`