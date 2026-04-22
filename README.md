# ReDeA

This project is organized into three core components: the **Detection Model (Detect)**, the **Embedding Model (Embedding)**, and **End-to-End Inference and Evaluation (Eval)**.

---

## 1. Detection Model (Detect Model)

This module is responsible for identifying non-standard expressions (slang), and includes both training (SFT/DPO) and inference stages. All models are stored under the `detect/` directory.

### 1.1 Training Directory Structure
* **Root directory**: `detect/train`
* **Subdirectories**:
    * `detect/train/sft`: SFT training for the Qwen3 8B model  
    * `detect/train/dpo`: DPO training for the Qwen3 8B model  

### 1.2 Training Data Locations
The actual training data are stored in the corresponding data directories:
* `/dft/data` (for SFT tasks)
* `/dpo/data` (for DPO tasks)

### 1.3 Training Execution (Qwen3 Example)
* **Training configuration file**: `detect/train/sft/sft/qwen3_full_sft.yaml`
* **Command**:
    ```bash
    nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train sft/qwen3_full_sft.yaml" > sft/train.log 2>&1 &
    ```

### 1.4 Inference and Evaluation
* **Inference script**: `/detect/infer/detect_infer.py`
* **Source dataset**: `/benchmark/LoMI.json`
* **Command**:
    ```bash
    CUDA_VISIBLE_DEVICES=2 python eval/data/dpo_detect.py
    ```

---

## 2. Embedding Model and Knowledge Base (Embedding & Link)

This module is responsible for training the embedding model and building/loading the Faiss vector index. All related code is located under the `embedding/` directory.

### 2.1 Embedding Model Training
* **Training program directory**: `embedding/train/sentence_transformer`
* **Key files**:
    * `data.py`: construction of training data  
    * `train.py`: main training script  
* **Model output directory**: `embedding/train/sentence_transformer/models`

### 2.2 Faiss Index Management
Located under `embedding/train/retrieve/`:
* **Index construction**: `faiss_set.py`
* **Index loading**: `faiss_load.py`

---

## 3. End-to-End Pipeline Execution and Evaluation (Eval)

This module orchestrates the entire translation workflow, including data input, detection, embedding-based retrieval, inference, and scoring. All related code is located under the `eval/` directory.

### 3.1 Core Data and Pipeline
| Stage | File / Script Path | Description |
| :--- | :--- | :--- |
| **LoMI test dataset** | `/benchmark/LoMI.json` | Raw input data |
| **Emerging Slang test dataset** | `/benchmark/emerging_slang.json` | Raw input data |
| **Slang detection** | `/detect/infer/detect_infer.py` | Runs the detection model |
| **Slang embedding** | `embedding/retrieve/faiss_load.py` | Retrieve |Evaluates retrieval quality |

### 3.2 Translation Inference
* **Slang translation agent inference**: `/eval/eval_infer.py`

### 3.3 Result Evaluation
* **COMET scoring**: `eval/comet/main.py`
* **LLM-based evaluation**: `eval/LAJ/all.py`
