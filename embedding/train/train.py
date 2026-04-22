import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from datasets import load_from_disk

MODEL_NAME = "/data/shared/Qwen3-Embedding-8B/"  
DATA_PATH = "slang/Link/embedding_train/sentence_transformer/processed_dataset"
OUTPUT_DIR = "slang/Link/embedding_train/sentence_transformer/models/qwen3-8b-finetuned-new"

print("正在加载数据集...")
dataset = load_from_disk(DATA_PATH)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"正在加载模型: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)


loss = MultipleNegativesRankingLoss(model=model)


args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=200,
    save_total_limit=2,           
    
    num_train_epochs=3,           
    per_device_train_batch_size=64, 
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    
    fp16=True,                    
    gradient_accumulation_steps=1, 

    batch_sampler=BatchSamplers.NO_DUPLICATES,
    
    run_name="qwen3-8b-slang-finetune",
)


trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)

print("开始训练...")
trainer.train()


print(f"训练完成，正在保存模型到 {OUTPUT_DIR}/final")
model.save_pretrained(f"{OUTPUT_DIR}/final")
