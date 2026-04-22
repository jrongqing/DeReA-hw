import json
import random
from datasets import Dataset, DatasetDict

INPUT_FILE = "slang/Link/embedding_train/06B_embedding/pie_train_data_cleaned.json"
OUTPUT_PATH = "slang/Link/embedding_train/sentence_transformer/processed_dataset"

def load_and_process_data(filepath):
    anchors = []
    positives = []
    
    print(f"正在读取数据: {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"原始条目数: {len(raw_data)}")
    
    for entry in raw_data:
        positive_text = f"{entry['idiom']}: {entry['definition']}"
        
        try:
            variations = json.loads(entry['train_data'])
            
            for query in variations:
                if query.strip(): 
                    anchors.append(query)
                    positives.append(positive_text)
                    
        except json.JSONDecodeError:
            print(f"Warning: 解析 train_data 失败: {entry.get('idiom')}")
            continue

    print(f"生成的训练对总数: {len(anchors)}")
    
    dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives
    })
    
    return dataset


full_dataset = load_and_process_data(INPUT_FILE)


dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)

print(f"正在保存处理后的数据集到: {OUTPUT_PATH}")
dataset_split.save_to_disk(OUTPUT_PATH)
print("数据处理完成。")
