import faiss
from tqdm import tqdm
import json
import numpy as np
from sentence_transformers import SentenceTransformer



model_path = "model/retrieval/qwen3-0.6b-finetuned-new/final"
model = SentenceTransformer(model_path)


index = faiss.read_index('model\retrieval\index\udKB_index_0.6b.faiss')


def get_most_similar_idiom(query,index, k=3):
    query_embedding = model.encode([query])  
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    D, I = index.search(query_embedding, k)  
    return I[0], D[0]

with open("data\udKB.json", 'r', encoding='utf-8') as f:
    idiom_dict = json.load(f)

input_path = "eval\detect\result\detect_result.json"
output_path = "eval\retrieval\LoML-linked.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)
for entry in tqdm(data):
    rag_info = []
    for idiom in entry["detect_idiom"]:
        most_similar_idiom, distance = get_most_similar_idiom(idiom,index)
        for id in most_similar_idiom:
            rag_info.append(idiom_dict[id])
        
    entry["rag_info"] = rag_info

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
