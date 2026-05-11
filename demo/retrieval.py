import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


detect_idiom = ["give someone the benefit of the doubt"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_path = PROJECT_ROOT / "model" / "retrieval" / "qwen3-0.6b-finetuned-new" / "final" / "final"
index_path = PROJECT_ROOT / "model" / "retrieval" / "index" / "udKB_index_0.6b.faiss"
idiom_dict_path = PROJECT_ROOT / "data" / "udKB.json"

model = SentenceTransformer(str(model_path))
index = faiss.read_index(str(index_path))


def get_most_similar_idiom(query, index, k=3):
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0], query_embedding[0]


def get_idiom_info(idiom_dict, idiom_id):
    idiom_id = int(idiom_id)
    if isinstance(idiom_dict, list):
        return idiom_dict[idiom_id]
    return idiom_dict[str(idiom_id)]


with idiom_dict_path.open("r", encoding="utf-8") as f:
    idiom_dict = json.load(f)

rag_info = []
for idiom in detect_idiom:
    most_similar_ids, distances, embedding = get_most_similar_idiom(idiom, index, k=3)
    top3 = []
    for idiom_id, distance in zip(most_similar_ids, distances):
        matched_info = get_idiom_info(idiom_dict, idiom_id)
        top3.append(
            {
                "query": idiom,
                "matched_id": int(idiom_id),
                "score": float(distance),
                "matched_info": matched_info,
            }
        )
        rag_info.append(matched_info)

    print("Query idiom:")
    print(idiom)
    print("\nEmbedding shape:")
    print(embedding.shape)
    print("\nTop 3 retrieved info:")
    print(json.dumps(top3, ensure_ascii=False, indent=2))

print("\nFinal rag_info:")
print(json.dumps(rag_info, ensure_ascii=False, indent=2))


# Example:
# python demo/retrieval.py
