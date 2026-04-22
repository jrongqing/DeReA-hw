import faiss
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import torch
import gc


DATA_PATH = 'slang/udKB.json'
MODEL_PATH = "slang/Link/embedding_train/sentence_transformer/models/qwen3-0.6b-finetuned-new/final"
OUTPUT_DIR = 'slang/Link/embedding_train/06B_embedding/index'
os.makedirs(OUTPUT_DIR, exist_ok=True)
INDEX_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'udKB_index_0.6b.faiss')
EMBEDDING_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'udKB_embeddings_0.6b.npy')


print(f"Loading model from: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
model.max_seq_length = 512


model.half() 
print(f"Model loaded on device: {model.device} (FP16 enabled)")


def data_generator(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:

        idiom = entry.get('idiom')
        if idiom is None:
            idiom = ""
        else:
            idiom = str(idiom).strip() 
            

        definition = entry.get('definitions')
        if definition is None:
            definition = ""
        else:
            definition = str(definition).strip()

        yield f"{idiom}: {definition}"
    
    del data
    gc.collect()

print("Counting entries...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    total_entries = len(json.load(f))


embedding_dim = model.get_sentence_embedding_dimension() 
print(f"Embedding dim: {embedding_dim}")


index = faiss.IndexFlatIP(embedding_dim)


if os.path.exists(EMBEDDING_OUTPUT_PATH):
    os.remove(EMBEDDING_OUTPUT_PATH)
    

all_embeddings_mmap = np.memmap(
    EMBEDDING_OUTPUT_PATH, 
    mode='w+', 
    dtype='float32', 
    shape=(total_entries, embedding_dim)
)


BATCH_SIZE = 512 
batch_texts = []
global_idx = 0

print("Starting encoding and indexing...")
gen = data_generator(DATA_PATH)

pbar = tqdm(total=total_entries, unit="docs")

for text in gen:
    batch_texts.append(text)
    
    if len(batch_texts) >= BATCH_SIZE:
        embeddings = model.encode(
            batch_texts, 
            batch_size=BATCH_SIZE, 
            normalize_embeddings=True, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        

        current_batch_len = len(embeddings)
        all_embeddings_mmap[global_idx : global_idx + current_batch_len] = embeddings
        

        index.add(embeddings)
        

        global_idx += current_batch_len
        pbar.update(current_batch_len)

        batch_texts = []


if batch_texts:
    embeddings = model.encode(
        batch_texts, 
        batch_size=BATCH_SIZE, 
        normalize_embeddings=True, 
        convert_to_numpy=True, 
        show_progress_bar=False
    )
    current_batch_len = len(embeddings)
    all_embeddings_mmap[global_idx : global_idx + current_batch_len] = embeddings
    index.add(embeddings)
    pbar.update(current_batch_len)

pbar.close()


all_embeddings_mmap.flush()


print(f"Saving FAISS index to {INDEX_OUTPUT_PATH}...")
faiss.write_index(index, INDEX_OUTPUT_PATH)

print("Done! ✅")
