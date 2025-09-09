

from sentence_transformers import SentenceTransformer
import ollama
import pandas as pd
import os
import re
import chromadb
import numpy as np
from tqdm import tqdm


embedding_model_path = "output_folder/finetuning_embedding_model/selected_finetuned_embedding_model"
embedding_model = SentenceTransformer(embedding_model_path)

def get_embedding(text):
    return embedding_model.encode(text, normalize_embeddings=True).tolist()

def load_units_by_filename(file_path: str, sheet_name: str = "1") -> dict:

    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=["filename", "unit"])
    df.columns = ["filename", "unit"]  
    df["filename"] = df["filename"].astype(str).str.replace("。", "", regex=False) 
    grouped = df.groupby("filename")["unit"].apply(lambda x: x.dropna().astype(str).tolist())
    return grouped.to_dict()
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
def split_and_cluster_chunks(units: list, threshold: float = 0.9,
                              min_chunk_size: int = 1, max_chunk_size: int = 10):

    print("Embedding generation in progress for text chunks...")
    
    embeddings_list = [get_embedding(unit) for unit in tqdm(units)]

    chunks, current_chunk = [], [units[0]]
    for i in range(1, len(units)):
        sim = cosine_similarity(embeddings_list[i - 1], embeddings_list[i])
        if sim < threshold and len(current_chunk) >= min_chunk_size:
            chunks.append(' '.join(current_chunk).strip())
            current_chunk = [units[i]]
        else:
            current_chunk.append(units[i])
        if len(current_chunk) >= max_chunk_size:
            chunks.append(' '.join(current_chunk).strip())
            current_chunk = []
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())

    print(f"A total of {len(chunks)} results have been generated.")
    return chunks

file_path = "knowledge_unit.xlsx"
sheet_name = "sheet1"

file_units_dict = load_units_by_filename(file_path, sheet_name)

filename_to_code = {}
code_to_filename = {}
for i, filename in enumerate(sorted(file_units_dict.keys()), 1):
    code = f"D{i}"
    filename_to_code[filename] = code
    code_to_filename[code] = filename
    print(f"{filename} => {code}")

chunk_records = []

for filename, units in file_units_dict.items():
    clustered_chunks = split_and_cluster_chunks(units)
    for chunk in clustered_chunks:
        chunk_records.append({"filename": filename, "unit": chunk})

df_chunks = pd.DataFrame(chunk_records)
chunk_output_path = "output_folder/automatic_chunk_for_cluster.xlsx"
df_chunks.to_excel(chunk_output_path, index=False)
print(f"Results have been saved in {chunk_output_path}")



