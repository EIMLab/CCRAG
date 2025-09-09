
import pandas as pd
import json
import torch
import os
import time
import hdbscan
import numpy as np
import pandas as pd
import chromadb
import re
import ollama
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI, AzureOpenAI
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from collections import defaultdict

SAVE_DIR = "output_folder"
os.makedirs(SAVE_DIR, exist_ok=True)

FILE_PATH_JSON = "automatic_chunk_for_cluster_to_json.json"
CLUSTER_CHUNK_JSON = "cluster_chunk_results.json"


file_path = "output_folder/automatic_chunk_for_cluster.xlsx"
sheet_name = "Sheet1"


df = pd.read_excel(file_path, sheet_name=sheet_name)

json_list = []
for _, row in df.iterrows():
    item = {
        "col_name": row.get('col_name', ''),
        "filename": row.get('filename', ''),  
        "unit": row.get('unit', ''),  
    }
    json_list.append(item)

with open(os.path.join(SAVE_DIR, FILE_PATH_JSON), 'w', encoding='utf-8') as f:
    json.dump(json_list, f, ensure_ascii=False, indent=2)

print("Excel_to_JSON")



model_path = "output_folder/finetuning_embedding_model/selected_finetuned_embedding_model"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

embedding_model = SentenceTransformer(model_path, device=device)
embedding_model.to(device)  

embedding_model.eval()



class TreeNode:
    def __init__(self, depth, text, parent_node=None):
        self.depth = depth             
        self.text = text              
        self.parent_node = parent_node  
        self.child_nodes = []        
        self.node_id = None   

    def add_child(self, child_node):
        self.child_nodes.append(child_node)

    def to_dict(self):
        return {
            "depth": self.depth,
            "text": self.text,
            "children": [child.to_dict() for child in self.child_nodes]
        }


client_openai = OpenAI(
    api_key="key",
    base_url="url",
)
client_azure = AzureOpenAI(
    api_version="version",
    azure_endpoint="url",
    api_key="key",
)
client_qwen =  OpenAI(
    api_key="key",
    base_url="url",
)


embedding_model = SentenceTransformer(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"


def embed_texts(text_list):
    embeddings = embedding_model.encode(text_list, batch_size=64, show_progress_bar=True, device=device)
    return embeddings

def cluster_embeddings(embeddings, min_cluster_size=2, n_components=30):

    embeddings_norm = normalize(embeddings)

    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_norm)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,  
        min_samples=1,                    
        metric='euclidean',
        prediction_data=True
    )
    labels = clusterer.fit_predict(embeddings_pca)

    return labels


def summarize_cluster(texts, cluster_id, round_num, label):
    if label == True:
        try:
            combined_text = "\n\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
            prompt = f"You are a professional document summarization assistant. The following is a collection of related document contents: \n{combined_text}\n\n请Please provide a summary of the above material. The summary should carefully extract and synthesize the key information, covering all essential details, with a length approximately half of the original content. The response must be written in formal and professional academic prose."
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            completion = client_qwen.chat.completions.create(
                model="qwen-plus",
                temperature=0,
                messages=messages,
            )
            summary = completion.choices[0].message.content
            return summary
        except Exception as e:
            error_message = str(e)
            print(error_message)
            time.sleep(10)
            return "\n\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    else:
        summary = "\n\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
        return summary

    
def recursive_cluster_and_summarize(nodes, depth=0, min_cluster_size=2):
    
    texts = [node.text for node in nodes]
    embeddings = embed_texts(texts)
    labels = cluster_embeddings(embeddings, min_cluster_size)

    clustered_nodes = {}
    max_label = max(labels) if len(labels) > 0 else 0
    singleton_id = max_label + 1

    outlier_count = 0 
    inlier_count = 0 

    for idx, label in enumerate(labels):
        if label == -1:
            clustered_nodes[singleton_id] = [nodes[idx]]
            singleton_id += 1
            outlier_count += 1 
        else:
            clustered_nodes.setdefault(label, []).append(nodes[idx])
            inlier_count += 1  

    layer_cluster_counts.append(len(clustered_nodes))

    flag = True
    for label in labels:
        if label != -1:
            flag = False
            break

    if len(clustered_nodes) <= 500 or flag:
        return nodes

    parent_nodes = []
    for cluster_id, child_nodes in tqdm(clustered_nodes.items(), desc=f"Clustering layer {depth}"):
        label = True if cluster_id <= max_label else False
        cluster_texts = [node.text for node in child_nodes]
        summary = summarize_cluster(cluster_texts, cluster_id, depth, label)

        parent_node = TreeNode(depth=depth + 1, text=summary)
        for child in child_nodes:
            child.parent_node = parent_node
            parent_node.add_child(child)

        parent_nodes.append(parent_node)

    return recursive_cluster_and_summarize(parent_nodes, depth=depth + 1, min_cluster_size=min_cluster_size)


if __name__ == "__main__":
    with open(os.path.join(SAVE_DIR, FILE_PATH_JSON), "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_texts = [item["unit"] for item in data]
    layer_cluster_counts = []

    initial_nodes = [TreeNode(depth=0, text=text) for text in raw_texts]

    top_nodes = recursive_cluster_and_summarize(initial_nodes, depth=0)

    print("\n📊 Cluster statistics results: ")
    for i, count in enumerate(layer_cluster_counts):
        print(f"Level {i}: {count} clusters")
    print(f"\n🔚 Total number of clustering levels：{len(layer_cluster_counts)}")

    root_node = TreeNode(depth=-1, text="root")
    for node in top_nodes:
        node.parent_node = root_node
        root_node.add_child(node)

    with open(os.path.join(SAVE_DIR, CLUSTER_CHUNK_JSON), "w", encoding="utf-8") as f:
        json.dump(root_node.to_dict(), f, ensure_ascii=False, indent=2)

    print("✅ Done!")


def build_tree_from_json(data, parent_node=None):
    node = TreeNode(depth=data["depth"], text=data["text"], parent_node=parent_node)
    for child_data in data.get("children", []):
        child_node = build_tree_from_json(child_data, parent_node=node)
        node.add_child(child_node)
    return node

def assign_ids_by_layer(root):
    layer_dict = {}
    def dfs(node, depth):
        if depth not in layer_dict:
            layer_dict[depth] = []
        layer_dict[depth].append(node)
        for child in node.child_nodes:
            dfs(child, depth + 1)
    dfs(root, 0)

    node_id_map = {}
    for depth, nodes in layer_dict.items():
        for i, node in enumerate(nodes):
            node_id = f"depth{depth}_{i}"
            node.node_id = node_id
            node_id_map[node_id] = node
    return layer_dict, node_id_map

def create_vector_library_A(data, embedding_model):
    filename_to_code = {}
    code_to_filename = {}

    filenames = sorted(set(item["filename"] for item in data))
    for i, filename in enumerate(filenames, 1):
        code = f"DOC{i:03d}" 
        filename_to_code[filename] = code
        code_to_filename[code] = filename
        print(f"{filename} => {code}")

    client = chromadb.HttpClient(host="localhost", port=8000)
    vector_libraries_A = {}


    for code in filename_to_code.values():
        try:
            client.delete_collection(name=code)
        except:
            pass

    for item in tqdm(data):
        filename = item["filename"]
        chunk_text = item["unit"]
        col_name = str(item["col_name"])

        file_code = filename_to_code[filename]
 
        if file_code not in vector_libraries_A:
            vector_libraries_A[file_code] = client.create_collection(name=file_code)

        chunk_embedding = embedding_model.encode([chunk_text])[0]

        try:
            vector_libraries_A[file_code].add(
                ids=[col_name],
                embeddings=[chunk_embedding],
                documents=[chunk_text]
            )
        except Exception as e:
            print(f"❌ Failed: file_code={file_code}, id={col_name}, chunk_length={len(chunk_text)}")
            print(f"Information: {e}")

            return vector_libraries_A


def build_layered_vector_libraries(layer_dict, embedding_model, save_name_prefix="Layer"):
    client = chromadb.HttpClient(host="localhost", port=8000)

    for depth in sorted(layer_dict.keys()):
        if depth == 0:
            continue 

        nodes = layer_dict[depth]
        collection_name = f"{save_name_prefix}{depth}"
        try:
            client.delete_collection(name=collection_name)
        except:
            pass
        collection = client.create_collection(name=collection_name)

        for node in tqdm(nodes, desc=f"Encoding depth {depth}"):
            embedding = embedding_model.encode([node.text])[0]
            child_ids = [child.node_id for child in node.child_nodes]

            metadata = {"children": json.dumps(child_ids)}

            collection.add(
                ids= [node.node_id],
                embeddings=[embedding],
                documents=[node.text],
                metadatas=[metadata]
            )
    print("✅ All done!")


if __name__ == "__main__":
    with open(os.path.join(SAVE_DIR, FILE_PATH_JSON), "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["unit"] for item in data]

    vector_libraries_A = create_vector_library_A(data, embedding_model)

    with open(os.path.join(SAVE_DIR, CLUSTER_CHUNK_JSON), "r", encoding="utf-8") as f:
        tree_data = json.load(f)


    root = build_tree_from_json(tree_data)


    layer_dict, node_id_map = assign_ids_by_layer(root)
    
    build_layered_vector_libraries(layer_dict, embedding_model)
    print("✅ Vector databases A and B have been successfully created and the old versions cleared!")



def hierarchical_search(query_embedding, max_depth, save_name_prefix="Layer"):
    client = chromadb.HttpClient(host="localhost", port=8000)
    current_ids = None  

    for depth in range(1, max_depth):
        collection_name = f"{save_name_prefix}{depth}"
        collection = client.get_or_create_collection(name=collection_name)

        if current_ids is None:
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=4
            )
        else:
            if current_ids is not None:
                result = collection.query(
                query_embeddings=[query_embedding],
                n_results=4,
                ids = current_ids
            )

        if not result["ids"] or not result["ids"][0]:
            print(f"❌ depth={depth} no results")
            return ""

        current_id = result["ids"][0][0]

        node_info = collection.get(ids=[current_id])
        if node_info and node_info["metadatas"] and node_info["metadatas"][0].get("children"):
            try:
                child_ids = json.loads(node_info["metadatas"][0]["children"])
                current_ids = child_ids
            except Exception as e:
                print(f"⚠️ Failed to parse child node: {e}")
                break
        else:
            print("⚠️ No child node found, exiting early.")
            break

    collection_name = f"{save_name_prefix}{max_depth}"
    collection = client.get_or_create_collection(name=collection_name)
    final_nodes = collection.get(ids=current_ids)

    return "\n".join([f"{i+1}. {text}" for i, text in enumerate(final_nodes["documents"])]) if final_nodes else ""

qa_path = "test_set.xlsx"
df_qa = pd.read_excel(qa_path, sheet_name="Sheet1", usecols=["number", "type", "stem", "option", "answer"])
df_qa["answer"] = df_qa["answer"].astype(str).str.strip().str.upper()
model_list = ["qwen2.5:0.5b", "qwen2.5:3b", "qwen2.5:7b", 'gemma3:1b', 'gemma3:4b', 'gemma3:12b']
for model_name in model_list:
    df_qa[model_name] = ""


chroma = chromadb.HttpClient(host="localhost", port=8000)


with open(os.path.join(SAVE_DIR, FILE_PATH_JSON), "r", encoding="utf-8") as f:
    data = json.load(f)

filename_to_code = {}
code_to_filename = {}

filenames = sorted(set(item["filename"] for item in data))
for i, filename in enumerate(filenames, 1):
    code = f"DOC{i:03d}"
    filename_to_code[filename] = code
    code_to_filename[code] = filename
    print(f"{filename} => {code}")

vector_libraries_A = {}
for code in filename_to_code.values():
    try:
        vector_libraries_A[code] = chroma.get_collection(name=code)
    except Exception as e:
        print(f"⚠️ Collection {code} not found in vector database A. Error message: {e}")

with open(os.path.join(SAVE_DIR, CLUSTER_CHUNK_JSON), "r", encoding="utf-8") as f:
    tree_data = json.load(f)

root = build_tree_from_json(tree_data)
layer_dict, node_id_map = assign_ids_by_layer(root)

vector_libraries_B = {}

for depth in sorted(layer_dict.keys()):
    if depth == 0:
        continue
    collection_name = f"Layer{depth}"
    try:
        vector_libraries_B[depth] = chroma.get_collection(name=collection_name)
        print(f"✅ Successfully loaded {collection_name}")
    except Exception as e:
        print(f"⚠️ Collection {collection_name} not found in vector database A. Error message: {e}")


count_A = 0 
count_B = 0

model_use_stats = {model: {"A": 0, "B": 0} for model in model_list}

for model_name in model_list:
    print(f"\n🚀 当前模型：{model_name}")

    for idx, row in tqdm(df_qa.iterrows(), total=len(df_qa)):
        qid = row["number"]
        qtype = row["type"]
        question = row["stem"]
        option = row["option"]
        answer = row["answer"]
        query_embedding = embedding_model.encode([question])[0]

        is_multiple_choice = (qtype == "multiple_choice") or (len(answer) > 1 and re.fullmatch(r"[A-E]+", answer))

        has_filename = False
        used_library = "B"

        file_name_match = re.search(r"《(.*?)》", question)
        if file_name_match:
            has_filename = True
            file_name = file_name_match.group(1)

            file_code = filename_to_code.get(file_name)
            if file_code:
                vector_library = vector_libraries_A.get(file_code)
            else:
                vector_library = None

            if vector_library is not None:
                try:
                    results = vector_library.query(query_embeddings=[query_embedding], n_results=8)
                    context = "\n\n".join(results["documents"][0])
                    count_A += 1
                    model_use_stats[model_name]["A"] += 1
                    used_library = "A"
                except:
                    print(f"⚠️ Failed to query vector database A, switched to vector database B: 《{file_name}》")
                    context = hierarchical_search(query_embedding, max(layer_dict.keys()))
                    count_B += 1
                    model_use_stats[model_name]["B"] += 1
                    used_library = "B"
            else:
                context = hierarchical_search(query_embedding, max(layer_dict.keys()))
                count_B += 1
                model_use_stats[model_name]["B"] += 1
                print(f"⚠️ The corresponding file name or ID was not found in vector database A: 《{file_name}》, using vector database B instead")
                used_library = "B"
        else:
            context = hierarchical_search(query_embedding, max(layer_dict.keys()))
            count_B += 1
            model_use_stats[model_name]["B"] += 1
            used_library = "B"

        df_qa.at[idx, f"{model_name}_whether_contains_filename"] = has_filename
        df_qa.at[idx, f"{model_name}_used_database"] = used_library

        model_prompt = f"""
Please answer the question based on the following context.
Context：
====
{context}
====
 Question type: {'Multiple-choice' if is_multiple_choice else 'Single-choice'}.
{question}
{option}

Requirements: 
(1) Output only the option letter(s), without explanation. 
(2) {'For multiple-choice questions, the answer should be in the format like ACD, without commas or spaces, letters in alphabetical order, no duplicates, and at most 5 letters.' if is_multiple_choice else 'For single-choice questions, the answer should be only one letter: A/B/C/D.'}.
"""

        # === 调用模型 ===
        try:
            response = ""
            stream = ollama.generate(model=model_name, prompt=model_prompt, stream=True, options={"temperature": 0})
            for chunk in stream:
                if chunk["response"]:
                    response += chunk["response"]
            response = re.sub(r"[^A-E]", "", response.upper())
            df_qa.at[idx, model_name] = response
        except Exception as e:
            print(f"Model {model_name} failed to answer question {qid}: {e}")
            df_qa.at[idx, model_name] = "ERROR"

print(f"\n📊 Vector database usage statistics (total): vector database A: {count_A} times, vector database B: {count_B} times")

print("\n📊 Number of times each model used vector database A/B：")
for model_name in model_list:
    a_used = model_use_stats[model_name]["A"]
    b_used = model_use_stats[model_name]["B"]
    print(f"{model_name}：vector database A = {a_used}，vector database B = {b_used}")

# %%
print("\n=== Overall Accuracy of Each Model ===")
for model_name in model_list:
    correct = (df_qa[model_name] == df_qa["answer"]).sum()
    total = len(df_qa)
    print(f"{model_name}: {correct}/{total} Accuracy: {correct / total:.2%}")

print("\n=== Accuracy of Each Model by Question Type ===")
for model_name in model_list:
    print(f"\nLLM: {model_name}")
    for qtype in df_qa["type"].unique():
        sub_df = df_qa[df_qa["type"] == qtype]
        correct = (sub_df[model_name] == sub_df["answer"]).sum()
        total = len(sub_df)
        print(f"{qtype}: {correct}/{total} Accuracy: {correct / total:.2%}")

print("\n=== Vector Database Usage Statistics (Total) ===")
print(f"Number of questions using Vector DB A (with filename): {count_A}")
print(f"Number of questions using Vector DB B (without filename, cluster retrieval): {count_B}")

print("\n=== Number of Questions Using Vector DB A/B by Model ===")
for model_name in model_list:
    used_A = model_use_stats[model_name]["A"]
    used_B = model_use_stats[model_name]["B"]
    print(f"{model_name}: Vector DB A = {used_A}, Vector DB B = {used_B}")

QA_output_path = "output_folder/Answer Results.xlsx"
df_qa.to_excel(QA_output_path, index=False)
print(f"\n✅ Answer results saved to: {QA_output_path}")



