import os
import json
import faiss
import numpy as np
import datetime
from sentence_transformers import SentenceTransformer
import torch

# ---------------- Config ----------------
MEMORY_FILE = "memory_store.json"
INDEX_FILE = "memory.index"
LINK_FILE = "memory_links.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

EMBEDDING_DIM = 384


# ---------------- Initialize ----------------
def initialize_memory():

    # Load memory file
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory_data = json.load(f)

        # Backward compatibility upgrade
        for m in memory_data:
            if "source" not in m:
                m["source"] = "legacy"
            if "source_id" not in m:
                m["source_id"] = None
    else:
        memory_data = []

    # Load FAISS index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    return memory_data, index


# ---------------- Save ----------------
def save_memory(memory_data, index):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory_data, f, indent=4)

    faiss.write_index(index, INDEX_FILE)


# ---------------- Link Utilities ----------------
def load_links():
    if os.path.exists(LINK_FILE):
        with open(LINK_FILE, "r") as f:
            return json.load(f)
    return {}


def save_links(links):
    with open(LINK_FILE, "w") as f:
        json.dump(links, f, indent=4)


def update_links(new_id, index, similarity_threshold=0.82):

    links = load_links()

    # embedding of new memory
    new_vector = index.reconstruct(new_id)

    for i in range(new_id):
        existing_vector = index.reconstruct(i)

        similarity = float(np.dot(new_vector, existing_vector))

        if similarity >= similarity_threshold:
            links.setdefault(str(new_id), []).append(i)
            links.setdefault(str(i), []).append(new_id)

    save_links(links)


# ---------------- Add Memory ----------------
def add_memory(content, memory_data, index, source="general", source_id=None):

    # Prevent duplicates
    for m in memory_data:
        if m["content"] == content:
            return  # Skip duplicate

    embedding = model.encode([content], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    index.add(embedding)

    memory_entry = {
        "content": content,
        "timestamp": str(datetime.datetime.now()),
        "source": source,
        "source_id": source_id
    }

    memory_data.append(memory_entry)

    new_id = len(memory_data) - 1
    update_links(new_id, index)

    save_memory(memory_data, index)

# ---------------- Basic Semantic Recall ----------------
def recall_memory(query, memory_data, index, top_k=3):

    if len(memory_data) == 0:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding,
        min(top_k, len(memory_data))
    )

    return [memory_data[i] for i in indices[0]]


# ---------------- Get Linked Memories ----------------
def get_linked_memories(memory_id, memory_data):

    links = load_links()

    if str(memory_id) not in links:
        return []

    linked_ids = links[str(memory_id)]

    return [memory_data[i] for i in linked_ids]


# ---------------- Link-Aware Recall ----------------
def recall_with_links(query, memory_data, index, top_k=3, max_link_expansion=2):

    if len(memory_data) == 0:
        return []

    # Step 1 — semantic recall
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding,
        min(top_k, len(memory_data))
    )

    recalled_ids = list(indices[0])
    expanded_ids = set(recalled_ids)

    # Step 2 — expand via links
    links = load_links()

    for mem_id in recalled_ids:
        linked = links.get(str(mem_id), [])
        for linked_id in linked[:max_link_expansion]:
            expanded_ids.add(linked_id)

    # Step 3 — return expanded memory objects
    return [memory_data[i] for i in expanded_ids]