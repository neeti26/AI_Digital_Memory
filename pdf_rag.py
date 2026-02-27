from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import torch
import time
import re
import os
import pickle

# ---------------- Load Embedding Model ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


# ---------------- Load PDF ----------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ---------------- Clean Text ----------------
def clean_text(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.lower().startswith("8. references"):
            break

        if "roll no" in line.lower():
            continue
        if "prn" in line.lower():
            continue
        if "date of submission" in line.lower():
            continue

        cleaned.append(line)

    return " ".join(cleaned)


# ---------------- Section-Based Chunking ----------------
def chunk_text(text):
    sections = re.split(r'\n?\s*\d+\.\s+', text)

    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) > 50:
            chunks.append(section)

    return chunks


# ---------------- Build Index ----------------
def build_index(chunks):
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        batch_size=64
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


# ---------------- Save Memory ----------------
def save_memory(index, chunks, metadata, folder="memory"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    faiss.write_index(index, os.path.join(folder, "index.faiss"))

    with open(os.path.join(folder, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print("Memory saved to disk.")


# ---------------- Load Memory ----------------
def load_memory(folder="memory"):
    index_path = os.path.join(folder, "index.faiss")
    chunks_path = os.path.join(folder, "chunks.pkl")
    metadata_path = os.path.join(folder, "metadata.pkl")

    if not os.path.exists(index_path):
        return None, None, None

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    print("Memory loaded from disk.")
    return index, chunks, metadata


# ---------------- Initialize Multi-Document Memory ----------------
def initialize_rag(doc_folder="documents"):
    index, chunks, metadata = load_memory()

    current_docs = set([f for f in os.listdir(doc_folder) if f.endswith(".pdf")])

    if chunks is None:
        chunks = []
        metadata = []
        index = None

    existing_docs = set(metadata)

    # ---------------- Remove Deleted Docs ----------------
    deleted_docs = existing_docs - current_docs
    if deleted_docs:
        print("Deleted documents detected:", deleted_docs)

        new_chunks = []
        new_metadata = []

        for chunk, doc in zip(chunks, metadata):
            if doc not in deleted_docs:
                new_chunks.append(chunk)
                new_metadata.append(doc)

        chunks = new_chunks
        metadata = new_metadata

        index = build_index(chunks)
        save_memory(index, chunks, metadata)

    # ---------------- Add New Docs ----------------
    new_docs = current_docs - existing_docs
    for file in new_docs:
        print(f"New document detected: {file}")

        path = os.path.join(doc_folder, file)
        text = load_pdf(path)
        text = clean_text(text)
        new_chunks = chunk_text(text)

        embeddings = model.encode(new_chunks, convert_to_numpy=True, batch_size=64)

        if index is None:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        for chunk in new_chunks:
            chunks.append(chunk)
            metadata.append(file)

        save_memory(index, chunks, metadata)

    if index is None:
        print("Building memory from scratch...")
        index = build_index(chunks)
        save_memory(index, chunks, metadata)

    print("Memory ready.")
    return chunks, metadata, index
# ---------------- Retrieve Chunks ----------------
def retrieve_chunks(user_query, chunks, metadata, index, top_k=1):
    start_time = time.time()
    query = user_query.strip().lower()

    mentioned_docs = []

    unique_docs = set(metadata)
    for doc_name in unique_docs:
        clean_name = doc_name.lower().replace(".pdf", "")
        if clean_name in query:
            mentioned_docs.append(doc_name)

    # If user explicitly mentions document names
    if mentioned_docs:
        retrieved_chunks = []
        retrieved_sources = []

        for doc in mentioned_docs:
            for i in range(len(metadata)):
                if metadata[i] == doc:
                    retrieved_chunks.append(chunks[i])
                    retrieved_sources.append(doc)
                    break

        retrieval_time = time.time() - start_time
        return retrieved_chunks, retrieved_sources, retrieval_time

    # Otherwise semantic search
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []
    retrieved_sources = []

    for i in indices[0]:
        retrieved_chunks.append(chunks[i])
        retrieved_sources.append(metadata[i])

    retrieval_time = time.time() - start_time
    return retrieved_chunks, retrieved_sources, retrieval_time