from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model on GPU
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Sample documents
documents = [
    "Neural networks use backpropagation to learn.",
    "Transformers use attention mechanisms.",
    "Random Forest is an ensemble learning method.",
    "XGBoost is a gradient boosting algorithm."
]

# Create embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Query
query = "How do neural networks learn?"
query_embedding = model.encode([query], convert_to_numpy=True)

# Search
k = 2
distances, indices = index.search(query_embedding, k)

print("Top results:")
for i in indices[0]:
    print(documents[i])