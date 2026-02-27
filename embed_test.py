from sentence_transformers import SentenceTransformer
import torch
import time

print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

texts = ["AI is transforming memory systems"] * 1000

start = time.time()
embeddings = model.encode(texts, batch_size=64)
end = time.time()

print("Embedding shape:", embeddings.shape)
print("Time taken:", end - start)
