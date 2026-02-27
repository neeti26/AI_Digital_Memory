import time
import json
import torch
import os
from pdf_rag import initialize_rag, retrieve_chunks
from local_llm import generate_answer

# ---------------- Initialize Memory ----------------
print("Initializing AI Digital Memory...")
chunks, metadata, index = initialize_rag("documents")
print("Memory Ready!")

conversation_history = []
debug_mode = False

# ---------------- Logging Function ----------------
def log_interaction(question, answer):
    log_entry = {
        "question": question,
        "answer": answer,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if not os.path.exists("memory_logs.json"):
        with open("memory_logs.json", "w") as f:
            json.dump([], f)

    with open("memory_logs.json", "r+") as f:
        data = json.load(f)
        data.append(log_entry)
        f.seek(0)
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question (type 'exit' to quit): ")

        # Exit
        if user_query.lower() == "exit":
            break

        # Stats Command
        if user_query.lower() == "/stats":
            print("\n===== MEMORY STATS =====")
            print("Total Documents:", len(set(metadata)))
            print("Total Chunks:", len(chunks))
            print("Vector Dimension:", index.d)
            print("Embedding Model: all-MiniLM-L6-v2")
            print("LLM Model: microsoft/Phi-3-mini-4k-instruct")
            print("GPU Available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("GPU:", torch.cuda.get_device_name(0))
                print("GPU Memory Allocated:",
                      round(torch.cuda.memory_allocated() / 1024**2, 2), "MB")
            print("========================")
            continue

        # Debug Toggle
        if user_query.lower() == "/debug on":
            debug_mode = True
            print("Debug mode enabled.")
            continue

        if user_query.lower() == "/debug off":
            debug_mode = False
            print("Debug mode disabled.")
            continue

        start_total = time.time()

        # -------- Retrieval --------
        retrieved_chunks, sources, retrieval_time = retrieve_chunks(
            user_query, chunks, metadata, index, top_k=1
        )

        context = "\n\n".join(retrieved_chunks)

        # Add conversation memory
        history_context = ""
        for q, a in conversation_history[-3:]:
            history_context += f"Previous Question: {q}\nPrevious Answer: {a}\n\n"

        full_context = history_context + context

        # -------- Generation --------
        start_generation = time.time()
        answer = generate_answer(full_context, user_query)
        generation_time = time.time() - start_generation

        conversation_history.append((user_query, answer))

        # Log to file
        log_interaction(user_query, answer)

        total_time = time.time() - start_total

        print("\n==============================")
        print("Final Answer:\n")
        print(answer)

        print("\nRetrieved From:")
        for s in sources:
            print("-", s)

        if debug_mode:
            print("\n--- DEBUG INFO ---")
            print("Retrieved Chunk Preview:\n")
            print(context[:500])

        print("\n------------------------------")
        print(f"Retrieval Time: {retrieval_time:.4f} sec")
        print(f"Generation Time: {generation_time:.4f} sec")
        print(f"Total Time: {total_time:.4f} sec")
        print("==============================")