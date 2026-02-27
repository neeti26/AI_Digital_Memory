import os
import json
import numpy as np
import faiss

LINK_FILE = "memory_links.json"


def load_links():
    if os.path.exists(LINK_FILE):
        with open(LINK_FILE, "r") as f:
            return json.load(f)
    return {}


def save_links(links):
    with open(LINK_FILE, "w") as f:
        json.dump(links, f, indent=4)


def update_links(new_index_id, memory_data, index, similarity_threshold=0.75):

    links = load_links()

    # Get embedding of newly added memory
    new_vector = index.reconstruct(new_index_id)

    # Compare against all previous vectors
    for i in range(new_index_id):
        existing_vector = index.reconstruct(i)

        similarity = np.dot(new_vector, existing_vector)

        if similarity >= similarity_threshold:
            links.setdefault(str(new_index_id), []).append(i)
            links.setdefault(str(i), []).append(new_index_id)

    save_links(links)