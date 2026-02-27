from pypdf import PdfReader

def ingest_pdf(file_path, doc_name, add_memory, memory_data, index, chunk_size=800):

    reader = PdfReader(file_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    chunks = [
        full_text[i:i + chunk_size]
        for i in range(0, len(full_text), chunk_size)
    ]

    for chunk in chunks:
        add_memory(
            chunk,
            memory_data,
            index,
            source="document",
            source_id=doc_name
        )

    return len(chunks)