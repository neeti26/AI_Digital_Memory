AI Digital Memory

A modular, retrieval-augmented AI memory system designed to provide persistent contextual awareness for AI agents and local LLM workflows.

AI Digital Memory enables document ingestion, embedding generation, vector indexing, and intelligent retrieval to simulate long-term memory for AI systems.

Overview

Modern LLMs lack persistent memory across sessions. This project implements a structured memory layer that allows:

Document ingestion (PDF / text)

Embedding generation

Vector storage & indexing

Retrieval-Augmented Generation (RAG)

CLI-based interaction

Local LLM experimentation

The goal is to simulate a long-term contextual memory system that can be extended into:

Personal AI assistants

Research agents

Knowledge retrieval systems

Autonomous task agents

Architecture

High-level flow:

Document Ingestion

Load PDF/text documents

Chunk content

Clean & preprocess

Embedding Layer

Convert chunks into vector embeddings

Store metadata

Vector Index

FAISS or similar vector database

Efficient similarity search

Retrieval Layer

Query embedding generation

Nearest-neighbor search

Context injection into LLM

LLM Interaction

Local or API-based model

Augmented response generation

Project Structure
AI_Digital_Memory/
│
├── main.py                # Core orchestration logic
├── memory_engine.py       # Memory system abstraction
├── document_ingest.py     # Document loading + chunking
├── pdf_rag.py             # Retrieval-Augmented Generation pipeline
├── local_llm.py           # Local model experimentation
├── brain_cli.py           # Command-line interface
│
├── *.faiss                # Vector index files
├── *.pkl / *.json         # Metadata & serialized memory
│
└── requirements.txt       # Dependencies
Installation
1. Clone Repository
git clone https://github.com/neeti26/AI_Digital_Memory.git
cd AI_Digital_Memory
2. Create Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate

Mac/Linux:

python -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Usage
Ingest Documents

Modify the input path inside document_ingest.py, then run:

python document_ingest.py

This will:

Load documents

Chunk text

Generate embeddings

Build or update vector index

Query the Memory System
python brain_cli.py query "Your question here"

This will:

Convert query to embedding

Retrieve relevant chunks

Pass context to LLM

Generate response

Run Full Pipeline
python main.py

Runs ingestion + retrieval workflow.

Example Workflow

Add research PDFs to /docs

Run ingestion

Query memory:

python brain_cli.py query "Summarize chapter 2"

System retrieves semantically similar chunks and generates a context-aware answer.

Design Principles

Modular architecture

Vector-database driven retrieval

LLM-agnostic (can swap models)

CLI-first experimentation

Extensible memory abstraction

Future Improvements

Web API (FastAPI)

Persistent database backend

Streaming responses

GUI dashboard

Multi-agent memory sharing

Cloud deployment

Memory decay / scoring system

Semantic compression
