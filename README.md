# FinDoc Assist

FinDoc Assist is a production-style Retrieval-Augmented Generation (RAG) document QA system built for a Junior AI Engineer interview project.

## Features

- FastAPI backend with REST endpoints
- Simple server-rendered web UI
- PDF and text ingestion
- Chunking, embeddings, and vector retrieval
- OpenAI-compatible LLM adapter with Groq/OpenRouter-style support
- PII masking and out-of-scope / prompt injection guardrails
- SQLite metadata store with persistent local uploads and vector index
- Summary endpoint for document-level summaries

## Quick Start

1. Create a virtual environment and install dependencies.
2. Optionally install `sentence-transformers` and `faiss-cpu` for stronger retrieval.
3. Set environment variables from `.env.example`.
4. Start the app:

```bash
uvicorn app.main:app --reload
```

5. Open `http://127.0.0.1:8000`.

## Environment

The app is designed to work even when no provider API key is present:

- Embeddings fall back to a deterministic local hashing embedder if `sentence-transformers` is unavailable.
- LLM answers fall back to a grounded extractive responder if no upstream provider is configured.

That keeps the project runnable for interview demos while still exposing a production-style provider abstraction.
