# simple-rag — Minimal, production-oriented RAG starter

## Overview

simple-rag demonstrates a compact Retrieval-Augmented Generation (RAG) pipeline:

- Ingest text documents, split into chunks, and store embeddings in a Postgres vector store (pgvector).
- Retrieve relevant chunks and ground a generative model (Gemini) with retrieved context.
- Expose a small FastAPI service to query the knowledge base.

This repo emphasizes clarity and reproducibility for prototyping RAG systems.

## Repository layout

```
ml-rag-project/
├── .env                # Local secrets (gitignored). See .env.reference
├── .env.reference      # Example env values and required keys
├── docker-compose.yml  # Postgres + pgvector (dev)
├── main.py             # FastAPI app + /ask endpoint
├── schemas.py          # Pydantic request/response models
├── database.py         # Vector store (PGVector) and embeddings setup
└── ingest.py           # Simple ingestion script (loads knowledge.txt by default)
```

## What the code does (brief)

- Embeddings: uses GoogleGenerativeAIEmbeddings (Gemini) configured in `database.py`.
- Vector store: PGVector via `langchain_postgres` (collection name: `knowledge_base`).
- Ingestion: `ingest.py` loads a text file, splits into chunks (chunk_size=1000, overlap=100) and stores embeddings.
- API: `main.py` provides `POST /ask` which retrieves top-k (k=4) chunks, builds a grounding prompt, calls Gemini (`gemini-2.0-flash-lite`), and returns an answer with source chunks and similarity scores.

## Prerequisites

- Python 3.10+
- Docker & docker-compose (for local Postgres + pgvector)
- Google API key with Gemini access (set `GOOGLE_API_KEY`)

## Environment variables

Copy `.env.reference` to `.env` and fill values. Minimal variables:

- `GOOGLE_API_KEY` — API key for Gemini (embeddings + generation)
- `DATABASE_URL` — Postgres connection string, e.g.:
  `postgresql://postgres:mysecretpassword@localhost:5432/ai_assistant`

Note: the included `docker-compose.yml` starts Postgres with user `postgres` and password `mysecretpassword`.

## Quickstart (local development)

1. Copy env template and edit secrets

   cp .env.reference .env

   # Edit .env: set GOOGLE_API_KEY and DATABASE_URL

2. Start Postgres + pgvector

   docker-compose up -d

   # Wait until the DB container reports healthy

3. Install Python dependencies

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

4. Ingest documents

   By default `ingest.py` reads `knowledge.txt` in the repo root:

   python ingest.py

   To ingest other sources, replace the loader in `ingest.py` or add a small CLI wrapper.

5. Run the API server

   uvicorn main:app --reload --host 0.0.0.0 --port 8000

## API Usage

Endpoint: POST /ask

Request body:

{
"question": "Your question here"
}

Curl example:

curl -s -X POST "http://localhost:8000/ask" \
 -H "Content-Type: application/json" \
 -d '{"question":"How do I ingest documents?"}'

Response format (from `schemas.py`):

{
"answer": "...",
"sources": [
{ "content": "...", "score": 0.1234, "metadata": {...} }
],
"model_used": "gemini-2.0-flash"
}

## Implementation notes & tuning

- Chunking: current split uses chunk_size=1000, chunk_overlap=100. Tune for your data and embedding context window.
- Retrieval: top-k=4; consider reranking or hybrid filtering for precision.
- Prompting: keep retrieved context concise and instruct the model not to hallucinate.
- Embeddings: Gemini embeddings are used; switching providers requires updating `database.py`.

## Troubleshooting

- No results: confirm `DATABASE_URL` is identical in both `ingest.py` and `main.py`.
- DB connection issues: check `docker ps` and container logs; verify credentials and host/port.
- API errors: inspect FastAPI logs (stack trace) for exceptions; unhandled exceptions return HTTP 500.
- Gemini auth: verify `GOOGLE_API_KEY` and quota/permissions.

## Next steps / improvements

- Add CLI flags for ingestion (dirs, file globs, recursive).
- Batch embeddings and implement rate-limit backoff for LLM calls.
- Add unit & integration tests for ingestion and retrieval.
- Add authentication, request quotas, observability, and Dockerfile for production.
