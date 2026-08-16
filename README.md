# simple-rag — Minimal, production-oriented RAG starter

## Overview

simple-rag is a compact Retrieval-Augmented Generation (RAG) pipeline built to be easy to read, run, and extend rather than to showcase every possible feature. It does three things:

- Ingests text documents, splits them into chunks, and stores embeddings in a Postgres vector store (pgvector).
- Retrieves relevant chunks and grounds a generative model (Gemini) with that context.
- Exposes a small FastAPI service to query the knowledge base over HTTP.

The goal was to strip a RAG system down to its essential moving parts — ingestion, storage, retrieval, generation — so each piece is easy to reason about and swap out independently.

## Repository layout

```
ml-rag-project/
├── .env                # Local secrets (gitignored). See .env.reference
├── .env.reference      # Example env values and required keys
├── docker-compose.yml  # Postgres + pgvector (dev)
├── main.py             # FastAPI app + /ask endpoint
├── schemas.py           # Pydantic request/response models
├── database.py          # Vector store (PGVector) and embeddings setup
└── ingest.py            # Simple ingestion script (loads knowledge.txt by default)
```

## What the code does

- **Embeddings**: uses `GoogleGenerativeAIEmbeddings` (Gemini), configured in `database.py`.
- **Vector store**: PGVector via `langchain_postgres` (collection name: `knowledge_base`).
- **Ingestion**: `ingest.py` loads a text file, splits it into chunks (`chunk_size=1000`, `chunk_overlap=100`), and stores the resulting embeddings.
- **API**: `main.py` exposes `POST /ask`, which retrieves the top-k (`k=4`) most similar chunks, builds a grounding prompt, calls Gemini (`gemini-2.0-flash-lite`), and returns an answer along with the source chunks and their similarity scores.

## Design decisions & tradeoffs

A few choices in this project aren't arbitrary defaults — they're deliberate tradeoffs worth calling out, since they'd change in a different context.

**Postgres + pgvector over a managed vector database.**
I chose pgvector instead of a dedicated vector store like Pinecone or Weaviate because the metadata (source, chunk position, ingestion timestamp) and the vector embeddings live in the same relational database. That keeps the system to a single service to operate locally, avoids a second network hop per query, and means standard SQL can filter or join against vector results. The tradeoff is that pgvector's approximate nearest-neighbor performance doesn't scale as gracefully as a purpose-built vector database once the corpus grows into the millions of chunks — at that point, a managed store with sharding and specialized indexing would likely outperform it.

**Chunk size of 1000 characters with 100-character overlap.**
Larger chunks preserve more surrounding context per retrieval, which matters for documents where meaning depends on nearby sentences (e.g., technical explanations, not isolated facts). The 100-character overlap exists so a sentence split across a chunk boundary isn't stranded without its neighboring context in either chunk. The cost is that larger chunks mean fewer distinct chunks retrieved per query, so more of the token budget for the generation prompt goes to context than to output — the value in this tradeoff would shift for shorter, fact-dense documents like FAQs, where smaller chunks return more precise, less redundant context.

**Top-k = 4 for retrieval.**
Four chunks balances answer grounding against prompt bloat. Fewer chunks risk missing relevant context if the top match isn't a perfect fit; more chunks increase the odds of pulling in irrelevant text that dilutes the prompt and raises token cost. This is a starting point rather than a tuned value — for production use, a reranking step or hybrid keyword + vector search would likely produce a more precise top-k than similarity search alone.

**Gemini for both embeddings and generation.**
Using a single provider (`database.py` for embeddings, `main.py` for generation) simplified initial setup and kept a single API key/auth path. Swapping to a different embeddings provider would require re-embedding the entire corpus, since embedding spaces aren't interchangeable across models — this is the main coupling cost of the current design, and one to weigh before scaling the corpus much further.

## Prerequisites

- Python 3.10+
- Docker & docker-compose (for local Postgres + pgvector)
- Google API key with Gemini access (set `GOOGLE_API_KEY`)

## Environment variables

Copy `.env.reference` to `.env` and fill in the values. At minimum:

- `GOOGLE_API_KEY` — API key for Gemini (embeddings + generation)
- `DATABASE_URL` — Postgres connection string, e.g.:
  `postgresql://postgres:mysecretpassword@localhost:5432/ai_assistant`

Note: the included `docker-compose.yml` starts Postgres with user `postgres` and password `mysecretpassword`. Change these before deploying anywhere beyond local development.

## Quickstart (local development)

1. **Copy the env template and edit secrets**
   ```
   cp .env.reference .env
   # Edit .env: set GOOGLE_API_KEY and DATABASE_URL
   ```

2. **Start Postgres + pgvector**
   ```
   docker-compose up -d
   # Wait until the DB container reports healthy
   ```

3. **Install Python dependencies**
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Ingest documents**
   By default, `ingest.py` reads `knowledge.txt` from the repo root:
   ```
   python ingest.py
   ```
   To ingest other sources, replace the loader in `ingest.py` or add a small CLI wrapper.

5. **Run the API server**
   ```
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## API usage

**Endpoint:** `POST /ask`

**Request body:**
```json
{
  "question": "Your question here"
}
```

**Curl example:**
```
curl -s -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I ingest documents?"}'
```

**Response format** (from `schemas.py`):
```json
{
  "answer": "...",
  "sources": [
    { "content": "...", "score": 0.1234, "metadata": {...} }
  ],
  "model_used": "gemini-2.0-flash"
}
```

## Implementation notes & tuning

- **Chunking**: current split uses `chunk_size=1000`, `chunk_overlap=100`. Tune for your data and embedding context window — see rationale above.
- **Retrieval**: top-k=4; consider reranking or hybrid filtering for precision at scale.
- **Prompting**: keep retrieved context concise and instruct the model not to hallucinate beyond the provided sources.
- **Embeddings**: Gemini embeddings are used; switching providers requires updating `database.py` and re-embedding the corpus.

## Troubleshooting

- **No results**: confirm `DATABASE_URL` is identical in both `ingest.py` and `main.py`.
- **DB connection issues**: check `docker ps` and container logs; verify credentials and host/port.
- **API errors**: inspect FastAPI logs (stack trace) for exceptions; unhandled exceptions return HTTP 500.
- **Gemini auth**: verify `GOOGLE_API_KEY` and quota/permissions.

## Next steps / improvements

- Add CLI flags for ingestion (dirs, file globs, recursive).
- Batch embeddings and implement rate-limit backoff for LLM calls.
- Add unit & integration tests for ingestion and retrieval.
- Add reranking after initial vector retrieval to improve precision.
- Add authentication, request quotas, observability, and a Dockerfile for production.
