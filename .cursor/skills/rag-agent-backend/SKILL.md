---
name: rag-agent-backend
description: Guides the RAG-agent Python backend (ingestion, ChromaDB, bi-encoder/cross-encoder retrieval, models). Use when modifying rag_agent/backend, vector stores, embeddings, ingestion, search, or tests for this repository.
---

# RAG backend (rag-agent)

Use this skill to work on this repo without rediscovering the architecture each time.

## Stack and commands

- Package manager: **Poetry** (`pyproject.toml`). Python 3.11–3.12.
- Tests: `poetry run pytest` (see `[tool.pytest.ini_options]`: `pythonpath = ["rag_agent"]`).
- Lint / types: **Ruff**, **mypy** (strict, `ignore_missing_imports`).

## Code map

| Area | Role |
|------|------|
| `backend/app/services/ingestion_service.py` | Orchestration: chunking → embeddings → store; `search_similar_chunks` (retrieval + rerank). |
| `backend/app/services/embedding_service.py` | Bi-encoder `SentenceTransformer`, cross-encoder `CrossEncoder`; lazy loading. |
| `backend/app/services/document_processor.py` | Splitting / partitioning (unstructured). |
| `backend/app/vector_store/chromadb_store.py` | ChromaDB implementation (`PersistentClient`, chunk + document collections, cosine). |
| `backend/app/vector_store/base.py` | Vector store contract. |
| `backend/app/models/chunk.py`, `document.py` | Domain models. |
| `rag_agent/main.py` | CLI / demo entry point if present. |

## Pipeline contract

1. **Ingestion**: file → `DocumentProcessor` → `DocumentChunk` + texts → `EmbeddingService.embed_batch` → `store.add_chunks` then `store.add_embeddings` (order matters: chunks cached before vectors).
2. **Search**: `embed_text(query)` → `store.search(embedding, k=…)` → load chunks → `rerank(query, candidates, top_k=k)`.

Keep embedding dimensions consistent: the store must match `embedding_service.dimension` (e.g. 384 for `all-MiniLM-L6-v2`).

## ChromaDB

- Client: `chromadb.PersistentClient` with `Settings(anonymized_telemetry=False)`.
- Metadata: primitives + serialized JSON for complex fields (`metadata_json`).

## Tests

- `rag_agent/tests/conftest.py`: **`embedding_service`** fixture with **session** scope — reuse it to avoid reloading models on every test.
- `IngestionService` accepts an injected **`embedding_service`** for tests with an isolated store.

## Project principles

- Change only what the task requires; match existing style (imports, types).
- Do not add dependencies without justification.
- For any retrieval / store change: review `base.py`, `chromadb_store.py`, and related tests.
