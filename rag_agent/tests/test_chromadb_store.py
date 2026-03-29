from datetime import datetime
from pathlib import Path

import pytest
from backend.app.models.chunk import DocumentChunk
from backend.app.models.document import Document
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.ingestion_service import IngestionService
from backend.app.vector_store.chromadb_store import ChromaDBStore


@pytest.fixture
def store(tmp_path: Path) -> ChromaDBStore:
    """ChromaDB store under a per-test temp dir; pytest removes tmp_path after the test."""
    return ChromaDBStore(
        persist_directory=str(tmp_path / "chroma"),
        dimension=384,
    )


@pytest.fixture
def ingestion_service(
    store: ChromaDBStore,
    embedding_service: EmbeddingService,
) -> IngestionService:
    """Ingestion service backed by ChromaDB store."""
    return IngestionService(store=store, embedding_service=embedding_service)


@pytest.fixture
def sample_doc_path() -> Path:
    """Path to sample text document."""
    return Path(__file__).parent / "fixtures" / "sample_doc.txt"


def test_ingest_document_and_search(
    ingestion_service: IngestionService,
    sample_doc_path: Path,
) -> None:
    """Ingest a document, then search for similar chunks."""
    document_id = ingestion_service.ingest_document(str(sample_doc_path))
    assert document_id is not None

    results = ingestion_service.search_similar_chunks("What is machine learning?", k=2)
    assert len(results) >= 1
    chunk, score = results[0]
    assert "machine learning" in chunk.content.lower() or score > 0


def test_add_get_chunk(
    ingestion_service: IngestionService,
    store: ChromaDBStore,
    sample_doc_path: Path,
) -> None:
    """Add document via ingestion, then get chunk by ID."""
    document_id = ingestion_service.ingest_document(str(sample_doc_path))
    chunks = store.get_chunks_by_document(document_id)
    assert len(chunks) >= 1

    chunk_id = chunks[0].id
    retrieved = store.get_chunk(chunk_id)
    assert retrieved is not None
    assert retrieved.id == chunk_id
    assert retrieved.content == chunks[0].content


def test_get_document_and_list(
    ingestion_service: IngestionService,
    store: ChromaDBStore,
    sample_doc_path: Path,
) -> None:
    """Add document, get metadata, list documents."""
    document_id = ingestion_service.ingest_document(str(sample_doc_path))

    doc = store.get_document(document_id)
    assert doc is not None
    assert doc.id == document_id
    assert doc.file_name == "sample_doc.txt"
    assert doc.chunk_count >= 1

    docs = store.list_documents()
    assert len(docs) >= 1
    assert any(d.id == document_id for d in docs)


def test_search_by_embedding(
    store: ChromaDBStore,
    embedding_service: EmbeddingService,
) -> None:
    """Add chunks manually, search by embedding vector."""
    doc_id = "test_doc_1"
    chunks = [
        DocumentChunk(
            id=f"{doc_id}_chunk_0",
            document_id=doc_id,
            content="Python is a programming language.",
            chunk_index=0,
            created_at=datetime.now(),
        ),
        DocumentChunk(
            id=f"{doc_id}_chunk_1",
            document_id=doc_id,
            content="Machine learning uses neural networks.",
            chunk_index=1,
            created_at=datetime.now(),
        ),
    ]
    texts = [c.content for c in chunks]
    embeddings = embedding_service.embed_batch(texts)

    store.add_chunks(chunks)
    store.add_embeddings([c.id for c in chunks], embeddings)
    store.add_document(
        Document(
            id=doc_id,
            file_path="test.txt",
            file_name="test.txt",
            file_type="txt",
            chunk_count=2,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    )

    query_emb = embedding_service.embed_text("What is Python?")
    results = store.search(query_emb, k=2)
    assert len(results) >= 1
    chunk_ids = [r[0] for r in results]
    assert f"{doc_id}_chunk_0" in chunk_ids


def test_list_chunks_after_ingest(
    ingestion_service: IngestionService,
    store: ChromaDBStore,
    sample_doc_path: Path,
) -> None:
    """list_chunks returns every stored chunk in document order."""
    document_id = ingestion_service.ingest_document(str(sample_doc_path))
    by_doc = store.get_chunks_by_document(document_id)
    all_chunks = store.list_chunks()
    assert len(all_chunks) == len(by_doc)
    assert {c.id for c in all_chunks} == {c.id for c in by_doc}
    listed = ingestion_service.list_all_chunks()
    assert len(listed) == len(by_doc)


def test_list_chunks_empty_store(store: ChromaDBStore) -> None:
    """Fresh store has no chunks."""
    assert store.list_chunks() == []


def test_delete_document(
    ingestion_service: IngestionService,
    store: ChromaDBStore,
    sample_doc_path: Path,
) -> None:
    """Ingest document, delete it, verify store is empty."""
    document_id = ingestion_service.ingest_document(str(sample_doc_path))
    assert store.get_document(document_id) is not None

    store.delete_document(document_id)

    assert store.get_document(document_id) is None
    assert store.get_chunks_by_document(document_id) == []
    assert len(store.list_documents()) == 0
