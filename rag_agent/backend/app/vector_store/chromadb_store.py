import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.app.models.chunk import DocumentChunk
from backend.app.models.document import Document

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


def _chunk_to_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    """Convert DocumentChunk to ChromaDB-compatible metadata (primitives only)."""
    meta: dict[str, Any] = {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
    }
    if chunk.created_at is not None:
        meta["created_at"] = chunk.created_at.isoformat()
    if chunk.metadata is not None:
        meta["metadata_json"] = json.dumps(chunk.metadata)
    return meta


def _metadata_to_chunk(
    chunk_id: str, document: str, metadata: dict[str, Any] | None
) -> DocumentChunk:
    """Build DocumentChunk from ChromaDB document + metadata."""
    if metadata is None:
        metadata = {}
    metadata_json = metadata.get("metadata_json")
    created_at_str = metadata.get("created_at")
    return DocumentChunk(
        id=chunk_id,
        document_id=metadata.get("document_id", ""),
        content=document or "",
        chunk_index=int(metadata.get("chunk_index", 0)),
        metadata=json.loads(metadata_json) if metadata_json else None,
        created_at=datetime.fromisoformat(created_at_str) if created_at_str else None,
    )


def _document_to_metadata(doc: Document) -> dict[str, Any]:
    """Convert Document to ChromaDB-compatible metadata."""
    meta: dict[str, Any] = {
        "file_path": doc.file_path,
        "file_name": doc.file_name,
        "file_type": doc.file_type,
        "chunk_count": doc.chunk_count,
    }
    if doc.created_at is not None:
        meta["created_at"] = doc.created_at.isoformat()
    if doc.updated_at is not None:
        meta["updated_at"] = doc.updated_at.isoformat()
    if doc.metadata is not None:
        meta["metadata_json"] = json.dumps(doc.metadata)
    return meta


def _metadata_to_document(doc_id: str, metadata: dict[str, Any] | None) -> Document:
    """Build DocumentMetadata from ChromaDB metadata."""
    if metadata is None:
        metadata = {}
    metadata_json = metadata.get("metadata_json")
    created_at_str = metadata.get("created_at")
    updated_at_str = metadata.get("updated_at")
    return Document(
        id=doc_id,
        file_path=metadata.get("file_path", ""),
        file_name=metadata.get("file_name", ""),
        file_type=metadata.get("file_type", ""),
        chunk_count=int(metadata.get("chunk_count", 0)),
        created_at=datetime.fromisoformat(created_at_str) if created_at_str else None,
        updated_at=datetime.fromisoformat(updated_at_str) if updated_at_str else None,
        metadata=json.loads(metadata_json) if metadata_json else None,
    )


class ChromaDBStore:
    """Vector store backend using ChromaDB.

    Uses two collections: one for chunks (vectors + content + metadata),
    one for document metadata. Cosine similarity for semantic search.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "rag_chunks",
        dimension: int = 384,
    ):
        if chromadb is None:
            raise ImportError(
                "ChromaDB is not installed. Install it with: pip install chromadb"
            )

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.dimension = dimension
        self._dummy_embedding = [0.0] * dimension

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        self._chunks = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._documents = self._client.get_or_create_collection(
            name=f"{collection_name}_documents",
            metadata={"hnsw:space": "cosine"},
        )

        self._pending_chunks: dict[str, DocumentChunk] = {}

    def add_document(self, document: Document) -> None:
        """Add or update document metadata in the documents collection."""
        self._documents.upsert(
            ids=[document.id],
            embeddings=[self._dummy_embedding],
            metadatas=[_document_to_metadata(document)],
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Cache chunks for add_embeddings (metadata stored with embeddings)."""
        for chunk in chunks:
            self._pending_chunks[chunk.id] = chunk

    def add_embeddings(
        self, chunk_ids: list[str], embeddings: list[list[float]]
    ) -> None:
        """Add chunk embeddings, documents, and metadata to ChromaDB."""
        if not chunk_ids or not embeddings:
            return
        if len(chunk_ids) != len(embeddings):
            raise ValueError("chunk_ids and embeddings must have the same length")

        ids_list: list[str] = []
        docs_list: list[str] = []
        metas_list: list[dict[str, Any]] = []
        for chunk_id in chunk_ids:
            chunk = self._pending_chunks.pop(chunk_id, None)
            if chunk is None:
                raise ValueError(
                    f"Chunk {chunk_id} not found in cache; call add_chunks first"
                )
            ids_list.append(chunk_id)
            docs_list.append(chunk.content)
            metas_list.append(_chunk_to_metadata(chunk))

        self._chunks.add(
            ids=ids_list,
            embeddings=embeddings,
            documents=docs_list,
            metadatas=metas_list,
        )

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve chunk by ID from ChromaDB."""
        result = self._chunks.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        doc = result["documents"][0] if result["documents"] else ""
        meta = result["metadatas"][0] if result["metadatas"] else None
        return _metadata_to_chunk(chunk_id, doc, meta)

    def get_chunks_by_document(self, document_id: str) -> list[DocumentChunk]:
        """Retrieve all chunks for a document, sorted by chunk_index."""
        result = self._chunks.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return []
        chunks = [
            _metadata_to_chunk(cid, doc, meta)
            for cid, doc, meta in zip(
                result["ids"],
                result["documents"] or [""] * len(result["ids"]),
                result["metadatas"] or [{}] * len(result["ids"]),
                strict=True,
            )
        ]
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def get_document(self, document_id: str) -> Document | None:
        """Retrieve document metadata by ID."""
        result = self._documents.get(
            ids=[document_id],
            include=["metadatas"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0] if result["metadatas"] else None
        return _metadata_to_document(document_id, meta)

    def list_documents(self) -> list[Document]:
        """List all documents in the store."""
        result = self._documents.get(include=["metadatas"])
        if not result["ids"]:
            return []
        return [
            _metadata_to_document(doc_id, meta)
            for doc_id, meta in zip(
                result["ids"],
                result["metadatas"] or [{}] * len(result["ids"]),
                strict=True,
            )
        ]

    def list_chunks(self) -> list[DocumentChunk]:
        """List all indexed chunks, sorted by document_id then chunk_index."""
        n = self._chunks.count()
        if n == 0:
            return []
        result = self._chunks.get(
            include=["documents", "metadatas"],
            limit=n,
        )
        if not result["ids"]:
            return []
        chunks = [
            _metadata_to_chunk(cid, doc, meta)
            for cid, doc, meta in zip(
                result["ids"],
                result["documents"] or [""] * len(result["ids"]),
                result["metadatas"] or [{}] * len(result["ids"]),
                strict=True,
            )
        ]
        chunks.sort(key=lambda c: (c.document_id, c.chunk_index))
        return chunks

    def search(
        self, query_embedding: list[float], k: int = 5
    ) -> list[tuple[str, float]]:
        """Search for similar chunks by embedding.

        Returns (chunk_id, distance).
        """
        result = self._chunks.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._chunks.count()),
            include=["distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        ids = result["ids"][0]
        distances = result["distances"][0] if result["distances"] else [0.0] * len(ids)
        return list(zip(ids, distances, strict=True))

    def delete_document(self, document_id: str) -> None:
        """Delete document and all its chunks."""
        self._chunks.delete(where={"document_id": document_id})
        self._documents.delete(ids=[document_id])

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunk vectors from the index."""
        if chunk_ids:
            self._chunks.delete(ids=chunk_ids)
