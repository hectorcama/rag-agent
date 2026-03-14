from abc import ABC, abstractmethod
from typing import Protocol

from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.models.embedding import Embedding


class VectorStoreBackend(Protocol):
    """Protocol for vector store backends (local or remote).

    Implementations must support metadata storage (documents, chunks) and
    vector search (embeddings, similarity search).
    """

    def add_document(self, document: Document) -> None:
        """Add or update a document in the store."""
        ...

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the store (metadata only; use add_embeddings for vectors)."""
        ...

    def add_embeddings(self, chunk_ids: list[str], embeddings: list[Embedding]) -> None:
        """Add embedding vectors for the given chunk IDs."""
        ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by ID."""
        ...

    def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        """Retrieve all chunks for a document, sorted by chunk_index."""
        ...

    def get_document(self, document_id: str) -> Document | None:
        """Retrieve document metadata by ID."""
        ...

    def list_documents(self) -> list[Document]:
        """List all documents in the store."""
        ...

    def search(
        self, query_embedding: Embedding, k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks by embedding.

        Returns:
            List of (chunk_id, distance) sorted by distance ascending.
            Lower distance = more similar.

        """
        ...

    def delete_document(self, document_id: str) -> None:
        """Delete a document and all its chunks (metadata + vectors)."""
        ...

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunk vectors from the index (metadata is deleted via delete_document)."""
        ...


class AbstractVectorStoreBackend(ABC):
    """Abstract base class for vector store backends.

    Use this when you need to enforce implementation via inheritance
    (e.g. for shared helper methods). Otherwise use VectorStoreBackend Protocol.
    """

    @abstractmethod
    def add_document(self, document: Document) -> None:
        """Add or update a document in the store."""
        ...

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the store."""
        ...

    @abstractmethod
    def add_embeddings(self, chunk_ids: list[str], embeddings: list[Embedding]) -> None:
        """Add embedding vectors for the given chunk IDs."""
        ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by ID."""
        ...

    @abstractmethod
    def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        """Retrieve all chunks for a document."""
        ...

    @abstractmethod
    def get_document(self, document_id: str) -> Document | None:
        """Retrieve document metadata by ID."""
        ...

    @abstractmethod
    def list_documents(self) -> list[Document]:
        """List all documents."""
        ...

    @abstractmethod
    def search(
        self, query_embedding: Embedding, k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks by embedding."""
        ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete a document and all its chunks."""
        ...

    @abstractmethod
    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunk vectors from the index."""
        ...
