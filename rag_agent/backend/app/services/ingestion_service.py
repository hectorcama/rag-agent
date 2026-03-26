import uuid
from datetime import datetime
from pathlib import Path

from unstructured.documents.elements import Element

from backend.app.models.chunk import DocumentChunk
from backend.app.models.document import Document
from backend.app.services.document_processor import DocumentProcessor
from backend.app.services.embedding_service import EmbeddingService
from backend.app.vector_store.base import VectorStoreBackend
from backend.app.vector_store.chromadb_store import ChromaDBStore


class IngestionService:
    """Orchestrates the complete document ingestion pipeline.

    Handles document processing, chunking, embedding generation, and storage
    via a pluggable VectorStoreBackend. Default backend is ChromaDB.
    """

    def __init__(
        self,
        store: VectorStoreBackend | None = None,
        embedding_service: EmbeddingService | None = None,
        data_dir: str = "data",
        biencoder_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """Initialize the ingestion service.

        Args:
            store: Vector store backend. If None, uses ChromaDBStore.
            embedding_service: Optional embedding service. If provided, reused.
            data_dir: Directory for ChromaDB persistence (used when store is None).
            biencoder_model: Bi-encoder model for document/query embeddings.
            cross_encoder_model: Cross-encoder model for reranking.

        """
        self.document_processor = DocumentProcessor()
        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService(
                biencoder_model=biencoder_model,
                cross_encoder_model=cross_encoder_model,
            )
        )
        self.store: VectorStoreBackend = (
            store
            if store is not None
            else ChromaDBStore(
                persist_directory=f"{data_dir}/chroma",
                dimension=self.embedding_service.dimension,
            )
        )

    def ingest_document(
        self,
        file_path: str,
        document_id: str | None = None,
        languages: list[str] | None = None,
        chunking_strategy: str = "by_title",
        max_characters: int = 500,
        new_after_n_chars: int = 450,
        combine_text_under_n_chars: int = 500,
        multipage_sections: bool = True,
        pdf_strategy: str | None = None,
    ) -> str:
        """Ingest a document into the RAG system.

        Processes the document, generates embeddings, and stores via the backend.

        Args:
            file_path: Path to the document file.
            document_id: Optional document ID. If not provided, generates a UUID.
            languages: List of Tesseract language codes for OCR.
            chunking_strategy: Chunking strategy to apply.
            max_characters: Maximum chunk size.
            new_after_n_chars: Soft maximum chunk size.
            combine_text_under_n_chars: Combine small sections under this size.
            multipage_sections: Whether to respect page boundaries.
            pdf_strategy: Partitioning strategy for PDFs.

        Returns:
            Document ID of the ingested document.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file type is not supported.

        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate document ID if not provided
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Detect file type
        file_type = self.document_processor._detect_file_type(file_path)

        # Process document into chunks
        if languages is None:
            languages = ["eng", "fra"]

        chunked_elements: list[Element] = self.document_processor.process_document(
            file_path=file_path,
            languages=languages,
            chunking_strategy=chunking_strategy,
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars,
            combine_text_under_n_chars=combine_text_under_n_chars,
            multipage_sections=multipage_sections,
            pdf_strategy=pdf_strategy,
        )

        # Convert elements to DocumentChunk objects
        chunks: list[DocumentChunk] = []
        chunk_texts: list[str] = []

        for idx, element in enumerate(chunked_elements):
            chunk_id = f"{document_id}_chunk_{idx}"
            content = str(element).strip()

            if not content:  # Skip empty chunks
                continue

            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=content,
                chunk_index=idx,
                created_at=datetime.now(),
                metadata={
                    "element_type": getattr(element, "category", "Unknown"),
                    "file_path": str(file_path),
                },
            )

            chunks.append(chunk)
            chunk_texts.append(content)

        if not chunks:
            raise ValueError("No valid chunks extracted from document")

        # Generate embeddings
        embeddings = self.embedding_service.embed_batch(chunk_texts)

        # Store chunks and document metadata
        self.store.add_chunks(chunks)
        self.store.add_embeddings([chunk.id for chunk in chunks], embeddings)

        document_metadata = Document(
            id=document_id,
            file_path=str(file_path),
            file_name=file_path_obj.name,
            file_type=file_type,
            chunk_count=len(chunks),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.store.add_document(document_metadata)

        return document_id

    def search_similar_chunks(
        self,
        query: str,
        k: int = 5,
        rerank_candidates: int = 20,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search for similar chunks using bi-encoder retrieval + cross-encoder reranking.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            rerank_candidates: Number of candidates to fetch for reranking (default 20).

        Returns:
            List of tuples (chunk, relevance_score) sorted by relevance (higher is better).

        """
        # Bi-encoder: encode query and retrieve candidates from store
        query_embedding = self.embedding_service.embed_text(query)
        fetch_k = min(rerank_candidates, max(k * 4, 20))
        results = self.store.search(query_embedding, k=fetch_k)

        if not results:
            return []

        # Build (chunk_id, content) for reranking
        candidates: list[tuple[str, str]] = []
        for chunk_id, _distance in results:
            chunk = self.store.get_chunk(chunk_id)
            if chunk:
                candidates.append((chunk_id, chunk.content))

        if not candidates:
            return []

        # Cross-encoder: rerank candidates
        reranked = self.embedding_service.rerank(query, candidates, top_k=k)

        # Return (DocumentChunk, relevance_score)
        chunk_results = []
        for chunk_id, _content, score in reranked:
            chunk = self.store.get_chunk(chunk_id)
            if chunk:
                chunk_results.append((chunk, score))

        return chunk_results

    def get_document_chunks(self, document_id: str) -> list[DocumentChunk]:
        """Retrieve all chunks for a document.

        Args:
            document_id: ID of the document.

        Returns:
            List of chunks for the document.

        """
        return self.store.get_chunks_by_document(document_id)

    def delete_document(self, document_id: str) -> None:
        """Delete a document and all its chunks from the system.

        Args:
            document_id: ID of the document to delete.

        """
        self.store.delete_document(document_id)
