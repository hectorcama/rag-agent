from backend.app.settings import apply_runtime_env, get_settings

apply_runtime_env()

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


def _resolve_hf_token(explicit: str | None) -> str | None:
    """Token for Hugging Face Hub (gated/private models).

    Reads settings if not passed.
    """
    if explicit is not None:
        stripped = explicit.strip()
        return stripped or None
    return get_settings().hf_token


class EmbeddingService:
    """Service for generating embeddings and reranking.

    Uses a bi-encoder (SentenceTransformer) for encoding documents and queries into
    embeddings. Uses a cross-encoder for reranking candidate results to improve
    retrieval accuracy.
    """

    def __init__(
        self,
        biencoder_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        hf_token: str | None = None,
    ):
        """Initialize the embedding service.

        Args:
            biencoder_model: Name of the bi-encoder model for embeddings.
            cross_encoder_model: Name of the cross-encoder model for reranking.
            hf_token: Hugging Face Hub token. If None, uses ``HF_TOKEN`` or
                ``HUGGING_FACE_HUB_TOKEN`` from the environment (after ``.env``).
        """
        self.biencoder_model_name = biencoder_model
        self.cross_encoder_model_name = cross_encoder_model
        self._hf_token = _resolve_hf_token(hf_token)
        self._biencoder: SentenceTransformer | None = None
        self._cross_encoder: CrossEncoder | None = None
        self._dimension = 384  # Dimension for all-MiniLM-L6-v2

    def _hub_kwargs(self) -> dict[str, str]:
        if self._hf_token is None:
            return {}
        return {"token": self._hf_token}

    def _load_biencoder(self) -> None:
        """Lazy load the bi-encoder on first use."""
        if self._biencoder is None:
            self._biencoder = SentenceTransformer(
                self.biencoder_model_name,
                **self._hub_kwargs(),
            )

    def _load_cross_encoder(self) -> None:
        """Lazy load the cross-encoder on first use."""
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(
                self.cross_encoder_model_name,
                **self._hub_kwargs(),
            )

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text using the bi-encoder.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        self._load_biencoder()
        if self._biencoder is None:
            raise RuntimeError("Bi-encoder failed to load")
        embedding = self._biencoder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for a batch of texts using the bi-encoder.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in each batch.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        self._load_biencoder()
        if self._biencoder is None:
            raise RuntimeError("Bi-encoder failed to load")

        embeddings = self._biencoder.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, str, float]]:
        """Rerank candidates using the cross-encoder.

        Args:
            query: The search query.
            candidates: List of (chunk_id, content) tuples to rerank.
            top_k: Number of top results to return. If None, return all sorted.

        Returns:
            List of (chunk_id, content, score) sorted by score descending.
        """
        if not candidates:
            return []

        self._load_cross_encoder()
        if self._cross_encoder is None:
            raise RuntimeError("Cross-encoder failed to load")

        pairs = [[query, content] for _chunk_id, content in candidates]
        scores = self._cross_encoder.predict(pairs)

        scored = [
            (chunk_id, content, float(score))
            for (chunk_id, content), score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda x: x[2], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored
