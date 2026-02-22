"""Test demonstrating the bi-encoder + cross-encoder pipeline.

Bi-encoder encodes documents and queries independently for fast retrieval.
Cross-encoder reranks candidates for higher accuracy.
"""

from backend.app.services.embedding_service import EmbeddingService

MAX_MODEL_ENCODER_LENGTH = 256
MAX_MODEL_CROSS_ENCODER_LENGTH = 512
MAX_CHARS_FOR_DOCUMENT_CROSS_ENCODING = 2000
MAX_CHARS_FOR_TEXT_ENCODING = 1200

# Sample documents simulating a small corpus
SAMPLE_DOCS = [
    (
        "doc_1",
        "Python is a programming language known for its simplicity and readability.",
    ),
    ("doc_2", "Machine learning models can be trained to recognize patterns in data."),
    ("doc_3", "The quick brown fox jumps over the lazy dog."),
    (
        "doc_4",
        "Natural language processing enables computers to understand human text.",
    ),
    ("doc_5", "JavaScript is used for building interactive web applications."),
]


def test_biencoder_encodes_documents_and_queries() -> None:
    """Bi-encoder produces embeddings for documents and queries."""
    service = EmbeddingService()

    doc_embeddings = service.embed_batch([content for _id, content in SAMPLE_DOCS])
    assert len(doc_embeddings) == len(SAMPLE_DOCS)
    assert len(doc_embeddings[0]) == service.dimension

    query = "What programming language is easy to read?"
    query_embedding = service.embed_text(query)
    assert len(query_embedding) == service.dimension


def test_cross_encoder_reranks_candidates() -> None:
    """Cross-encoder reranks (query, document) pairs by relevance."""
    service = EmbeddingService()

    query = "What is Python used for?"
    candidates = [(doc_id, content) for doc_id, content in SAMPLE_DOCS]

    reranked = service.rerank(query, candidates, top_k=3)

    expected_top_k = 3
    assert len(reranked) == expected_top_k
    # Results should be (chunk_id, content, score)
    for item in reranked:
        assert len(item) == expected_top_k
        assert isinstance(item[0], str)
        assert isinstance(item[1], str)
        assert isinstance(item[2], float)
    # Python doc should rank high for this query
    top_contents = [r[1] for r in reranked]
    assert any("python" in c.lower() for c in top_contents)


def test_full_pipeline_biencoder_then_rerank() -> None:
    """Full pipeline: bi-encoder retrieval simulation + cross-encoder reranking.

    In a real system, the bi-encoder would retrieve from FAISS. Here we use
    all docs as candidates and show that reranking improves ordering.
    """
    service = EmbeddingService()

    query = "How do computers understand human language?"
    candidates = [(doc_id, content) for doc_id, content in SAMPLE_DOCS]

    # Simulate bi-encoder: we'd use cosine similarity on embeddings to get top-N
    # Here we pass all candidates to cross-encoder
    reranked = service.rerank(query, candidates, top_k=2)

    # NLP-related doc should be top
    top_content = reranked[0][1]
    assert "language" in top_content.lower() or "text" in top_content.lower()


def test_rerank_empty_candidates_returns_empty() -> None:
    """Rerank with no candidates returns empty list."""
    service = EmbeddingService()
    result = service.rerank("query", [], top_k=5)
    assert result == []


def test_chunk_exceeding_encoder_limit_truncates_silently() -> None:
    """When chunk size exceeds encoder limit, sentence-transformers truncates silently.

    Bi-encoder (all-MiniLM-L6-v2): 256 tokens max.
    ~4 chars/token -> ~1200+ chars exceeds limit.
    """
    service = EmbeddingService()
    service._load_biencoder()
    assert service._biencoder is not None

    # Build text ~1500 chars (~375 tokens) - exceeds bi-encoder's 256 limit
    long_chunk = " ".join(
        [
            "This is a sample sentence that we repeat to create a long document."
            " It contains many words and will exceed the typical token limit."
        ]
        * 15
    )
    assert len(long_chunk) > MAX_CHARS_FOR_TEXT_ENCODING

    # Verify token count exceeds limit (optional check)
    tokenizer = (
        getattr(service._biencoder, "tokenizer", None)
        or service._biencoder[0].tokenizer
    )
    tokens = tokenizer(long_chunk, return_tensors=None, add_special_tokens=True)
    token_count = len(tokens["input_ids"])
    assert token_count > MAX_MODEL_ENCODER_LENGTH, (
        "Test setup: long_chunk should exceed 256 tokens"
    )

    # Encode: should NOT raise, truncation happens silently
    embedding = service.embed_text(long_chunk)
    assert len(embedding) == service.dimension

    # Batch encode: same behavior
    embeddings = service.embed_batch([long_chunk, "short"])
    expected_embeddings_count = 2
    assert len(embeddings) == expected_embeddings_count
    assert len(embeddings[0]) == service.dimension


def test_cross_encoder_long_document_truncates_silently() -> None:
    """Cross-encoder (512 tokens) also truncates long documents silently."""
    service = EmbeddingService()

    # Document ~2500 chars (~625 tokens) - exceeds 512
    long_doc = " ".join(
        ["A relevant passage about machine learning and neural networks."] * 35
    )
    assert len(long_doc) > MAX_CHARS_FOR_DOCUMENT_CROSS_ENCODING

    candidates = [("chunk_1", long_doc)]
    result = service.rerank("What is machine learning?", candidates, top_k=1)

    # Should complete without error, return valid score
    assert len(result) == 1
    assert isinstance(result[0][2], float)
