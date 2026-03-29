"""Pytest fixtures shared across the test suite."""

import pytest
from backend.app.services.embedding_service import EmbeddingService


@pytest.fixture(scope="session")
def embedding_service() -> EmbeddingService:
    """Session-scoped EmbeddingService with pre-loaded models.

    Loads bi-encoder and cross-encoder once for the entire test run.
    """
    service = EmbeddingService()
    service._load_biencoder()
    service._load_cross_encoder()
    return service
