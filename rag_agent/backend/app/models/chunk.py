import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Chunk of a document for vector storage and retrieval."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(
        ..., description="The ID of the document this chunk belongs to"
    )
    content: str
    chunk_index: int
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
