import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class Embedding(BaseModel):
    """Represents an embedding for a chunk."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str = Field(
        ..., description="The ID of the chunk this embedding belongs to"
    )
    embedding: list[float]
    created_at: datetime | None = None
