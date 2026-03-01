import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document with metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    file_path: str
    file_type: str
    chunk_count: int = 0
    created_at: datetime | None = None
    metadata: dict[str, Any] | None = None
