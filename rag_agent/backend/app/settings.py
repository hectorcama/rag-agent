import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _REPO_ROOT / ".env"
# Avoid passing a missing path to pydantic-settings (clean CI without ``.env``).
_DOTENV_FILES: tuple[Path, ...] | None = (_ENV_FILE,) if _ENV_FILE.is_file() else None


class Settings(BaseSettings):
    """Runtime configuration.

    (env vars + optional ``.env`` at repository root).

    """

    model_config = SettingsConfigDict(
        env_file=_DOTENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Unstructured package analytics.
    scarf_no_analytics: str = Field(default="true")
    do_not_track: str = Field(default="true")

    # Hugging face token to access models.
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    )

    @field_validator("hf_token", mode="before")
    @classmethod
    def strip_hf_token(cls, value: object) -> str | None:
        """Normalize empty or whitespace-only token to ``None``."""
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value).strip() or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


def apply_runtime_env() -> None:
    """Push settings into ``os.environ``.

    Done before imports that read the process env.

    Uses ``setdefault`` so variables already set
    (e.g. GitHub Actions secrets) win.
    Telemetry vars satisfy ``unstructured``;
    HF vars satisfy Hub clients that only inspect
    the environment.
    """
    settings = get_settings()
    os.environ.setdefault("SCARF_NO_ANALYTICS", settings.scarf_no_analytics)
    os.environ.setdefault("DO_NOT_TRACK", settings.do_not_track)
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", settings.hf_token)
