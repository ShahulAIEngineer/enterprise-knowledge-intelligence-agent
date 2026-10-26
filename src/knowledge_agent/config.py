
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, environment-based application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "openai/gpt-oss-120b"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    data_dir: Path = Path("data")

    chunk_size: int = Field(default=800, ge=100, le=5000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    top_k: int = Field(default=4, ge=1, le=20)
    max_upload_mb: int = Field(default=10, ge=1, le=100)

    @property
    def groq_key(self) -> str:
        """Return the API key as a string."""
        return self.groq_api_key.get_secret_value().strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
