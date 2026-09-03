"""Typed playground configuration loaded from the environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide search UI settings.

    Environment variables use the ``PLAYGROUND_`` prefix. A local ``.env``
    file is loaded automatically; unknown keys are ignored so existing RAG
    secrets can live in the same file.
    """

    model_config = SettingsConfigDict(
        env_prefix="PLAYGROUND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_title: str = "RAG Retriever Playground"
    host: str = "127.0.0.1"
    port: int = Field(default=5001, ge=1, le=65535)
    reload: bool = True
    index_options: tuple[str, ...] = ("chunk_256", "chunk_512", "chunk_1024")
    default_index: str = "chunk_256"
    default_top_n: int = Field(default=5, ge=1)
    max_top_n: int = Field(default=20, ge=1)
    log_level: str = "INFO"
    log_json: bool = True

    @model_validator(mode="after")
    def coerce_index_and_page_size(self) -> Settings:
        """Keep the default index and top-k inside the configured bounds."""
        if not self.index_options:
            self.index_options = ("chunk_256",)
        if self.default_index not in self.index_options:
            self.default_index = self.index_options[0]
        self.max_top_n = max(self.max_top_n, 1)
        self.default_top_n = min(max(self.default_top_n, 1), self.max_top_n)
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached playground settings."""
    return Settings()
