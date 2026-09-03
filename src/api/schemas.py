"""Pydantic models for the retriever playground."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.settings import get_settings


def parse_int(value: object, default: int) -> int:
    """Parse an int from form input, returning ``default`` when blank or invalid."""
    if value in (None, ""):
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def clamp_offset(from_: int, size: int, total: int) -> int:
    """Keep an Elasticsearch offset on a valid page."""
    from_ = max(from_, 0)
    if total and from_ >= total:
        return max(total - size, 0)
    return from_


def page_count(size: int, total: int) -> int:
    """Return how many pages ``total`` hits fill at ``size`` per page."""
    if not total:
        return 1
    return max((total + size - 1) // size, 1)


class SearchForm(BaseModel):
    """Submitted search form fields."""

    query: str = Field(default="", description="Search query")
    index_name: str = Field(
        default_factory=lambda: get_settings().default_index,
        description="Vector/ES index name",
    )
    top_n: int = Field(
        default_factory=lambda: get_settings().default_top_n,
        ge=1,
        description="Result page size",
    )
    from_: int = Field(default=0, ge=0, description="Elasticsearch result offset")
    ask_agent: bool = Field(default=False, description="Also generate an agent answer")

    @field_validator("query", "index_name", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        """Trim whitespace from incoming form strings."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("index_name")
    @classmethod
    def coerce_index(cls, value: str) -> str:
        """Fall back to the default index when the form value is unknown."""
        settings = get_settings()
        if value not in settings.index_options:
            return settings.default_index
        return value

    @field_validator("top_n", mode="before")
    @classmethod
    def coerce_top_n(cls, value: object) -> object:
        """Clamp page size so HTML form posts do not 422."""
        settings = get_settings()
        return min(max(parse_int(value, settings.default_top_n), 1), settings.max_top_n)

    @field_validator("from_", mode="before")
    @classmethod
    def coerce_from(cls, value: object) -> object:
        """Treat blank or invalid offsets as the first page."""
        return max(parse_int(value, 0), 0)

    def vector_cache_key(self) -> tuple[str, str, int, bool]:
        """Return the in-memory cache key for vector/agent results."""
        return (self.query.casefold(), self.index_name, self.top_n, self.ask_agent)

    def es_cache_key(self) -> tuple[str, str, int, int]:
        """Return the in-memory cache key for Elasticsearch pages."""
        return (self.query.casefold(), self.index_name, self.top_n, self.from_)


class SearchResult(BaseModel):
    """One retrieved chunk shown in the UI and document page."""

    id: str
    name: str
    content: str
    page: Any = None
    source: str | None = None
    score: float | None = None
    query: str = ""


class RerankItem(BaseModel):
    """Cohere rerank row from the vector pipeline."""

    model_config = ConfigDict(extra="allow")

    index: int | None = None
    relevance_score: float | None = None


class VectorRagPayload(BaseModel):
    """Shape returned by ``rag.main.vector_rag``."""

    model_config = ConfigDict(extra="allow")

    docs: list[str] = Field(default_factory=list)
    original_docs: list[dict[str, Any]] = Field(default_factory=list)
    rerank: list[RerankItem] = Field(default_factory=list)

    @field_validator("rerank", mode="before")
    @classmethod
    def empty_rerank(cls, value: object) -> object:
        """Treat a missing rerank list as empty."""
        return value or []


class EsHitSource(BaseModel):
    """Elasticsearch ``_source`` document fields used by the UI."""

    model_config = ConfigDict(extra="allow")

    content: str = ""
    source: str | None = None
    page: Any = None


class EsHit(BaseModel):
    """One Elasticsearch hit from ``rag.main.search``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str | None = Field(default=None, alias="_id")
    score: float | None = Field(default=None, alias="_score")
    source: EsHitSource = Field(default_factory=EsHitSource, alias="_source")


class EsSearchPayload(BaseModel):
    """Shape returned by ``rag.main.search``."""

    model_config = ConfigDict(extra="allow")

    results: list[EsHit] = Field(default_factory=list)
    total: int = 0

    @field_validator("total", mode="before")
    @classmethod
    def unwrap_total(cls, value: object) -> object:
        """Accept ES ``{"value": n}`` totals or a bare integer."""
        if isinstance(value, dict):
            return value.get("value", 0)
        if value is None:
            return 0
        return value


class VectorCacheEntry(BaseModel):
    """Cached vector retrieval (and optional agent) response."""

    results: list[SearchResult] = Field(default_factory=list)
    answer: str | None = None
    error: str | None = None


class EsCacheEntry(BaseModel):
    """Cached Elasticsearch page."""

    es_results: list[SearchResult] = Field(default_factory=list)
    es_total: int = 0
    es_error: str | None = None


class EsPagination(BaseModel):
    """Previous/next offsets for Elasticsearch result pages."""

    from_: int = 0
    es_page: int = 1
    es_pages: int = 1
    es_has_prev: bool = False
    es_has_next: bool = False
    es_prev_from: int = 0
    es_next_from: int = 0

    @classmethod
    def from_offset(cls, from_: int, size: int, total: int) -> EsPagination:
        """Compute pagination from an offset, page size, and hit count."""
        size = max(size, 1)
        from_ = clamp_offset(from_, size, total)
        pages = page_count(size, total)
        page = (from_ // size) + 1
        return cls(
            from_=from_,
            es_page=page,
            es_pages=pages,
            es_has_prev=from_ > 0,
            es_has_next=from_ + size < total,
            es_prev_from=max(from_ - size, 0),
            es_next_from=from_ + size,
        )


class SearchPageContext(BaseModel):
    """Jinja context for the search page."""

    query: str = ""
    index_name: str = Field(default_factory=lambda: get_settings().default_index)
    index_options: tuple[str, ...] = Field(default_factory=lambda: get_settings().index_options)
    top_n: int = Field(default_factory=lambda: get_settings().default_top_n)
    results: list[SearchResult] = Field(default_factory=list)
    es_results: list[SearchResult] = Field(default_factory=list)
    from_: int = 0
    total: int = 0
    es_total: int = 0
    answer: str | None = None
    error: str | None = None
    es_error: str | None = None
    cached: bool = False
    ask_agent: bool = False
    es_page: int = 1
    es_pages: int = 1
    es_has_prev: bool = False
    es_has_next: bool = False
    es_prev_from: int = 0
    es_next_from: int = 0

    @classmethod
    def from_search(
        cls,
        form: SearchForm,
        *,
        results: list[SearchResult] | None = None,
        es_results: list[SearchResult] | None = None,
        es_total: int = 0,
        answer: str | None = None,
        error: str | None = None,
        es_error: str | None = None,
        cached: bool = False,
    ) -> SearchPageContext:
        """Build template context from a form submission and search output."""
        results = results or []
        es_results = es_results or []
        page = EsPagination.from_offset(form.from_, form.top_n, es_total)
        return cls(
            query=form.query,
            index_name=form.index_name,
            index_options=get_settings().index_options,
            top_n=form.top_n,
            results=results,
            es_results=es_results,
            from_=page.from_,
            total=len(results),
            es_total=es_total,
            answer=answer,
            error=error,
            es_error=es_error,
            cached=cached,
            ask_agent=form.ask_agent,
            es_page=page.es_page,
            es_pages=page.es_pages,
            es_has_prev=page.es_has_prev,
            es_has_next=page.es_has_next,
            es_prev_from=page.es_prev_from,
            es_next_from=page.es_next_from,
        )


class DocumentPageContext(BaseModel):
    """Jinja context for a single retrieved document."""

    title: str
    content: str = ""
    meta: SearchResult | None = None
