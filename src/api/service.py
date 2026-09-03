"""Search orchestration, result mapping, and playground caches."""

from __future__ import annotations

import hashlib
from typing import Any

import structlog

from api.schemas import (
    EsCacheEntry,
    EsHit,
    EsSearchPayload,
    RerankItem,
    SearchForm,
    SearchPageContext,
    SearchResult,
    VectorCacheEntry,
    VectorRagPayload,
)
from rag.main import agent_rag, search, vector_rag
from rag.tracing import flush_braintrust, trace_span

log = structlog.get_logger(__name__)


class PlaygroundState:
    """In-memory document pages and search caches for the playground."""

    def __init__(self) -> None:
        """Initialize empty document and cache stores."""
        self.documents: dict[str, SearchResult] = {}
        self.vector_cache: dict[tuple[str, str, int, bool], VectorCacheEntry] = {}
        self.es_cache: dict[tuple[str, str, int, int], EsCacheEntry] = {}

    def remember(self, *row_lists: list[SearchResult]) -> None:
        """Keep document pages reachable after a cache hit."""
        for rows in row_lists:
            for row in rows:
                self.documents[row.id] = row


def document_id(content: str, metadata: dict[str, Any]) -> str:
    """Build a stable id from source, page, and a content prefix."""
    raw = f"{metadata.get('source', '')}:{metadata.get('page', '')}:{content[:80]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _rerank_at(parsed: VectorRagPayload, index: int) -> RerankItem:
    """Return the rerank row at ``index``, or an empty row."""
    if index >= len(parsed.rerank):
        return RerankItem()
    return parsed.rerank[index]


def _metadata_at(parsed: VectorRagPayload, index: int) -> dict[str, Any]:
    """Return original-document metadata at ``index``, or an empty dict."""
    if index >= len(parsed.original_docs):
        return {}
    return parsed.original_docs[index]


def _vector_row(
    index: int, content: str, parsed: VectorRagPayload, query: str
) -> SearchResult:
    """Map one vector hit into a UI result row."""
    rerank = _rerank_at(parsed, index)
    orig_idx = index if rerank.index is None else rerank.index
    metadata = _metadata_at(parsed, orig_idx)
    return SearchResult(
        id=document_id(content, metadata),
        name=metadata.get("source") or f"Result {index + 1}",
        content=content,
        page=metadata.get("page"),
        source=metadata.get("source"),
        score=rerank.relevance_score,
        query=query,
    )


def format_vector_results(payload: dict[str, Any], query: str) -> list[SearchResult]:
    """Turn vector_rag output into Elastic-style result rows."""
    parsed = VectorRagPayload.model_validate(payload)
    return [
        _vector_row(index, content, parsed, query)
        for index, content in enumerate(parsed.docs)
    ]


def format_es_results(payload: dict[str, Any], query: str) -> list[SearchResult]:
    """Turn rag.main.search hits into Elastic-style result rows."""
    parsed = EsSearchPayload.model_validate(payload)
    return [_es_hit_to_result(hit, query) for hit in parsed.results]


def _es_hit_to_result(hit: EsHit, query: str) -> SearchResult:
    """Map one Elasticsearch hit into a UI result row."""
    content = hit.source.content or ""
    metadata = {"source": hit.source.source, "page": hit.source.page}
    doc_id = hit.id or document_id(content, metadata)
    return SearchResult(
        id=doc_id,
        name=hit.source.source or doc_id,
        content=content,
        page=hit.source.page,
        source=hit.source.source,
        score=hit.score,
        query=query,
    )


def es_total_count(payload: dict[str, Any], fallback: int) -> int:
    """Read an ES total, falling back to the current page length."""
    total_meta = payload.get("total")
    if isinstance(total_meta, dict):
        return int(total_meta.get("value", fallback))
    if isinstance(total_meta, int):
        return total_meta
    return fallback


def _agent_answer(form: SearchForm) -> str | None:
    """Ask the agent only when the form checkbox is set."""
    if not form.ask_agent:
        return None
    agent = agent_rag(form.query, store_index=form.index_name, top_n=form.top_n)
    return agent.get("content")


def _fetch_vector(form: SearchForm) -> VectorCacheEntry:
    """Run vector retrieval and optional generation."""
    results: list[SearchResult] = []
    answer = None
    error = None
    try:
        payload = vector_rag(form.index_name, form.query, top_n=form.top_n)
        results = format_vector_results(payload, form.query)
        answer = _agent_answer(form)
    except Exception as exc:  # noqa: BLE001
        log.exception("vector_search_failed")
        error = str(exc)
    return VectorCacheEntry(results=results, answer=answer, error=error)


def _load_vector(form: SearchForm, state: PlaygroundState) -> tuple[VectorCacheEntry, bool]:
    """Return cached vector results, or fetch and store them."""
    key = form.vector_cache_key()
    hit = state.vector_cache.get(key)
    if hit is not None:
        state.remember(hit.results)
        log.info("vector_cache_hit", cache_key=list(key))
        return hit, True
    entry = _fetch_vector(form)
    state.remember(entry.results)
    state.vector_cache[key] = entry
    return entry, False


def _fetch_es(form: SearchForm) -> EsCacheEntry:
    """Run Elasticsearch search for one page."""
    es_results: list[SearchResult] = []
    es_total = 0
    es_error = None
    try:
        payload = search(
            form.query,
            store_index=form.index_name,
            size=form.top_n,
            from_=form.from_,
        )
        es_results = format_es_results(payload, form.query)
        es_total = es_total_count(payload, len(es_results))
    except Exception as exc:  # noqa: BLE001
        log.exception("es_search_failed")
        es_error = str(exc)
    return EsCacheEntry(es_results=es_results, es_total=es_total, es_error=es_error)


def _load_es(form: SearchForm, state: PlaygroundState) -> tuple[EsCacheEntry, bool]:
    """Return a cached Elasticsearch page, or fetch and store it."""
    key = form.es_cache_key()
    hit = state.es_cache.get(key)
    if hit is not None:
        state.remember(hit.es_results)
        log.info("es_cache_hit", cache_key=list(key))
        return hit, True
    entry = _fetch_es(form)
    state.remember(entry.es_results)
    state.es_cache[key] = entry
    return entry, False


def run_search(form: SearchForm, state: PlaygroundState) -> SearchPageContext:
    """Run vector RAG and Elasticsearch search for a form submission."""
    if not form.query:
        return SearchPageContext.from_search(form)

    with trace_span(
        "playground.search",
        input={"query": form.query, "index_name": form.index_name, "top_n": form.top_n},
        metadata={"ask_agent": form.ask_agent, "from": form.from_},
    ):
        vector, vector_cached = _load_vector(form, state)
        es, es_cached = _load_es(form, state)
    flush_braintrust()

    return SearchPageContext.from_search(
        form,
        results=vector.results,
        es_results=es.es_results,
        es_total=es.es_total,
        answer=vector.answer,
        error=vector.error,
        es_error=es.es_error,
        cached=vector_cached and es_cached,
    )
