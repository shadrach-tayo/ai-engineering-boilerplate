"""FastAPI search UI modeled on the Elastic search-tutorial starter."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from markupsafe import Markup

from rag.main import agent_rag, search, vector_rag

load_dotenv()

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_md = (
    MarkdownIt("commonmark", {"html": False, "breaks": True, "linkify": True})
    .enable("table")
    .enable("strikethrough")
)


def render_markdown(text: str | None) -> Markup:
    """Convert markdown to HTML for templates."""
    if not text:
        return Markup("")
    return Markup(_md.render(text))


templates.env.filters["markdown"] = render_markdown

app = FastAPI(title="RAG Retriever Playground")

INDEX_OPTIONS = ("chunk_256", "chunk_512", "chunk_1024")
DEFAULT_INDEX = "chunk_256"

# Last search hits, keyed so result links can render a document page.
_documents: dict[str, dict[str, Any]] = {}
# Vector/agent responses keyed by (query, index, top_k, ask_agent).
_vector_cache: dict[tuple[str, str, int, bool], dict[str, Any]] = {}
# ES pages keyed by (query, index, size, from_).
_es_cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}


def _doc_id(content: str, metadata: dict[str, Any]) -> str:
    raw = f"{metadata.get('source', '')}:{metadata.get('page', '')}:{content[:80]}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _format_results(payload: dict[str, Any], query: str) -> list[dict[str, Any]]:
    """Turn vector_rag output into Elastic-style result rows."""
    contents: list[str] = payload.get("docs") or []
    original_meta: list[dict[str, Any]] = payload.get("original_docs") or []
    reranks: list[dict[str, Any]] = payload.get("rerank") or []

    rows: list[dict[str, Any]] = []
    for i, content in enumerate(contents):
        rerank = reranks[i] if i < len(reranks) else {}
        orig_idx = rerank.get("index", i)
        metadata = original_meta[orig_idx] if orig_idx < len(original_meta) else {}
        doc_id = _doc_id(content, metadata)
        row = {
            "id": doc_id,
            "name": metadata.get("source") or f"Result {i + 1}",
            "content": content,
            "page": metadata.get("page"),
            "source": metadata.get("source"),
            "score": rerank.get("relevance_score"),
            "query": query,
        }
        _documents[doc_id] = row
        rows.append(row)
    return rows


def _format_es_results(payload: dict[str, Any], query: str) -> list[dict[str, Any]]:
    """Turn rag.main.search hits into Elastic-style result rows."""
    rows: list[dict[str, Any]] = []
    for hit in payload.get("results") or []:
        source = hit.get("_source") or {}
        content = source.get("content") or ""
        metadata = {
            "source": source.get("source"),
            "page": source.get("page"),
        }
        doc_id = hit.get("_id") or _doc_id(content, metadata)
        row = {
            "id": doc_id,
            "name": source.get("source") or doc_id,
            "content": content,
            "page": source.get("page"),
            "source": source.get("source"),
            "score": hit.get("_score"),
            "query": query,
        }
        _documents[doc_id] = row
        rows.append(row)
    return rows


def _vector_cache_key(
    query: str, index_name: str, top_n: int, ask_agent: bool
) -> tuple[str, str, int, bool]:
    return (query.casefold(), index_name, top_n, ask_agent)


def _es_cache_key(
    query: str, index_name: str, size: int, from_: int
) -> tuple[str, str, int, int]:
    return (query.casefold(), index_name, size, from_)


def _es_total_count(payload: dict[str, Any], fallback: int) -> int:
    total_meta = payload.get("total")
    if isinstance(total_meta, dict):
        return int(total_meta.get("value", fallback))
    if isinstance(total_meta, int):
        return total_meta
    return fallback


def _remember_documents(*row_lists: list[dict[str, Any]]) -> None:
    """Keep document pages reachable after a cache hit."""
    for rows in row_lists:
        for row in rows:
            _documents[row["id"]] = row


def _empty_context() -> dict[str, Any]:
    return {
        "query": "",
        "index_name": DEFAULT_INDEX,
        "index_options": INDEX_OPTIONS,
        "top_n": 5,
        "results": [],
        "es_results": [],
        "from_": 0,
        "total": 0,
        "es_total": 0,
        "answer": None,
        "error": None,
        "es_error": None,
        "cached": False,
        "ask_agent": False,
        "es_page": 1,
        "es_pages": 1,
        "es_has_prev": False,
        "es_has_next": False,
        "es_prev_from": 0,
        "es_next_from": 0,
    }


def _es_page_meta(from_: int, size: int, total: int) -> dict[str, Any]:
    """Compute previous/next offsets for Elasticsearch pagination."""
    size = max(size, 1)
    from_ = max(from_, 0)
    if total and from_ >= total:
        from_ = max(total - size, 0)
    pages = max((total + size - 1) // size, 1) if total else 1
    page = (from_ // size) + 1
    return {
        "from_": from_,
        "es_page": page,
        "es_pages": pages,
        "es_has_prev": from_ > 0,
        "es_has_next": from_ + size < total,
        "es_prev_from": max(from_ - size, 0),
        "es_next_from": from_ + size,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the empty search page."""
    return templates.TemplateResponse(request, "index.html", _empty_context())


@app.post("/", response_class=HTMLResponse)
async def handle_search(
    request: Request,
    query: str = Form(""),
    index_name: str = Form(DEFAULT_INDEX),
    top_n: int = Form(5),
    from_: int = Form(0),
    ask_agent: str | None = Form(None),
) -> HTMLResponse:
    """Run vector RAG and Elasticsearch search, then render both result lists."""
    query = query.strip()
    index_name = index_name.strip()
    if index_name not in INDEX_OPTIONS:
        index_name = DEFAULT_INDEX
    top_n = max(top_n, 1)
    from_ = max(from_, 0)
    want_agent = bool(ask_agent)
    cached = False
    results: list[dict[str, Any]] = []
    es_results: list[dict[str, Any]] = []
    es_total = 0
    answer = None
    error = None
    es_error = None

    if query:
        vector_key = _vector_cache_key(query, index_name, top_n, want_agent)
        vector_hit = _vector_cache.get(vector_key)
        if vector_hit is not None:
            cached = True
            results = vector_hit["results"]
            answer = vector_hit["answer"]
            error = vector_hit["error"]
            _remember_documents(results)
            logger.info("Vector cache hit for %s", vector_key)
        else:
            try:
                payload = vector_rag(index_name, query, top_n=top_n)
                results = _format_results(payload, query)
                if want_agent:
                    agent = agent_rag(query, store_index=index_name, top_n=top_n)
                    answer = agent.get("content")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Vector search failed")
                error = str(exc)
            _vector_cache[vector_key] = {
                "results": results,
                "answer": answer,
                "error": error,
            }

        es_key = _es_cache_key(query, index_name, top_n, from_)
        es_hit = _es_cache.get(es_key)
        if es_hit is not None:
            es_results = es_hit["es_results"]
            es_total = es_hit["es_total"]
            es_error = es_hit["es_error"]
            _remember_documents(es_results)
            logger.info("ES cache hit for %s", es_key)
        else:
            cached = False
            try:
                es_payload = search(
                    query, store_index=index_name, size=top_n, from_=from_
                )
                es_results = _format_es_results(es_payload, query)
                es_total = _es_total_count(es_payload, len(es_results))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Elasticsearch search failed")
                es_error = str(exc)
            _es_cache[es_key] = {
                "es_results": es_results,
                "es_total": es_total,
                "es_error": es_error,
            }

    page = _es_page_meta(from_, top_n, es_total)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": query,
            "index_name": index_name,
            "index_options": INDEX_OPTIONS,
            "top_n": top_n,
            "results": results,
            "es_results": es_results,
            "from_": page["from_"],
            "total": len(results),
            "es_total": es_total,
            "answer": answer,
            "error": error,
            "es_error": es_error,
            "cached": cached,
            "ask_agent": want_agent,
            "es_page": page["es_page"],
            "es_pages": page["es_pages"],
            "es_has_prev": page["es_has_prev"],
            "es_has_next": page["es_has_next"],
            "es_prev_from": page["es_prev_from"],
            "es_next_from": page["es_next_from"],
        },
    )


@app.get("/document/{doc_id}", response_class=HTMLResponse)
async def get_document(request: Request, doc_id: str) -> HTMLResponse:
    """Render a single retrieved chunk."""
    document = _documents.get(doc_id)
    if document is None:
        return templates.TemplateResponse(
            request,
            "document.html",
            {"title": "Document not found", "content": "", "meta": None},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "document.html",
        {
            "title": document["name"],
            "content": document["content"],
            "meta": document,
        },
    )


def run() -> None:
    """Start the search UI with uvicorn."""
    uvicorn.run("api.app:app", host="127.0.0.1", port=5001, reload=True)


if __name__ == "__main__":
    run()
