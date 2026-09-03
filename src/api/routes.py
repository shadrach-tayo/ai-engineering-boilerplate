"""HTTP routes for the retriever playground."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from markupsafe import Markup

from api.metrics import metrics_response
from api.schemas import DocumentPageContext, SearchForm, SearchPageContext
from api.service import PlaygroundState, run_search
from api.settings import Settings, get_settings

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

router = APIRouter(tags=["playground"])


def get_store(request: Request) -> PlaygroundState:
    """Return the process-wide playground caches from app state."""
    return request.app.state.store


Store = Annotated[PlaygroundState, Depends(get_store)]
Config = Annotated[Settings, Depends(get_settings)]


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Expose Prometheus counters and latency histograms."""
    return metrics_response()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, settings: Config) -> HTMLResponse:
    """Render the empty search page."""
    context = SearchPageContext(
        index_name=settings.default_index,
        index_options=settings.index_options,
        top_n=settings.default_top_n,
    )
    return templates.TemplateResponse(request, "index.html", context.model_dump())


@router.post("/", response_class=HTMLResponse)
async def handle_search(
    request: Request,
    form: Annotated[SearchForm, Form()],
    store: Store,
) -> HTMLResponse:
    """Run vector RAG and Elasticsearch search, then render both result lists."""
    context = run_search(form, store)
    return templates.TemplateResponse(request, "index.html", context.model_dump())


@router.get("/document/{doc_id}", response_class=HTMLResponse)
async def get_document(
    request: Request,
    doc_id: str,
    store: Store,
) -> HTMLResponse:
    """Render a single retrieved chunk."""
    document = store.documents.get(doc_id)
    if document is None:
        return templates.TemplateResponse(
            request,
            "document.html",
            DocumentPageContext(title="Document not found").model_dump(),
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return templates.TemplateResponse(
        request,
        "document.html",
        DocumentPageContext(
            title=document.name,
            content=document.content,
            meta=document,
        ).model_dump(),
    )
