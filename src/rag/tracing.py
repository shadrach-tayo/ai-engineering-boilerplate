"""Braintrust tracing for the RAG pipeline, search API, and eval suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

import braintrust
from dotenv import load_dotenv

from rag.request_id import get_request_id

# Patch providers before LangChain OpenAI clients are imported.
# https://www.braintrust.dev/docs/instrument/trace-llm-calls
braintrust.auto_instrument()


@cache
def setup_braintrust() -> None:
    """Start a Braintrust logger so LLM calls appear on the dashboard.

    The SDK reads ``BRAINTRUST_API_KEY`` from the environment or
    ``.env.braintrust``. Skip init when no key is present so CI does not
    prompt for login.
    """
    load_dotenv(".env.braintrust", override=True)
    api_key = os.getenv("BRAINTRUST_API_KEY")
    if not api_key:
        return
    braintrust.init_logger(
        api_key=api_key,
        project=os.getenv("BRAINTRUST_PROJECT", "langgraph-mcp"),
        project_id=os.getenv("BRAINTRUST_PROJECT_ID"),
        async_flush=False,
    )


def flush_braintrust() -> None:
    """Send buffered spans to Braintrust."""
    braintrust.flush()  # type: ignore[no-untyped-call]


def _span_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy kwargs and stamp the current request id onto span metadata."""
    request_id = get_request_id()
    if not request_id:
        return kwargs
    metadata = dict(kwargs.get("metadata") or {})
    metadata.setdefault("request_id", request_id)
    return {**kwargs, "metadata": metadata}


@contextmanager
def trace_span(name: str, **kwargs: Any) -> Iterator[Any]:
    """Open a parent span for a playground or eval request."""
    with braintrust.start_span(name=name, **_span_kwargs(kwargs)) as span:
        yield span
