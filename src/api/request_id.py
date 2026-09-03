"""HTTP middleware that binds one request id across logs and traces."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag.request_id import reset_request_id, set_request_id
from rag.tracing import trace_span

REQUEST_ID_HEADER = "x-request-id"


def resolve_request_id(request: Request) -> str:
    """Reuse an inbound request id or mint a new one."""
    incoming = request.headers.get(REQUEST_ID_HEADER, "").strip()
    return incoming or uuid.uuid4().hex


@contextmanager
def bind_request_id(request_id: str) -> Iterator[str]:
    """Expose ``request_id`` to stdlib logs, structlog, and Braintrust."""
    token = set_request_id(request_id)
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        yield request_id
    finally:
        reset_request_id(token)
        structlog.contextvars.unbind_contextvars("request_id")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a request id to context, traces, and the response header."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Bind the id for the request lifetime and open a parent trace span."""
        request_id = resolve_request_id(request)
        request.state.request_id = request_id
        with bind_request_id(request_id), trace_span(
            "http.request",
            metadata={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        ):
            response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
