"""Structured JSON logging for the playground API."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

import structlog
from starlette.requests import Request
from starlette.responses import Response

from api.metrics import observe_request

log = structlog.get_logger("api")

RequestHandler = Callable[[Request], Awaitable[Response]]


def _renderer(json_logs: bool) -> structlog.types.Processor:
    """Return a JSON renderer or a colored console renderer."""
    if json_logs:
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer()


def configure_logging(*, json_logs: bool = True, log_level: str = "INFO") -> None:
    """Send structlog and stdlib logs through one JSON (or console) formatter."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _renderer(json_logs),
        ],
        foreign_pre_chain=shared,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


async def log_request(request: Request, call_next: RequestHandler) -> Response:
    """Emit one structured access log line; request id comes from middleware."""
    structlog.contextvars.bind_contextvars(
        method=request.method,
        path=request.url.path,
    )
    started = time.perf_counter()
    response = await call_next(request)
    duration_s = time.perf_counter() - started
    observe_request(request, response.status_code, duration_s)
    log.info(
        "http_request",
        status_code=response.status_code,
        duration_ms=round(duration_s * 1000, 2),
    )
    return response
