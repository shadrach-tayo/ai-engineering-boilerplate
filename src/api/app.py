"""FastAPI search UI modeled on the Elastic search-tutorial starter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.logging import configure_logging, log_request
from api.request_id import RequestIdMiddleware
from api.settings import get_settings

settings = get_settings()
configure_logging(json_logs=settings.log_json, log_level=settings.log_level)

from api.routes import router  # noqa: E402
from api.service import PlaygroundState  # noqa: E402
from rag.tracing import flush_braintrust, setup_braintrust  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Enable Braintrust tracing for the playground and flush on shutdown."""
    setup_braintrust()
    yield
    flush_braintrust()


app = FastAPI(title=settings.app_title, lifespan=lifespan)
app.state.store = PlaygroundState()
app.state.settings = settings
app.middleware("http")(log_request)
app.add_middleware(RequestIdMiddleware)
app.include_router(router)


def run() -> None:
    """Start the search UI with uvicorn."""
    config = get_settings()
    uvicorn.run(
        "api.app:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        access_log=False,
        log_config=None,
    )


if __name__ == "__main__":
    run()
