"""OAuth helpers for authenticating against a remote MCP HTTP server."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pprint import pprint
from typing import Any
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

logger = logging.getLogger(__name__)

DEFAULT_MCP_SERVER_URL = "https://mcp.livemigrate.ai/api/mcp"
DEFAULT_CALLBACK_PORT = 3030
DEFAULT_SCOPE = "openid profile email offline_access"


class InMemoryTokenStorage(TokenStorage):
    """Keep OAuth tokens/client info in process memory (non-blocking)."""

    def __init__(self) -> None:
        """Constructor."""  # noqa: D401
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:  # noqa: D102
        pprint(self._tokens)
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store the latest OAuth token pair."""
        pprint(tokens)
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Return cached OAuth client metadata, if any."""
        pprint(self._client_info)
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Cache OAuth client metadata after registration."""
        pprint(client_info)
        self._client_info = client_info


class CallbackServer:
    """Local HTTP server that captures the OAuth redirect callback."""

    def __init__(self, port: int = DEFAULT_CALLBACK_PORT) -> None:
        """Bind the callback listener to ``port`` without starting it yet."""
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.callback_data: dict[str, Any] = {
            "authorization_code": None,
            "state": None,
            "error": None,
        }

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        callback_data = self.callback_data

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                if "code" in params:
                    callback_data["authorization_code"] = params["code"][0]
                    callback_data["state"] = params.get("state", [None])[0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h1>Authorization successful</h1>"
                        b"<p>You can close this window and return to the app.</p>"
                        b"</body></html>"
                    )
                elif "error" in params:
                    callback_data["error"] = params["error"][0]
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        f"<html><body><h1>Authorization failed</h1>"
                        f"<p>{params['error'][0]}</p></body></html>".encode()
                    )
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        return Handler

    def _start_sync(self) -> None:
        if self._server is not None:
            return
        self._server = HTTPServer(("localhost", self.port), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("OAuth callback server listening on http://localhost:%s/callback", self.port)

    def _stop_sync(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._server = None
        self._thread = None

    async def start(self) -> None:
        """Start the callback HTTP server on a background thread."""
        await asyncio.to_thread(self._start_sync)

    async def stop(self) -> None:
        """Stop the callback HTTP server and join its thread."""
        await asyncio.to_thread(self._stop_sync)

    async def wait_for_callback(self, timeout: float = 300) -> tuple[str, str | None]:
        """Poll for the authorization code without blocking the event loop."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.callback_data["authorization_code"]:
                return self.callback_data["authorization_code"], self.callback_data["state"]
            if self.callback_data["error"]:
                raise RuntimeError(f"OAuth error: {self.callback_data['error']}")
            await asyncio.sleep(0.1)
        raise TimeoutError("Timed out waiting for OAuth callback")


def create_oauth_provider(
    server_url: str | None = None,
    *,
    callback_port: int | None = None,
    scope: str | None = None,
    storage: TokenStorage | None = None,
) -> OAuthClientProvider:
    """Build an OAuthClientProvider wired for local browser login."""
    mcp_url = server_url or os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)
    port = callback_port or int(os.getenv("MCP_OAUTH_CALLBACK_PORT", DEFAULT_CALLBACK_PORT))
    oauth_scope = scope or os.getenv("MCP_OAUTH_SCOPE", DEFAULT_SCOPE)
    redirect_uri = f"http://localhost:{port}/callback"

    callback_server = CallbackServer(port=port)
    token_storage = storage or InMemoryTokenStorage()

    async def redirect_handler(authorization_url: str) -> None:
        callback_server.callback_data = {
            "authorization_code": None,
            "state": None,
            "error": None,
        }
        await callback_server.start()
        logger.info("Opening browser for MCP OAuth: %s", authorization_url)
        await asyncio.to_thread(webbrowser.open, authorization_url)

    async def callback_handler() -> tuple[str, str | None]:
        try:
            return await callback_server.wait_for_callback()
        finally:
            await callback_server.stop()
    
    return OAuthClientProvider(
        server_url=mcp_url,
        client_metadata=OAuthClientMetadata.model_validate(
            {
                "client_name": "langgraph-mcp-agent",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "scope": oauth_scope,
            }
        ),
        storage=token_storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
