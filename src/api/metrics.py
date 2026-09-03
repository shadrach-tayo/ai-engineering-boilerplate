"""Prometheus request counters and latency histograms for Grafana."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


def request_path_label(request: Request) -> str:
    """Use the route template so document ids do not explode cardinality."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


def observe_request(request: Request, status_code: int, duration_s: float) -> None:
    """Record one finished request. Scrapes of ``/metrics`` are skipped."""
    path = request_path_label(request)
    if path == "/metrics":
        return
    REQUESTS.labels(request.method, path, str(status_code)).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(duration_s)


def metrics_response() -> Response:
    """Return the current registry in Prometheus text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
