"""Shared retrieval pipeline used by the graphs and the search UI."""

from dotenv import load_dotenv

load_dotenv()

from rag.tracing import setup_braintrust  # noqa: E402

setup_braintrust()

from rag.pipeline import RagConfig, RagPipeline, RetrievalResult  # noqa: E402

__all__ = ["RagConfig", "RagPipeline", "RetrievalResult"]
