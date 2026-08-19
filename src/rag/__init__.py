"""Rag playground.

Rag module for practising rag techniques.
"""

from dotenv import load_dotenv

load_dotenv()

from rag.pipeline import RagConfig, RagPipeline, RetrievalResult  # noqa: E402

__all__ = ["RagConfig", "RagPipeline", "RetrievalResult"]
