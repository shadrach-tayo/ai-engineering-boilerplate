"""Rag playground.

Rag module for practising rag techniques.
"""

from dotenv import load_dotenv

from rag.pipeline import RagConfig, RagPipeline, RetrievalResult

load_dotenv()

__all__ = ["RagConfig", "RagPipeline", "RetrievalResult"]
