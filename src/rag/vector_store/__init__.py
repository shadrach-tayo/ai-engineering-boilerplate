"""Vector store protocols and backends."""

from typing import Protocol


class VectorStoreLike(Protocol):
    """Minimal document store interface used by the RAG pipeline."""

    def add_documents(self, documents):
        """Persist documents in the store."""
        ...

    def as_retriever(self, k: int = 4):
        """Return a retriever limited to ``k`` results."""
        ...
