"""Tests for ingest index and target selection."""

import pytest

from rag.ingest import INDEXES, resolve_indexes, resolve_targets


def test_resolve_indexes_defaults_to_all_chunk_sizes() -> None:
    assert resolve_indexes(None) == INDEXES
    assert resolve_indexes([]) == INDEXES


def test_resolve_indexes_filters_to_requested_names() -> None:
    assert resolve_indexes(["chunk_512"]) == (("chunk_512", 512),)


def test_resolve_indexes_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="chunk_99"):
        resolve_indexes(["chunk_99"])


def test_resolve_targets_defaults_to_hybrid() -> None:
    assert resolve_targets(None) == ("hybrid",)
    assert resolve_targets([]) == ("hybrid",)


def test_resolve_targets_dedupes_and_keeps_order() -> None:
    assert resolve_targets(["vector", "hybrid", "vector"]) == ("vector", "hybrid")


def test_resolve_targets_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="pinecone"):
        resolve_targets(["pinecone"])
