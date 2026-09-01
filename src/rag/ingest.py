"""Rebuild Postgres and/or Elasticsearch indexes with Voyage-3.5 embeddings."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pdf_inspector
from langchain_core.documents import Document

from rag.pipeline import RagConfig, RagPipeline

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
GUIDES = ("openai-guide.pdf", "claude-guide.pdf", "support-agent.pdf")
INDEX_SIZES = {
    "chunk_256": 256,
    "chunk_512": 512,
    "chunk_1024": 1024,
}
INDEXES = tuple(INDEX_SIZES.items())
TARGETS = ("vector", "hybrid")


def load_guide_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load page-level markdown from the three agent-guide PDFs."""
    documents: list[Document] = []
    created_at = datetime.now().isoformat()
    for name in GUIDES:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing source PDF: {path}")
        result = pdf_inspector.extract_pages_markdown(path.as_posix())
        for page in result.pages:
            text = (page.markdown or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page.page,
                        "source": name,
                        "created_at": created_at,
                    },
                )
            )
    logger.info("Loaded %s pages from %s", len(documents), data_dir)
    return documents


def resolve_indexes(names: Sequence[str] | None = None) -> tuple[tuple[str, int], ...]:
    """Return ``(index_name, chunk_size)`` pairs for the requested indexes."""
    if not names:
        return INDEXES
    unknown = [name for name in names if name not in INDEX_SIZES]
    if unknown:
        raise ValueError(f"Unknown index names: {', '.join(unknown)}")
    return tuple((name, INDEX_SIZES[name]) for name in names)


def resolve_targets(names: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return ingest targets. Default is Elasticsearch hybrid only."""
    if not names:
        return ("hybrid",)
    unknown = [name for name in names if name not in TARGETS]
    if unknown:
        raise ValueError(f"Unknown targets: {', '.join(unknown)}")
    return tuple(dict.fromkeys(names))


def ingest_indexes(
    *,
    data_dir: Path = DATA_DIR,
    indexes: tuple[tuple[str, int], ...] = INDEXES,
    targets: tuple[str, ...] = ("hybrid",),
) -> None:
    """Write the agent-guide PDFs into the selected stores and chunk indexes."""
    documents = load_guide_documents(data_dir)
    for index_name, chunk_size in indexes:
        logger.info(
            "Ingesting %s (chunk_size=%s, targets=%s)",
            index_name,
            chunk_size,
            ",".join(targets),
        )
        pipeline = RagPipeline(
            RagConfig(
                index_name=index_name,
                chunk_size=chunk_size,
                embedding_dim=1024,
                embedding_model="voyage-3.5",
                rerank=False,
            )
        )
        pipeline.ingest(documents, index_name=index_name, targets=targets)
        logger.info("Finished %s", index_name)


def ingest_hybrid_indexes(
    *,
    data_dir: Path = DATA_DIR,
    indexes: tuple[tuple[str, int], ...] = INDEXES,
) -> None:
    """Recreate Elasticsearch indexes with Voyage-3.5 kNN vectors."""
    ingest_indexes(data_dir=data_dir, indexes=indexes, targets=("hybrid",))


def main(argv: Sequence[str] | None = None) -> None:
    """CLI to rebuild vector and/or hybrid indexes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--indexes",
        nargs="+",
        choices=tuple(INDEX_SIZES),
        help="Chunk indexes to rebuild. Default: all three.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGETS,
        default=["hybrid"],
        help="Stores to write. CI uses `vector` only (Postgres/pgvector).",
    )
    args = parser.parse_args(argv)
    ingest_indexes(
        data_dir=args.data_dir,
        indexes=resolve_indexes(args.indexes),
        targets=resolve_targets(args.targets),
    )


if __name__ == "__main__":
    main()
