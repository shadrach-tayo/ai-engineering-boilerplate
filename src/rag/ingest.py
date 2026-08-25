"""Rebuild Elasticsearch hybrid indexes with the same embedder as pgvector."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pdf_inspector
from langchain_core.documents import Document

from rag.pipeline import RagConfig, RagPipeline

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
GUIDES = ("openai-guide.pdf", "claude-guide.pdf", "support-agent.pdf")
INDEXES = (
    ("chunk_256", 256),
    ("chunk_512", 512),
    ("chunk_1024", 1024),
)


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


def ingest_hybrid_indexes(
    *,
    data_dir: Path = DATA_DIR,
    indexes: tuple[tuple[str, int], ...] = INDEXES,
) -> None:
    """Recreate Elasticsearch indexes with Voyage-3.5 kNN vectors."""
    documents = load_guide_documents(data_dir)
    for index_name, chunk_size in indexes:
        logger.info("Ingesting hybrid index %s (chunk_size=%s)", index_name, chunk_size)
        pipeline = RagPipeline(
            RagConfig(
                index_name=index_name,
                chunk_size=chunk_size,
                embedding_dim=1024,
                embedding_model="voyage-3.5",
                rerank=False,
            )
        )
        pipeline.ingest(documents, index_name=index_name, targets=("hybrid",))
        logger.info("Finished %s", index_name)


def main() -> None:
    """CLI to rebuild Elasticsearch hybrid indexes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    ingest_hybrid_indexes(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
