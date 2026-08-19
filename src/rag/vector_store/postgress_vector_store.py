"""Postgres + pgvector store."""

import logging
import os

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_text_splitters import (
    CharacterTextSplitter,
    # RecursiveCharacterTextSplitter,
)

from rag.embeddings import embeddings_model

logger = logging.getLogger(__file__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://rag:rag@localhost:54325/rag",
)


class PostgresVectorStoreManager:
    """Vector database class."""

    def __init__(
        self,
        index_name: str,
        *,
        embedding_dim: int = 1024,
        chunk_size: int | None = None,
        chunk_overlap: int = 30,
        embeddings: Embeddings | None = None,
    ):  # noqa: D107
        logger.info("init db class")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_name = index_name
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL is not set")
        self.engine = PGEngine.from_connection_string(url=DATABASE_URL)

        # CREATE TABLES IF MISSING
        if chunk_size:
            self.engine.init_vectorstore_table(
                table_name=index_name,
                vector_size=embedding_dim,
            )

        self.vector_store = PGVectorStore.create_sync(
            engine=self.engine,
            table_name=index_name,
            embedding_service=embeddings or embeddings_model,
        )

    def add_documents(self, documents):  # noqa: D102
        logger.info("splitting documents...")
        splits = self._split_documents(documents)
        self.vector_store.add_documents(splits)
        logger.info(f"Added {len(splits)} to vector store")

    def as_retriever(self, k: int = 4):  # noqa: D102
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def _split_documents(self, documents):
        logger.info("splitting documents")
        # text_splitter = RecursiveCharacterTextSplitter(
        # chunk_size=1000,
        #     chunk_overlap=200,
        #     add_start_index=True,
        # )
        # return text_splitter.split_documents(documents)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_documents(documents)
