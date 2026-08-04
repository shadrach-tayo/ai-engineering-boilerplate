"""Postgres + pgvector store"""

import logging
import os

from langchain_postgres import PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import embeddings_model

logger = logging.getLogger(__file__)

DATABASE_URL = os.getenv("DATABASE_URL")


class PostgresVectorStoreManager:
    """Vector database class"""  # noqa: D415

    def __init__(self, index_name: str, *, embedding_dim: int = 1024):  # noqa: D107
        logger.info("init db class")
        self.index_name = index_name
        self.engine = PGEngine.from_connection_string(url=DATABASE_URL)

        # CREATE TABLES IF MISSING
        # if not self.engine.
        # self.engine.init_vectorstore_table(
        #     table_name=index_name,
        #     vector_size=embedding_dim,
        # )

        self.vector_store = PGVectorStore.create_sync(
            engine=self.engine,
            table_name=index_name,
            embedding_service=embeddings_model,
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        return text_splitter.split_documents(documents)
