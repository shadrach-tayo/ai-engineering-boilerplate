"""Vector database module."""

import logging
import os
import time

from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from rag.embeddings import embeddings_model

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pineconeAPI = Pinecone(api_key=PINECONE_API_KEY)

logger = logging.getLogger(__file__)


class PineconeVectorStoreManager:
    """Vector database class."""

    def __init__(self, index_name: str, *, embedding_dim: int = 1024):
        """PineconeVectorStoreManager Constructor."""
        logger.info("init db class")
        self.index_name = index_name
        existing_indexes = [
            index_info["name"] for index_info in pineconeAPI.list_indexes()
        ]
        if index_name not in existing_indexes:
            pineconeAPI.create_index(
                name=self.index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            while not pineconeAPI.describe_index(index_name).status["ready"]:
                time.sleep(5)

        self.index = pineconeAPI.Index(index_name)
        self.vector_store = PineconeVectorStore(
            embedding=embeddings_model, index=self.index
        )

    def add_documents(self, documents):
        """PineconeVectorStoreManager Constructor."""
        logger.info("splitting documents...")
        splits = self._split_documents(documents)
        self.vector_store.add_documents(splits)
        logger.info(f"Added {len(splits)} to vector store")

    # def as_retriever(self, k: int = 4):
    #     return self.vector_store.as_retriever()

    def _split_documents(self, documents):
        logger.info("splitting documents")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(documents)
        return all_splits
