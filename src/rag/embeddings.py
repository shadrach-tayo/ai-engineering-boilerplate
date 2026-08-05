"""Voyage AI embedding model used by the RAG vector stores."""

import os

from langchain_voyageai import VoyageAIEmbeddings
from pydantic import SecretStr

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
embeddings_model = VoyageAIEmbeddings(
    api_key=SecretStr(VOYAGE_API_KEY),
    model="voyage-3.5",
)
