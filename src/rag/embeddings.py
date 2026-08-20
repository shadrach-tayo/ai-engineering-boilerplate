"""Voyage AI embedding model used by the RAG vector stores."""

import logging
import os

from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import SecretStr

load_dotenv()

logger = logging.getLogger(__name__)

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
embeddings_model = VoyageAIEmbeddings(
    api_key=SecretStr(VOYAGE_API_KEY),
    model="voyage-3.5",
)


def preload_voyage_tokenizer() -> None:
    """Download and cache the Voyage tokenizer so later embeds skip Hugging Face I/O."""
    try:
        embeddings_model._client.tokenizer(embeddings_model.model)
    except Exception:  # noqa: BLE001
        logger.warning("Could not preload Voyage tokenizer", exc_info=True)
