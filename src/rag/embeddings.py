import os

from langchain_voyageai import VoyageAIEmbeddings

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
embeddings_model = VoyageAIEmbeddings(voyage_api_key=VOYAGE_API_KEY, model="voyage-3.5")
