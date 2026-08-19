"""Search module for elastic search package."""

import json
import logging
import os
import time
from pprint import pprint

from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")


def reciprocal_rank_fuse(
    result_lists: list[list[dict]],
    *,
    rank_constant: int = 60,
) -> list[dict]:
    """Merge ranked hit lists with reciprocal rank fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank_constant + rank)
            docs[doc_id] = hit
    ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    fused = []
    for doc_id in ranked_ids:
        hit = dict(docs[doc_id])
        hit["_score"] = scores[doc_id]
        fused.append(hit)
    return fused


class Search:
    """..."""

    def __init__(self, chunk_size: int, embedding_dims: int):
        """Entry function."""
        self.chunk_size = chunk_size
        self.embedding_dims = embedding_dims
        self.model = embedding_model
        self.es = Elasticsearch(ELASTICSEARCH_URL)
        client_info = self.es.info()
        logger.info("Connecting to Elasticsearch!")
        pprint(client_info.body)

    def get_mapping(self, index: str):  # noqa: D102
        return self.es.indices.get_mapping(index=index)

    def create_index(self, index_name: str = "default-index"):
        """Create or Upsert existing index."""
        self.es.indices.delete(index=index_name, ignore_unavailable=True)
        dims = self.model.get_embedding_dimension()
        self.es.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    }
                }
            },
        )

    def get_embedding(self, text: str) -> list[float]:  # noqa: D102
        return self.model.encode(text).tolist()

    def insert_document(self, index_name: str, document):
        """Insert new document to index."""
        splits = self._split_documents(document)
        for split in splits:
            self.es.index(
                index=index_name,
                body=self._to_es_doc(split),
            )

    def insert_documents(self, index_name: str, documents):
        """Bulk Insert new documents to index."""
        operations = []
        splits = self._split_documents(documents)
        for document in splits:
            operations.append({"index": {"_index": index_name}})
            operations.append(self._to_es_doc(document))
        return self.es.bulk(operations=operations)

    def reindex(self, index_name: str, documents):
        """Refresh an index."""
        self.create_index(index_name=index_name)
        return self.insert_documents(index_name=index_name, documents=documents)

    def search(self, index_name: str, **query_args):
        """Run search query on index."""
        return self.es.search(index=index_name, **query_args)

    def hybrid_search(
        self,
        index_name: str,
        question: str,
        *,
        size: int = 10,
        from_: int = 0,
        rank_constant: int = 60,
        window_size: int | None = None,
    ):
        """Fuse BM25 and kNN rankings with reciprocal rank fusion."""
        window = max(window_size or 50, size)
        query_vector = self.get_embedding(question)
        bm25 = self.search(
            index_name=index_name,
            query={"multi_match": {"query": question, "fields": ["content"]}},
            size=window,
        )
        knn = self.search(
            index_name=index_name,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": window,
                "num_candidates": max(50, window * 2),
            },
            size=window,
        )
        fused = reciprocal_rank_fuse(
            [bm25["hits"]["hits"], knn["hits"]["hits"]],
            rank_constant=rank_constant,
        )
        page = fused[from_ : from_ + size]
        return {
            "hits": {
                "hits": page,
                "total": {"value": len(fused), "relation": "eq"},
            }
        }

    def retrieve_document(self, index: str, id):
        """Retrieve a document by id from an index."""
        return self.es.get(index=index, id=id)

    def _to_es_doc(self, doc: Document) -> dict:
        return {
            "content": doc.page_content,
            "embedding": self.get_embedding(doc.page_content),
            **doc.metadata,  # page, source, created_at
        }

    def _split_documents(self, documents):
        logger.info("splitting documents")
        # text_splitter = RecursiveCharacterTextSplitter(
        # chunk_size=1000,
        #     chunk_overlap=200,
        #     add_start_index=True,
        # )
        # return text_splitter.split_documents(documents)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=self.chunk_size, chunk_overlap=30
        )
        return text_splitter.split_documents(documents)
