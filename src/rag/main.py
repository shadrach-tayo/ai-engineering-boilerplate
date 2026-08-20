"""Rag module for hybrid search apis."""

import json
import logging

# import os
import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

import pdf_inspector

# from docling.document_converter import DocumentConverter
# from langchain_community.retrievers import KNNRetriever
# from langchain_docling import DoclingLoader
from langchain_core.documents import Document

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
# from rag.embeddings import embeddings_model
from rag.elasticsearch_store.search import Search
from rag.pipeline import RagConfig, RagPipeline
from rag.vector_store.pinecone_vector_store import PineconeVectorStoreManager
from rag.vector_store.postgress_vector_store import PostgresVectorStoreManager

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# documents = [
#     "Caching embeddings enables the storage or temporary caching of embeddings, eliminating the necessity to recompute them each time.",
#     "An LLMChain is a chain that composes basic LLM functionality. It consists of a PromptTemplate and a language model (either an LLM or chat model). It formats the prompt template using the input key values provided (and also memory key values, if available), passes the formatted string to LLM and returns the LLM output.",
#     "A Runnable represents a generic unit of work that can be invoked, batched, streamed, and/or transformed.",
# ]

base_path = __file__

index_name = "chunk_1024"
chunk_size = 1024
embedding_dim = 256

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    """Return the process-wide pipeline configured from module defaults."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(
            RagConfig(
                index_name=index_name,
                chunk_size=chunk_size,
                embedding_dim=embedding_dim,
            )
        )
    return _pipeline


def main():
    """Main function."""  # noqa: D401
    # logger.info(documents)
    # print(f"\n")
    # document_embeds = voyage_embeddings.embed_documents(documents)
    # logger.info(len(document_embeds))

    # query = "What's an LLMChain?"
    # query_embed = voyage_embeddings.embed_query(query)
    # logger.info(query_embed[:5])
    # print(f"\n")

    # retriever = KNNRetriever.from_texts(documents, voyage_embeddings)
    # result = retriever.invoke(query)
    # top1_retrieved_doc = result[0].page_content
    # logger.info("Result: %s", result)

    sources = [
        Path(__file__).parent / "../../data/openai-guide.pdf",
        Path(__file__).parent / "../../data/claude-guide.pdf",
        Path(__file__).parent / "../../data/support-agent.pdf",
    ]

    # converter = DocumentConverter()
    # doc = converter.convert(source).document
    # logger.info(doc.export_to_markdown())

    # loader = DoclingLoader(sources[1])
    # docs = loader.load()
    # for d in docs[:3]:
    #     logger.info(d)

    # results = [
    #     pdf_inspector.extract_pages_markdown(source.as_posix()) for source in sources
    # ]

    # result = pdf_inspector.extract_pages_markdown(sources[0].as_posix())
    # logger.info(results)

    # markdown_guide = "\n\n".join(page.markdown for page in results[0].pages)
    # logger.info(markdown_guide)
    # headers_to_split_on = [
    #     ("#", "Header 1"),
    #     ("##", "Header 2"),
    #     ("###", "Header 3"),
    #     ("####", "Header 4"),
    # ]
    # markdown_splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=headers_to_split_on, strip_headers=False
    # )
    # markdown_docs = markdown_splitter.split_text(markdown_guide)
    # logger.info("\n\n")
    # logger.info(markdown_docs)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    # markdown_texts = text_splitter.split_documents(markdown_docs)
    # logger.info("\n\n")
    # logger.info(markdown_texts)

    # store = PostgresVectorStoreManager(index_name, chunk_size=chunk_size)

    # es.create_index(index_name)

    # for index, result in enumerate(results):
    #     logger.info(
    #         "source: %s",
    #         os.path.basename(sources[index].as_uri()),
    #     )
    #     source = os.path.basename(sources[index].as_uri())
    #     docs = [
    #         Document(
    #             page_content=page.markdown,
    #             metadata={
    #                 "page": page.page,
    #                 "source": source,
    #                 "created_at": datetime.now().isoformat(),
    #             },
    #         )
    #         for page in result.pages
    #     ]

    # store.add_documents(docs)
    # pg_index_injestion(index=index_name, documents=docs)
    # es_index_injestion(index=index_name, documents=docs)

    # index_data_sources("chunk_1024", 1024, 256, docs)
    # index_data_sources("chunk_512", 512, 256, docs)
    # index_data_sources("chunk_256", 256, 256, docs)

    # for page in result.pages:
    #     # print(
    #     #     f"Page {page.page}: {len(page.markdown)} chars, needs_ocr={page.needs_ocr}"
    #     # )

    #     document_embeds = embeddings_model.embed_documents(page.markdown)
    #     logger.info(len(document_embeds))
    # print(result.pdf_type)  # "text_based", "scanned", "image_based", "mixed"
    # print(result.confidence)  # 0.0 - 1.0
    # print(result.page_count)  # number of pages
    # print(result.markdown)

    logger.info("Rag main: ✅")


def index_data_sources(  # noqa: D103
    index_name: str, chunk_size: int, embedding_dim: int, documents: list[Document]
):
    es_index = Search(chunk_size, embedding_dim)
    es_index.create_index(index_name)
    results = es_index.insert_documents(index_name=index_name, documents=documents)
    logger.info(
        "ES index populated: %s, entries: %s", index_name, len(results["items"])
    )


def vector_rag(index_name: str, question: str, *, top_n: int = 5):
    """Retrieve relevant document from the vector database indexed by the index_name parameter."""
    return (
        get_pipeline()
        .retrieve(
            question,
            index_name=index_name,
            strategy="vector",
            top_k=top_n,
            rerank=True,
        )
        .as_vector_payload()
    )


def rerank(docs: list[str], question: str, top_n: int = 3):
    """Rerank document from vector retriever."""
    ranked = get_pipeline().rerank_documents(docs, question, top_n=top_n)
    return {"docs": ranked.docs, "rerank": ranked.rerank}


def save_data(data, file_name):  # noqa: D103
    save_file_path = Path(base_path).parent / f"../../data/{file_name}.json"
    with open(save_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
        logger.info(f"Saved: {save_file_path}")


def agent_rag(question: str, store_index: str | None = None, *, top_n: int = 5):
    """Answer a question using vector RAG and Elasticsearch hybrid hits."""
    return get_pipeline().generate(
        question,
        index_name=store_index or index_name,
        strategy="ensemble",
        top_k=top_n,
    )


def es_index_injestion(index: str, documents):
    """Index Elasticsearch cluster index."""
    results = get_pipeline().search_client.insert_documents(index, documents)
    logger.info("ES index populated: %s, entries: %s", index, len(results["items"]))
    # pprint(results['items'])


def pg_index_injestion(index: str, documents):
    """Postgres DB vector embedding index."""
    store = PostgresVectorStoreManager(index_name=index, chunk_size=chunk_size)
    store.add_documents(documents)
    logger.info("ES index populated: %s", index)


def search(
    question: str,
    store_index: str | None = None,
    *,
    size: int = 10,
    from_: int = 0,
):
    """Search Elasticsearch cluster index with BM25 + kNN RRF fusion."""
    return (
        get_pipeline()
        .retrieve(
            question,
            index_name=store_index or index_name,
            strategy="hybrid",
            top_k=size,
            from_=from_,
            rerank=False,
        )
        .as_es_payload()
    )


if __name__ == "__main__":
    # es = Search()
    response = search("What is an agent?", index_name)
    results = response.get("results", [])
    for result in results:
        pprint(
            {"source": result["_source"]["source"], "page": result["_source"]["page"]}
        )
    # for result in results
    # logger.info("mappings after: ")
    # pprint(es.get_mapping("chunk_1024"))
    # pprint(es.get_mapping("chunk_512"))
    # pprint(es.get_mapping("chunk_256"))
    # main()

    # questions = [
    #     "What is an Agent?",
    #     "What are the guardrails used in building AI agents?",
    #     "What are the Common architecture patterns for building AI Agents?",
    #     "What are the Common use cases and applications for AI agents?",
    #     "what are evaluations and how do we build evaluations for agentic systems?",
    # ]
    # rag_responses = []
    # agent_responses = []
    # for question in questions:
    #     rag_response = vector_rag(index_name, question)
    #     rag_responses.append(rag_response)
    #     agent_response = agent_rag(question)
    #     agent_responses.append(agent_response)

    # save_data(rag_responses, f"rag_{index_name}_rerank")
    # save_data(agent_responses, f"agent_{index_name}_rerank")
