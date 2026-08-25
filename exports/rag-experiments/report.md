# RAG pipeline experiment

Labeled 10-question retrieval eval across chunk indexes and retriever strategies.
Gold pages/phrases from `src/rag/eval/cases.py`. `top_k=5`, no Cohere rerank.
Hybrid kNN now uses **Voyage-3.5** (same embedder as Postgres), not MiniLM.
Source: `exports/rag-experiments/summary.json` (24 Aug 2026, after `uv run rag-ingest`).

## Findings

- **Best retrieval:** `chunk_512` + **vector** + **no rerank** — 100% source, page, phrase recall, in 1.7s.
- **The MiniLM hybrid collapse was an embedder mismatch, not a hybrid-retrieval result.** The first grid’s 20% page@k happened because ES kNN used `all-MiniLM-L6-v2` (384-d) while Postgres used Voyage-3.5 (1024-d). Hits piled onto `support-agent.pdf`. After reindexing ES with Voyage-3.5, hybrid is **100% source@5 and page@5** on every chunk size.
- **Hybrid still trails vector on phrases** (93% / 90% / 83% vs 100% / 100% / 97%). The remaining gap is real BM25 behavior: for “guardrails”, hybrid ranks openai-guide intro/SDK pages that mention the word, while Voyage’s top-5 includes the classifier/PII/moderation definition. BM25 catching exact terms is useful; it is not a drop-in replacement for the dense ranking on this labeled set.
- **Vector ≈ ensemble.** Ensemble concatenates vector + hybrid; gold pages were already in the Voyage hits, so extra ES docs add latency without lifting accuracy here.
- **chunk_1024** keeps 100% page@k but phrase recall drops to 97% on vector (gold phrases get diluted in larger windows).

| Rank | Index | Strategy | Rerank | source@k | page@k | phrase recall | auto relevant | errors | time (s) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | chunk_512 | vector | no | 100% | 100% | 100% | 100% | 0 | 1.7 |
| 2 | chunk_256 | vector | no | 100% | 100% | 100% | 100% | 0 | 2.1 |
| 3 | chunk_512 | ensemble | no | 100% | 100% | 100% | 100% | 0 | 3.3 |
| 4 | chunk_256 | ensemble | no | 100% | 100% | 100% | 100% | 0 | 3.3 |
| 5 | chunk_1024 | vector | no | 100% | 100% | 97% | 100% | 0 | 1.7 |
| 6 | chunk_1024 | ensemble | no | 100% | 100% | 97% | 100% | 0 | 3.3 |
| 7 | chunk_256 | hybrid | no | 100% | 100% | 93% | 100% | 0 | 1.7 |
| 8 | chunk_512 | hybrid | no | 100% | 100% | 90% | 100% | 0 | 1.6 |
| 9 | chunk_1024 | hybrid | no | 100% | 100% | 83% | 100% | 0 | 1.6 |
