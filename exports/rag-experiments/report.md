# RAG pipeline experiment

Labeled 10-question retrieval eval across chunk indexes and retriever strategies.
Gold pages/phrases from `src/rag/eval/cases.py`. `top_k=5`. Source: `exports/rag-experiments/summary.json`.

## Findings

- **Best retrieval:** `chunk_512` + **vector** + **no rerank** — 100% source, page, phrase recall, auto-relevant, in 1.7s.
- **Vector ≈ ensemble** without rerank. Ensemble just concatenates vector + hybrid, then the gold pages are already in the Voyage hits.
- **Hybrid (ES BM25 + MiniLM kNN) lost badly** (20% page@k). Hits collapsed onto `support-agent.pdf` even for OpenAI/Claude questions. Different embedder than Postgres Voyage-3.5.
- **Rerank (Cohere) did not beat raw vector.** The original 18-config grid burned a trial key (10 calls/min) and scored incomplete rows. A paced 5-question check (retrieve 10 → rerank to 5, 8s between Cohere calls, 0 errors) confirmed it: 2 indexes tied at 100%, `chunk_1024` phrase recall dropped 93% → 87%. See [`rerank-compare/report.md`](rerank-compare/report.md).
- **chunk_1024** keeps 100% page@k but phrase recall drops to 97% (gold phrases get diluted in larger windows).
- Generation (`main.py`-style 5 questions × 3 indexes × 3 strategies, rerank on) saved 24/45 answers; the rest failed on the same Cohere 429s.

| Rank | Index | Strategy | Rerank | source@k | page@k | phrase recall | auto relevant | errors | time (s) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | chunk_512 | vector | no | 100% | 100% | 100% | 100% | 0 | 1.7 |
| 2 | chunk_256 | vector | no | 100% | 100% | 100% | 100% | 0 | 2.1 |
| 3 | chunk_512 | ensemble | no | 100% | 100% | 100% | 100% | 0 | 2.2 |
| 4 | chunk_256 | ensemble | no | 100% | 100% | 100% | 100% | 0 | 2.4 |
| 5 | chunk_256 | vector | yes | 100% | 100% | 100% | 100% | 0 | 5.0 |
| 6 | chunk_256 | ensemble | yes | 100% | 100% | 100% | 100% | 5 | 19.8 |
| 7 | chunk_512 | vector | yes | 100% | 100% | 100% | 100% | 3 | 20.9 |
| 8 | chunk_512 | ensemble | yes | 100% | 100% | 100% | 100% | 5 | 27.7 |
| 9 | chunk_1024 | vector | yes | 100% | 100% | 100% | 100% | 4 | 29.9 |
| 10 | chunk_1024 | vector | no | 100% | 100% | 97% | 100% | 0 | 1.6 |
| 11 | chunk_1024 | ensemble | no | 100% | 100% | 97% | 100% | 0 | 2.5 |
| 12 | chunk_1024 | ensemble | yes | 100% | 100% | 89% | 100% | 4 | 26.8 |
| 13 | chunk_512 | hybrid | yes | 20% | 20% | 53% | 20% | 5 | 36.7 |
| 14 | chunk_512 | hybrid | no | 20% | 20% | 43% | 20% | 0 | 0.4 |
| 15 | chunk_1024 | hybrid | no | 20% | 20% | 43% | 20% | 0 | 0.6 |
| 16 | chunk_256 | hybrid | no | 20% | 20% | 43% | 20% | 0 | 4.0 |
| 17 | chunk_1024 | hybrid | yes | 14% | 14% | 33% | 14% | 3 | 39.5 |
| 18 | chunk_256 | hybrid | yes | 0% | 0% | 0% | 0% | 10 | 33.6 |
