# LangGraph RAG Assistants

An experimental documentation assistant built with
[LangGraph](https://github.com/langchain-ai/langgraph). The repository compares
four agent workflows over a shared retrieval-augmented generation (RAG)
pipeline:

- a tool-calling documentation assistant;
- a corrective Self-RAG workflow;
- Self-RAG with parallel query decomposition; and
- Self-RAG with hypothetical document embeddings (HyDE).

The project also contains a FastAPI retrieval playground, Postgres/pgvector and
Elasticsearch backends, retrieval evaluations, and LangGraph thread export
tools.

## Current capabilities

- Four graphs exposed through `langgraph.json` and LangGraph Studio.
- Vector retrieval with Voyage embeddings and Postgres/pgvector.
- Hybrid BM25 + Voyage-3.5 kNN retrieval with Elasticsearch RRF fusion.
- An ensemble strategy that combines vector and hybrid results.
- Optional Cohere reranking and OpenAI answer generation.
- Relevance, support, and answer-utility grading in the corrective graphs.
- A browser UI for comparing vector and Elasticsearch results.
- A 10-question labeled retrieval eval, an 18-configuration grid, DeepEval generator
  metrics, and named Braintrust experiments that compare two pipeline versions.

The code includes an OAuth provider and an MCP client scaffold. Remote MCP
connections are currently commented out, so the active graph tools are the
local documentation search and Tavily web search tools.

## Architecture

The indexing path uses 512-token chunks. The same Voyage-3.5 embeddings go
into Postgres/pgvector and Elasticsearch dense kNN; Elasticsearch also
indexes the chunk text for BM25. Query-time BM25 and kNN hits are fused with
RRF.

```mermaid
flowchart LR
    docs[Source documents] --> ingest[Ingest] --> chunk[Chunk: 512 tokens]
    chunk --> embed[Voyage-3.5 embeddings]
    embed --> pg[(Postgres + pgvector)]
    embed --> esKnn[Elasticsearch kNN]
    chunk --> esBm25[(Elasticsearch BM25)]

    question[User question] --> router[Multi-source router]
    router -->|local docs| tools[Local RAG tools]
    router -->|web fallback| tavily[Tavily]
    tools -->|vector search| pg
    tools -->|hybrid search| esKnn
    tools -->|hybrid search| esBm25
    pg --> fusion[RRF / hybrid fusion]
    esKnn --> fusion
    esBm25 --> fusion
    fusion --> grade{Evidence relevant?}
    tavily --> grade
    grade -->|yes| generate[Generate answer] --> answer([Grounded answer])
    grade -->|rewrite / retry| router
```

## Graphs

The four top-level graphs below are registered in `langgraph.json`.

### Documentation Assistant

- Graph ID: `documentation-assistant`
- Source: `src/agent/graph.py`

The model must call at least one tool. Tool results are combined into one
context block before the model runs again.

```mermaid
flowchart TD
    start([START]) --> assistant
    assistant -->|tool calls| tools
    assistant -->|no tool calls| finish([END])
    tools --> combine
    combine --> assistant
```

### Self-RAG Assistant

- Graph ID: `self-rag-assistant`
- Source: `src/agent/self_rag.py`

The graph decides whether retrieval is needed, grades retrieved passages,
rewrites failed searches, and grades both grounding support and answer utility.
Its first retrieval uses the local index; retries use web search.

```mermaid
flowchart TD
    start([START]) --> entry
    entry --> decide_retrieve
    decide_retrieve -->|retrieve: yes| retrieve
    decide_retrieve -->|retrieve: no| generate
    retrieve --> grade_documents
    grade_documents -->|relevant document or max iterations| generate
    grade_documents -->|no relevant documents| rewrite
    rewrite --> retrieve
    generate --> grade_supports
    grade_supports --> grade_utility
    grade_utility -->|utility >= 3 and iterations >= MAX_ITERS| decide_retrieve
    grade_utility -->|otherwise| finish([END])
```

### Query Decomposition Assistant

- Graph ID: `query-decomposition-assistant`
- Source: `src/agent/query_decomposition.py`

Retrieval questions are decomposed into as many as four subqueries. LangGraph
`Send` branches retrieve and grade those subqueries in parallel before the
workflow generates an answer or rewrites the search.

```mermaid
flowchart TD
    start([START]) --> entry
    entry --> decide_retrieve
    decide_retrieve -->|retrieve: yes| decompose
    decide_retrieve -->|retrieve: no| generate
    decompose -->|1-4 parallel Send branches| retrieve_sub
    retrieve_sub -->|relevant document or max iterations| generate
    retrieve_sub -->|no relevant documents| rewrite
    rewrite --> decompose
    generate --> grade_supports
    grade_supports --> grade_utility
    grade_utility -->|utility >= 3 and iterations >= MAX_ITERS| decide_retrieve
    grade_utility -->|otherwise| finish([END])
```

### HyDE Assistant

- Graph ID: `hyde-assistant`
- Source: `src/agent/hyde.py`

The nested HyDE subgraph writes a hypothetical answer, retrieves against that
text, and reranks the results against the original question. The parent graph
then applies the same corrective grading loop.

```mermaid
flowchart TD
    start([START]) --> entry
    entry --> decide_retrieve
    decide_retrieve -->|retrieve: no| generate

    subgraph hyde [HyDE retrieval subgraph]
        direction TB
        write_hypothetical --> retrieve_hyde
    end

    decide_retrieve -->|retrieve: yes| write_hypothetical
    retrieve_hyde --> grade_documents
    grade_documents -->|relevant document or max iterations| generate
    grade_documents -->|no relevant documents| rewrite
    rewrite --> write_hypothetical
    generate --> grade_supports
    grade_supports --> grade_utility
    grade_utility -->|utility >= 3 and iterations >= MAX_ITERS| decide_retrieve
    grade_utility -->|otherwise| finish([END])
```

## Project structure

```text
src/
├── agent/                 # LangGraph workflows and OAuth helper
├── api/                   # FastAPI retrieval playground
└── rag/
    ├── elasticsearch_store/
    ├── vector_store/
    ├── eval/              # Cases, metrics, single-run and grid evaluations
    └── pipeline.py        # Shared retrieval, reranking and generation pipeline
scripts/
└── export_langgraph_threads.py
exports/
├── rag-experiments/
└── thread-comparison/
```

## Setup

Requirements:

- Python 3.10 or newer;
- [uv](https://docs.astral.sh/uv/);
- Docker, when running the local data stores;
- API keys for the providers used by the selected workflow.

Install the project and development tools:

```bash
uv sync --group dev
cp .env.example .env
```

At minimum, set `OPENAI_API_KEY` and `VOYAGE_API_KEY`. Set
`DEEPSEEK_API_KEY` for DeepEval generator tests (they use `deepseek-chat`).
Set `COHERE_API_KEY` when reranking, `TAVILY_API_KEY` when a corrective workflow
can fall back to web search, and `LANGSMITH_API_KEY` to enable LangSmith
tracing.

To send LLM traces to [Braintrust](https://www.braintrust.dev/docs/instrument/trace-llm-calls),
set `BRAINTRUST_API_KEY` (or put it in a gitignored `.env.braintrust`) and
optionally `BRAINTRUST_PROJECT` (defaults to `langgraph-mcp`) or
`BRAINTRUST_PROJECT_ID`. Traces are emitted from the search playground and from
DeepEval generator tests.

Start the local stores:

```bash
docker compose up -d
```

Docker exposes Postgres on host port `54325` and Elasticsearch on `9200`.
Use the following Postgres URL when running against the included Compose
service:

```dotenv
DATABASE_URL=postgresql+psycopg://rag:rag@localhost:54325/rag
ELASTICSEARCH_URL=http://localhost:9200
```

The assistants and search UI expect populated `chunk_256`, `chunk_512`, and/or
`chunk_1024` indexes. Rebuild Elasticsearch hybrid indexes (Voyage-3.5 kNN +
BM25) with:

```bash
uv run rag-ingest
```

`RagPipeline.ingest(...)` can also write Postgres and/or Elasticsearch. The
hybrid path must use the same embedder as pgvector; mixing MiniLM kNN with
Voyage vectors is what made the first hybrid eval collapse onto one PDF.

## Run the graphs

Start the local LangGraph API and open its Studio URL:

```bash
uv run langgraph dev
```

Select any graph registered in `langgraph.json`. Graph inputs use LangChain
message objects, for example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What are common AI agent architecture patterns?"
    }
  ]
}
```

The graph modules warm the `chunk_1024` vector store during import, so Postgres
must be reachable and the Voyage key must be configured before the LangGraph
server loads them.

## Run the search playground

```bash
uv run search-api
```

Open <http://127.0.0.1:5001>. The UI compares reranked Postgres vector results
with paginated Elasticsearch hybrid results and can optionally generate an
answer. With `BRAINTRUST_API_KEY` set, each search is a parent span on the
Braintrust dashboard (retrieve, rerank, and generate nested underneath).

## How we measure quality

Quality is split so retrieval bugs are not judged by an LLM, and hallucinations
are not scored with phrase matching.

1. **Retrieval (labeled, no judge).** Ten gold questions in
   `src/rag/eval/cases.py` score `source@k`, `page@k`, and phrase recall.
   `rag-eval` / `rag-eval-grid` swept chunk size × strategy × rerank. The grid
   winner is `chunk_512` + vector + no rerank (100% source/page/phrase). Hybrid
   keeps page hits after the Voyage reindex but trails on phrases. Cohere rerank
   did not beat raw vector. Artifacts: `exports/rag-experiments/`.
2. **Generation (DeepEval, LLM-as-judge).** Thirty goldens
   (`src/rag/eval/tests/goldens.json`): 10 curated, 10 synthesized edges, 10
   out-of-domain refusals. In-domain metrics: Faithfulness, Answer Relevancy,
   Hallucination, Contextual Precision/Recall, Answer Correctness. OOD cases
   use a refusal GEval only. CI gates on Faithfulness ≥ 0.8 and hallucination
   contradiction rate ≤ 0.1. Run with `uv run rag-eval-generator`, not raw
   pytest, so Confident AI gets a named identifier.
3. **Observability.** Braintrust traces (`playground.search`, `rag.retrieve`,
   `rag.generate`, `eval.generator`) show how a request ran. Named Braintrust
   **experiments** compare two pipeline versions on the labeled retrieval set
   (see below). RAGAS wrappers were tried via DeepEval and dropped; the same
   dimensions remain as native Contextual Precision/Recall.

The DeepEval suite scores a *different* config than the grid winner: retrieve
10, Cohere rerank to 5. Named Braintrust experiments make that A/B explicit.

## Evaluate retrieval

Run one labeled evaluation:

```bash
uv run rag-eval --index chunk_512 --strategy vector --top-k 5 --no-rerank
```

Run the full chunk-size, strategy, and reranking grid:

```bash
uv run rag-eval-grid --top-k 5 --no-generate --no-rerank
```

Add `--no-generate` to skip answer generation and `--no-rerank` to skip
Cohere (trial keys 429 after a few calls). Outputs are written under
`exports/rag-experiments/`.

Cohere trial keys allow 10 rerank calls per minute. To compare vector vs
rerank without 429s, run the paced 5-question check (retrieve 10, rerank to
5, 8s between Cohere calls):

```bash
uv run rag-eval-grid --rerank-compare
```

Results land in `exports/rag-experiments/rerank-compare/`.

Compare the grid winner against the DeepEval generator config as named
Braintrust experiments (same 10 questions, scored at k=5):

```bash
uv run rag-eval-braintrust
```

That logs `chunk512-vector-k5-norerank` then `chunk512-vector-k10-rerank5`
with the first as `base_experiment`. Open the candidate URL and use Compare
to see source/page/phrase deltas. Requires `BRAINTRUST_API_KEY`,
`VOYAGE_API_KEY`, `COHERE_API_KEY`, and a populated `chunk_512` index.
`--baseline-only` / `--candidate-only` re-run one side;
`--no-pause` skips the Cohere trial-key pacing.

Run the DeepEval generator metrics with a stable Confident AI identifier derived
from the experiment and current PR or branch:

```bash
uv run rag-eval-generator --experiment latest-dataset
```

Use that command, not raw `pytest`. Only `deepeval test run` (wrapped by
`rag-eval-generator`) uploads pass/fail results, hyperparameters, and a
searchable identifier to Confident AI. `uv run pytest src/rag/eval/tests/...`
keeps scores local. Generator and judge LLM calls from that suite also show up
in Braintrust when `BRAINTRUST_API_KEY` is set.

Use `--identifier checkout-agent-v2` (or `DEEPEVAL_IDENTIFIER`) when an explicit
shared identifier is preferable. Additional options are forwarded to
`deepeval test run`.

Pull requests run the same suite in GitHub Actions (`.github/workflows/rag-eval.yml`).
The job comments a score table and **fails the build** when mean Faithfulness is
below `0.8` or the Hallucination contradiction rate is above `0.1`. Pytest
failures such as Answer Correctness do not fail CI. Required secrets:
`DEEPSEEK_API_KEY`, `VOYAGE_API_KEY`, `COHERE_API_KEY`, and `DATABASE_URL`
pointing at Postgres with a populated `chunk_512` index. Optional:
`CONFIDENT_API_KEY`, plus `BRAINTRUST_API_KEY` (and optionally
`BRAINTRUST_PROJECT` or `BRAINTRUST_PROJECT_ID`) to upload eval traces.

Regenerate the 30 single-turn goldens and push them to Confident AI:

```bash
uv run rag-eval-dataset --alias rag-agent-guides-single-turn-v1
```

The generator metrics load the same local dataset from
`src/rag/eval/tests/goldens.json`.

## Export graph comparisons

With `langgraph dev` running, export completed Studio threads for the three
corrective workflows:

```bash
uv run python scripts/export_langgraph_threads.py
```

The comparison is written to `exports/thread-comparison/`.

## Development

Run tests and static checks:

```bash
uv run pytest tests/unit_tests
uv run pytest tests/rag
uv run ruff check .
uv run mypy --strict src/
```

Integration tests require the external services, indexes, and provider
credentials used by the graphs.

