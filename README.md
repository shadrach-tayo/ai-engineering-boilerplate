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
- Hybrid BM25 + MiniLM kNN retrieval with Elasticsearch RRF fusion.
- An ensemble strategy that combines vector and hybrid results.
- Optional Cohere reranking and OpenAI answer generation.
- Relevance, support, and answer-utility grading in the corrective graphs.
- A browser UI for comparing vector and Elasticsearch results.
- A 10-question retrieval evaluation and an 18-configuration experiment grid.

The code includes an OAuth provider and an MCP client scaffold. Remote MCP
connections are currently commented out, so the active graph tools are the
local documentation search and Tavily web search tools.

## Architecture

The indexing path uses 512-token chunks. Chunks are embedded for pgvector and
indexed directly as text for Elasticsearch BM25; query-time results are fused
before the Self-RAG relevance check.

```mermaid
flowchart LR
    docs[Source documents] --> ingest[Ingest] --> chunk[Chunk: 512 tokens]
    chunk -->|semantic path| embed[Voyage embeddings] --> pg[(Postgres + pgvector)]
    chunk -->|lexical path| es[(Elasticsearch BM25)]

    question[User question] --> router[Multi-source router]
    router -->|local docs| tools[Local RAG tools]
    router -->|web fallback| tavily[Tavily]
    tools -->|vector search| pg
    tools -->|keyword search| es
    pg --> fusion[RRF / hybrid fusion]
    es --> fusion
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
`COHERE_API_KEY` when reranking, `TAVILY_API_KEY` when a corrective workflow
can fall back to web search, and `LANGSMITH_API_KEY` to enable LangSmith
tracing.

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
`chunk_1024` indexes. `RagPipeline.ingest(...)` is the current programmatic
ingestion API; there is not yet a dedicated ingestion command.

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
answer.

## Evaluate retrieval

Run one labeled evaluation:

```bash
uv run rag-eval --index chunk_512 --strategy vector --top-k 5 --no-rerank
```

Run the full chunk-size, strategy, and reranking grid:

```bash
uv run rag-eval-grid --top-k 5
```

Add `--no-generate` to skip answer generation. Outputs are written under
`exports/rag-experiments/`. The latest checked-in report found
`chunk_512 + vector + no rerank` to be the best tested configuration; see
[`exports/rag-experiments/report.md`](exports/rag-experiments/report.md) for
the full results and caveats.

Cohere trial keys allow 10 rerank calls per minute. To compare vector vs
rerank without 429s, run the paced 5-question check (retrieve 10, rerank to
5, 8s between Cohere calls):

```bash
uv run rag-eval-grid --rerank-compare
```

Results land in `exports/rag-experiments/rerank-compare/`.

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

