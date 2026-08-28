# RAG eval loop: measure, change, measure again

This note is source material for a blog post about iterating on a RAG pipeline with DeepEval. It records what we measured, what we changed, and what actually moved.

**Setup.** Agent-guide RAG over the `chunk_512` Postgres index. Vector retrieval. Generator and judge: DeepSeek `deepseek-chat`. Dataset: 30 goldens (10 normal, 10 edge, 10 refusal). Artifacts live under `data/deepeval/`. Date: 27 Aug 2026.

**Current result.** Suite **23 / 30**. Contextual Recall **20 / 20**. Contextual Precision **20 / 20**. Answer Correctness **13 / 20** — still the weakest metric.

Run the scored suite with:

```bash
uv run rag-eval-generator --experiment <name>
```

Live retrieval and generation happen in `src/rag/eval/tests/test_faithfulness.py`. Metric definitions live in `src/rag/eval/tests/metrics.py`.

---

## The loop, in one sentence

Measure retrieval and generation with a fixed golden set, form a hypothesis about the failure, change one retrieval or prompt knob, then measure again.

| Goal | Question the metric asks | Status |
| --- | --- | --- |
| Contextual Recall | Are the relevant chunks retrieved at all? Native DeepEval, threshold 0.5. | Done. Baseline missed `edge_06`; retrieve 10 then rerank to 5 recovers it. |
| Contextual Precision | Are retrieved chunks useful and well ranked? Native DeepEval, threshold 0.5. | Done. Cohere rerank fixed `q03` (0.42 → 1.00). |
| Answer Correctness | Is the final answer factually right vs `expected_output`? GEval, threshold 0.7. | Implemented; still 13/20. |
| Full 7-metric suite | Logged to `data/deepeval/test_run_<timestamp>.json`. | Done. |
| Experiment: higher top-k | Does increasing top-k improve recall? | Yes, if rerank then cuts to 5. Stuffing all 10 into the prompt hurts correctness. |
| Experiment: rerank | Does reranking improve precision? | Yes. `q03` 0.42 → 1.00; pass 19/20 → 20/20; mean 0.936 → 0.969 at k=5. |
| Experiment: longer answers | Does dropping the three-sentence cap (or adding chain-of-thought) improve Answer Correctness? | Not run yet. |

---

## How the seven metrics are implemented

RAGAS Contextual Precision / Recall were tried first (`RAGASContextual*` in DeepEval). `ragas` 0.4 imports `langchain_community.chat_models.vertexai` and crashes. DeepEval reports that as “Please install ragas,” which aborted the whole case. The suite now uses DeepEval’s native `ContextualPrecisionMetric` and `ContextualRecallMetric`, which accept the DeepSeek judge. Answer Correctness is GEval vs `expected_output` (DeepEval has no RAGAS answer-correctness wrapper).

| Metric | Asks | Threshold | Runs on | Module |
| --- | --- | --- | --- | --- |
| Faithfulness | Are answer claims in the retrieved context? | 0.7 | Answerable (20) | `FaithfulnessMetric` |
| Answer Relevancy | Does the answer address the question? | 0.7 | Answerable (20) | `AnswerRelevancyMetric` |
| Hallucination | Does the answer contradict retrieved context? | 0.3 | Answerable (20) | `HallucinationMetric` |
| Contextual Precision | Are useful chunks ranked above noise? | 0.5 | Answerable (20) | `ContextualPrecisionMetric` |
| Contextual Recall | Do retrieved chunks cover expected claims? | 0.5 | Answerable (20) | `ContextualRecallMetric` |
| Answer Correctness | Does the answer match labeled gold claims? | 0.7 | Answerable (20) | `GEval` |
| Unsupported Refusal | Does the model refuse out-of-domain inputs? | 0.7 | Failure goldens (10) | `GEval` |

DeepEval 4.x flipped Hallucination to an alignment score (higher is better). CI gates on contradiction rate = `1 - mean`.

---

## Experiment impact on the 20 answerable goldens

Pass rate (%) after the native-metric switch:

| Run | Contextual Recall | Contextual Precision | Answer Correctness |
| --- | ---: | ---: | ---: |
| Baseline k=5, no rerank | 95 | 95 | 50 |
| Rerank k=5, n=5 | 95 | 100 | 65 |
| k=10 stuffed into the prompt | 100 | 100 | 45 |
| k=10 then rerank to 5 (current) | 100 | 100 | 65 |

Pass rate hid a quality drop when all 10 chunks went into the prompt. Precision **mean** (0–1):

| Run | Precision mean |
| --- | ---: |
| Baseline k=5, no rerank | 0.936 |
| Rerank k=5, n=5 | 0.969 |
| k=10 into the prompt | 0.907 |
| k=10 then n=5 | 0.963 |

---

## Loop cycles

### Cycle 0 — the suite could not score

- **Measure:** 4 pass / 26 fail (`test_run_20260827_131956.json`, identifier `rag-deepseek-chat-main`).
- **Hypothesis:** RAGAS wrappers need the `ragas` package.
- **Change:** Native Contextual Precision / Recall; drop Hallucination / Faithfulness on refusals.
- **Result:** The suite scores. RAGAS was a nested import crash, not a missing install. Smoking gun: DeepEval RAGAS wrappers crashed on `ragas` → `langchain_community.vertexai`. Native metrics never ran.

### Cycle 1 — ranking, not coverage

- **Measure:** Precision 19/20, Recall 19/20, Correctness 10/20 (`test_run_20260827_132733.json`, identifier `rag-latest-dataset-main`). k=5, rerank off.
- **Hypothesis:** Table-of-contents / noise ranked above architecture chunks.
- **Change:** Enable Cohere rerank, keep k=5 (`rerank_top_n=5`).
- **Result:** Precision 20/20. `q03` 0.42 → 1.00. Recall unchanged (`edge_06` still 0). Correctness up 3 cases to 13/20, still the limiter. File: `test_run_20260827_140101.json`, identifier `rag-rerank-on-main`.

### Cycle 2 — the missing chunk was outside top-5

- **Measure:** `edge_06` recall 0.00.
- **Hypothesis:** The relevant chunk is outside top-5.
- **Change:** `top_k=10` and `rerank_top_n=10` (all 10 chunks in the prompt).
- **Result:** Recall 20/20. Precision mean 0.969 → 0.907. Correctness 13 → 9/20. File: `test_run_20260827_141441.json`, identifier `rag-topk-10-main`. Smoking gun: `edge_06` recall 0 → 1, but stuffing 10 chunks into the prompt dropped precision mean and correctness.

### Cycle 3 — extra candidates, not extra prompt stuffing

- **Measure:** Recall win, correctness regression.
- **Hypothesis:** We need the extra candidate for recall, not 10 chunks at generation time.
- **Change:** `top_k=10`, `rerank_top_n=5`.
- **Result:** Recall 20/20, precision mean 0.963, correctness back to 13/20. `edge_06` correctness 0.50 → 0.80. File: `test_run_20260827_142224.json`, identifier `rag-topk-10-rerank-5-main`. This is the current config.

### Cycle 4 — generation, not retrieval (not run)

- **Measure:** Correctness still 13/20.
- **Hypothesis:** The three-sentence generator prompt omits labeled claims even when retrieval is complete.
- **Change:** Chain-of-thought, or drop the sentence cap.
- **Result:** Not run. Intended command: `uv run rag-eval-generator --experiment cot-prompt`.

---

## Run cards

| Run | Identifier | File | top_k | rerank_top_n | Suite | Recall | Precision | Correctness |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| RAGAS import fail | `rag-deepseek-chat-main` | `test_run_20260827_131956.json` | 5 | off (n=3 unused) | 4 / 30 | 0/20 (error) | 0/20 (error) | 0/20 (error) |
| Native 7-metric baseline | `rag-latest-dataset-main` | `test_run_20260827_132733.json` | 5 | off | 20 / 30 | 19/20, mean 0.950 | 19/20, mean 0.936 | 10/20, mean 0.675 |
| Rerank on, k=5 n=5 | `rag-rerank-on-main` | `test_run_20260827_140101.json` | 5 | 5 | 23 / 30 | 19/20, mean 0.950 | 20/20, mean 0.969 | 13/20, mean 0.695 |
| k=10 stuffed into prompt | `rag-topk-10-main` | `test_run_20260827_141441.json` | 10 | 10 | 19 / 30 | 20/20, mean 1.000 | 20/20, mean 0.907 | 9/20, mean 0.680 |
| Current: k=10, rerank to 5 | `rag-topk-10-rerank-5-main` | `test_run_20260827_142224.json` | 10 | 5 | 23 / 30 | 20/20, mean 1.000 | 20/20, mean 0.963 | 13/20, mean 0.690 |

---

## What we learned

**Retrieval is mostly solved.** Rerank fixes ranking (precision). Retrieving 10 then keeping 5 fixes coverage (recall) without drowning the generator. Faithfulness, relevancy, hallucination, and out-of-domain refusal stay at or near 100%.

**Generation is the remaining gap.** Answer Correctness still fails 7 of 20 answerable goldens, mostly dense edge `expected_output`s versus “Use three sentences maximum.” See [Answer Correctness is the weakest metric](./deepeval-weakest-metric.md) for the case-level breakdown from the baseline run.

**Current eval knobs** (`test_faithfulness.py` `RAG_CONFIG`): strategy `vector`, index `chunk_512`, `top_k=10`, `rerank=True`, `rerank_top_n=5`, Cohere `rerank-v4.0-pro`. Generator prompt is still three sentences. Judge and generator: DeepSeek `deepseek-chat`.
