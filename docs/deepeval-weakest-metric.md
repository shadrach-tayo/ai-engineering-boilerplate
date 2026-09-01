# Answer Correctness is the weakest metric

This note is source material for a blog post about reading a DeepEval run and deciding what to change next. It is a close read of one scored run, not the full experiment history. For the measure → change → measure loop, see [RAG eval loop](./rag-eval-loop-journey.md).

**Source.** `data/deepeval/test_run_20260827_132733.json`. 30 goldens. DeepSeek generator and judge. Run identifier `rag-latest-dataset-main`. 20 of 30 cases passed overall; every remaining failure is Answer Correctness, sometimes with a retrieval miss.

This is the **native 7-metric baseline**: vector retrieval, `chunk_512`, top_k=5, rerank off.

---

## Headline numbers

| Stat | Value |
| --- | --- |
| Answer Correctness pass rate | 50% |
| Mean score vs 0.7 threshold | 0.675 |
| Answerable cases failed | 10 / 20 |
| Edge goldens failed this metric | 8 / 10 |

**What this is not.** Faithfulness, Answer Relevancy, Hallucination, and Unsupported Refusal all passed 100%. The generator is grounded and refuses out-of-domain questions. It is missing labeled claims, especially on dense edge goldens, under a three-sentence generator prompt.

---

## Pass rate by metric

Higher is better. Same JSON run.

| Metric | Pass rate |
| --- | ---: |
| Answer Correctness | 50% |
| Contextual Precision | 95% |
| Contextual Recall | 95% |
| Faithfulness | 100% |
| Answer Relevancy | 100% |
| Hallucination | 100% |
| Unsupported Refusal | 100% |

---

## Mean score vs threshold

| Metric | n | Pass | Fail | Mean score | Min | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Answer Correctness [GEval] | 20 | 10 | 10 | 0.675 | 0.20 | 0.70 |
| Contextual Precision | 20 | 19 | 1 | 0.936 | 0.42 | 0.50 |
| Contextual Recall | 20 | 19 | 1 | 0.950 | 0.00 | 0.50 |
| Faithfulness | 20 | 20 | 0 | 0.992 | 0.83 | 0.70 |
| Answer Relevancy | 20 | 20 | 0 | 0.974 | 0.80 | 0.70 |
| Hallucination | 20 | 20 | 0 | 1.000 | 1.00 | 0.30 |
| Unsupported Refusal [GEval] | 10 | 10 | 0 | 1.000 | 1.00 | 0.70 |

---

## Where Answer Correctness fails

Pass rate by dataset slice. Refusal goldens do not run this metric.

| Slice | Answer Correctness pass rate |
| --- | ---: |
| Normal (`q01`–`q10`) | 80% |
| Edge | 20% |
| Failure / refusal | n/a (metric not run; refusal GEval is 100%) |

### Why the edge slice tanks

Expected outputs for edge cases pack many named claims: section citations, KPI figures, vendor comparisons. The generator prompt caps answers at three sentences, so the judge scores **omissions** even when Faithfulness is 1.0.

Two retrieval misses amplify that:

- `q03` ranks relevant chunks below a table of contents (precision 0.42).
- `edge_06` retrieves none of the labeled claims (recall 0.00).

Later loops fixed those retrieval misses (Cohere rerank for `q03`; top_k=10 then rerank to 5 for `edge_06`). Answer Correctness only moved from 10/20 to 13/20. The remaining gap is generation, not ranking.

---

## Failing Answer Correctness cases

All ten failures from this baseline run:

| Case | Slice | Score | What the judge docked |
| --- | --- | ---: | --- |
| `edge_10` | edge | 0.20 | Missed Claude benchmark KPIs; claimed the docs lack that data |
| `edge_06` | edge | 0.30 | Did not recommend a single-agent start; recall of labeled claims is 0 |
| `q05_evaluations` | normal | 0.50 | Omitted offline LLM-as-judge, prompt iteration, and A/B validation |
| `edge_07` | edge | 0.50 | Missed required section citations and the translation SDK example |
| `q03_architecture_patterns` | normal | 0.60 | Dropped agentic workflows; retrieval ranked relevant chunks 3rd/4th |
| `edge_01` | edge | 0.60 | Covered tools and safeguards but omitted several labeled claims |
| `edge_03` | edge | 0.60 | Partial coverage of eval/ops claims in a multi-part question |
| `edge_05` | edge | 0.60 | Layered guardrails present; missing several named checks |
| `edge_08` | edge | 0.60 | Missed rollout gate metrics (SSR, tNPS, escalation) |
| `edge_09` | edge | 0.60 | Hit headline rates; omitted coverage and autonomy details |

Pattern: the judge is not calling the answers unfaithful. It is calling them **incomplete** relative to a dense gold label.

---

## Highest-leverage next step

That experiment is now run. Dropping the three-sentence cap, with retrieval held at k=10 → rerank 5 and **no** query decomposition, moved Answer Correctness **13/20 → 18/20** (`rag-no-sentence-cap-eval`, `test_run_20260901_154149.json`). Remaining fails: `q06` (0.60) and `edge_01` (0.50).

```bash
RAG_EVAL_PROMPT=no-sentence-cap uv run rag-eval-generator --experiment no-sentence-cap
```
