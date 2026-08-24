# Vector vs Cohere rerank (paced 5-question check)

Fair comparison: vector `@5` vs retrieve `@10` then Cohere rerank to 5. Questions: q01_agent_definition, q02_guardrails, q03_architecture_patterns, q05_evaluations, q09_coinbase_agents (openai-guide, claude-guide, support-agent). Pause 8.0s between Cohere calls (trial key is 10 requests/minute). **0 of 15 rerank calls 429'd.**

Wall-clock time includes the 8s pause after each Cohere call; `retrieve s` is query + rerank only.

| Index | Rerank | source@5 | page@5 | phrase recall | auto relevant | errors | retrieve (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| chunk_256 | no (top 5) | 100% | 100% | 100% | 100% | 0 | 1.2 |
| chunk_256 | yes (pool 10 → 5) | 100% | 100% | 100% | 100% | 0 | 3.5 |
| chunk_512 | no (top 5) | 100% | 100% | 100% | 100% | 0 | 0.9 |
| chunk_512 | yes (pool 10 → 5) | 100% | 100% | 100% | 100% | 0 | 3.7 |
| chunk_1024 | no (top 5) | 100% | 100% | 93% | 100% | 0 | 0.8 |
| chunk_1024 | yes (pool 10 → 5) | 100% | 100% | 87% | 100% | 0 | 3.7 |

## Verdict

Rerank did not beat raw vector on any index (2 ties, 1 vector win). The earlier grid claim holds under a fair, rate-limit-free protocol.

On `chunk_1024` / q03 (architecture patterns), rerank kept the gold page but dropped "decentralized" from the top-5 snippets (phrase recall 2/3 → 1/3). Voyage already had the gold hits; Cohere reordered them without adding recall, and cost ~3× retrieve latency even before the pacing sleeps.
