"""Run retrieval (and optional generation) across chunk sizes and strategies."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from rag.eval.cases import CASES, EvalCase  # noqa: E402
from rag.eval.metrics import CaseScore, score_case, summarize  # noqa: E402
from rag.pipeline import RagConfig, RagPipeline, RetrievalStrategy  # noqa: E402

INDEXES = ("chunk_256", "chunk_512", "chunk_1024")
STRATEGIES: tuple[RetrievalStrategy, ...] = ("vector", "hybrid", "ensemble")
# Five questions spanning openai-guide, claude-guide, and support-agent PDFs.
RERANK_COMPARE_IDS = (
    "q01_agent_definition",
    "q02_guardrails",
    "q03_architecture_patterns",
    "q05_evaluations",
    "q09_coinbase_agents",
)
# Cohere trial keys allow 10 rerank calls/minute.
COHERE_TRIAL_PAUSE_S = 8.0
MAIN_QUESTIONS = (
    "What is an Agent?",
    "What are the guardrails used in building AI agents?",
    "What are the Common architecture patterns for building AI Agents?",
    "What are the Common use cases and applications for AI agents?",
    "what are evaluations and how do we build evaluations for agentic systems?",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "exports" / "rag-experiments"


def _score_payload(score: CaseScore) -> dict[str, Any]:
    return {
        "id": score.case_id,
        "question": score.question,
        "gold_answer": score.gold_answer,
        "source_hit": score.source_hit,
        "page_hit": score.page_hit,
        "phrase_hits": score.phrase_hits,
        "phrase_misses": score.phrase_misses,
        "phrase_recall": score.phrase_recall,
        "auto_relevant": score.auto_relevant,
        "retrieved": score.retrieved,
    }


def _is_cohere_rate_limit(exc: BaseException) -> bool:
    text = str(exc)
    return "429" in text or "10 API calls / minute" in text


def _retrieve_with_retry(
    pipeline: RagPipeline,
    question: str,
    *,
    index_name: str,
    strategy: RetrievalStrategy,
    top_k: int,
    rerank: bool,
    rerank_top_n: int | None,
    max_retries: int,
    pause_s: float,
) -> Any:
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return pipeline.retrieve(
                question,
                index_name=index_name,
                strategy=strategy,
                top_k=top_k,
                rerank=rerank,
                rerank_top_n=rerank_top_n,
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_cohere_rate_limit(exc) or attempt == max_retries:
                raise
            wait = max(pause_s, 15.0) * (attempt + 1)
            time.sleep(wait)
    raise last or RuntimeError("retrieve retry exhausted")


def run_retrieval_config(
    pipeline: RagPipeline,
    *,
    index_name: str,
    strategy: RetrievalStrategy,
    rerank: bool,
    top_k: int,
    cases: tuple[EvalCase, ...] | None = None,
    retrieve_top_k: int | None = None,
    rerank_top_n: int | None = None,
    pause_s: float = 0.0,
    max_retries: int = 0,
) -> dict[str, Any]:
    """Retrieve and score labeled cases for one config."""
    eval_cases = cases or CASES
    pool_k = retrieve_top_k or top_k
    rows: list[dict[str, Any]] = []
    scores: list[CaseScore] = []
    started = time.perf_counter()
    for case in eval_cases:
        case_started = time.perf_counter()
        try:
            result = _retrieve_with_retry(
                pipeline,
                case.question,
                index_name=index_name,
                strategy=strategy,
                top_k=pool_k,
                rerank=rerank,
                rerank_top_n=rerank_top_n,
                max_retries=max_retries,
                pause_s=pause_s,
            )
            score = score_case(case, result, top_k=top_k)
            scores.append(score)
            rows.append(
                {
                    **_score_payload(score),
                    "elapsed_ms": round((time.perf_counter() - case_started) * 1000),
                    "payload": result.as_vector_payload(),
                    "es_payload": result.as_es_payload() if result.es_hits else None,
                    "strategy_used": result.strategy,
                    "error": None,
                }
            )
            if rerank and pause_s > 0:
                time.sleep(pause_s)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "error": str(exc),
                    "elapsed_ms": round((time.perf_counter() - case_started) * 1000),
                }
            )
            if rerank and pause_s > 0 and _is_cohere_rate_limit(exc):
                time.sleep(max(pause_s, 20.0))
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    summary = summarize(scores) if scores else {}
    summary["n_errors"] = sum(1 for row in rows if row.get("error"))
    summary["elapsed_ms"] = elapsed_ms
    summary["retrieve_ms"] = sum(int(row.get("elapsed_ms") or 0) for row in rows)
    return {
        "index_name": index_name,
        "strategy": strategy,
        "rerank": rerank,
        "top_k": top_k,
        "retrieve_top_k": pool_k,
        "rerank_top_n": rerank_top_n,
        "summary": summary,
        "cases": rows,
    }


def run_generation_config(
    pipeline: RagPipeline,
    *,
    index_name: str,
    strategy: RetrievalStrategy,
    rerank: bool,
    top_k: int,
    questions: tuple[str, ...] = MAIN_QUESTIONS,
) -> dict[str, Any]:
    """Generate answers for the original main.py question list."""
    answers: list[dict[str, Any]] = []
    started = time.perf_counter()
    for question in questions:
        case_started = time.perf_counter()
        try:
            payload = pipeline.generate(
                question,
                index_name=index_name,
                strategy=strategy,
                top_k=top_k,
                rerank=rerank,
            )
            answers.append(
                {
                    "question": question,
                    "content": payload.get("content"),
                    "docs": payload.get("docs"),
                    "reranks": payload.get("reranks"),
                    "es_docs": payload.get("es_docs"),
                    "strategy": payload.get("strategy"),
                    "elapsed_ms": round((time.perf_counter() - case_started) * 1000),
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            answers.append(
                {
                    "question": question,
                    "error": str(exc),
                    "elapsed_ms": round((time.perf_counter() - case_started) * 1000),
                }
            )
    return {
        "index_name": index_name,
        "strategy": strategy,
        "rerank": rerank,
        "top_k": top_k,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "answers": answers,
    }


def write_markdown(leaderboard: list[dict[str, Any]], out_path: Path) -> None:
    """Write a compact ranking of retrieval configs."""
    lines = [
        "# RAG pipeline experiment",
        "",
        "Labeled 10-question retrieval eval across chunk indexes and retriever strategies.",
        "",
        "| Rank | Index | Strategy | Rerank | source@k | page@k | phrase recall | auto relevant | errors | time (s) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(leaderboard, start=1):
        summary = row["summary"]
        lines.append(
            "| {rank} | {index} | {strategy} | {rerank} | {source:.0%} | {page:.0%} | "
            "{phrase:.0%} | {auto:.0%} | {errors} | {secs:.1f} |".format(
                rank=i,
                index=row["index_name"],
                strategy=row["strategy"],
                rerank="yes" if row["rerank"] else "no",
                source=summary.get("source_hit_at_k") or 0,
                page=summary.get("page_hit_at_k") or 0,
                phrase=summary.get("phrase_recall") or 0,
                auto=summary.get("auto_relevant") or 0,
                errors=int(summary.get("n_errors") or 0),
                secs=(summary.get("elapsed_ms") or 0) / 1000,
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grid(
    *,
    out_dir: Path = DEFAULT_OUT,
    top_k: int = 5,
    generate: bool = True,
) -> dict[str, Any]:
    """Sweep indexes, strategies, and rerank; persist every run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    gen_dir = out_dir / "generations"
    runs_dir.mkdir(exist_ok=True)
    gen_dir.mkdir(exist_ok=True)

    pipeline = RagPipeline(RagConfig(top_k=top_k, rerank=False))
    retrieval_runs: list[dict[str, Any]] = []
    generation_runs: list[dict[str, Any]] = []

    for index_name in INDEXES:
        pipeline.warmup(index_name)
        for strategy in STRATEGIES:
            for rerank in (False, True):
                slug = f"{index_name}_{strategy}_{'rerank' if rerank else 'norerank'}"
                run = run_retrieval_config(
                    pipeline,
                    index_name=index_name,
                    strategy=strategy,
                    rerank=rerank,
                    top_k=top_k,
                )
                (runs_dir / f"{slug}.json").write_text(
                    json.dumps(run, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                retrieval_runs.append(run)
                if generate and rerank:
                    gen = run_generation_config(
                        pipeline,
                        index_name=index_name,
                        strategy=strategy,
                        rerank=rerank,
                        top_k=top_k,
                    )
                    (gen_dir / f"{slug}.json").write_text(
                        json.dumps(gen, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    generation_runs.append(gen)

    leaderboard = sorted(
        retrieval_runs,
        key=lambda row: (
            -(row["summary"].get("page_hit_at_k") or 0),
            -(row["summary"].get("source_hit_at_k") or 0),
            -(row["summary"].get("phrase_recall") or 0),
            row["summary"].get("elapsed_ms") or 0,
        ),
    )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "top_k": top_k,
        "indexes": list(INDEXES),
        "strategies": list(STRATEGIES),
        "n_cases": len(CASES),
        "leaderboard": [
            {
                "index_name": row["index_name"],
                "strategy": row["strategy"],
                "rerank": row["rerank"],
                "summary": row["summary"],
            }
            for row in leaderboard
        ],
        "generation_runs": [
            {
                "index_name": row["index_name"],
                "strategy": row["strategy"],
                "rerank": row["rerank"],
                "elapsed_ms": row["elapsed_ms"],
                "n_answers": len(row["answers"]),
                "n_errors": sum(1 for a in row["answers"] if a.get("error")),
            }
            for row in generation_runs
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(leaderboard, out_dir / "report.md")
    return payload


def _rerank_compare_cases() -> tuple[EvalCase, ...]:
    by_id = {case.id: case for case in CASES}
    return tuple(by_id[case_id] for case_id in RERANK_COMPARE_IDS)


def write_rerank_compare_markdown(payload: dict[str, Any], out_path: Path) -> None:
    """Write the paced vector vs Cohere rerank comparison."""
    protocol = payload["protocol"]
    lines = [
        "# Vector vs Cohere rerank (paced 5-question check)",
        "",
        "Fair comparison: vector `@5` vs retrieve `@10` then Cohere rerank to 5. "
        f"Questions: {', '.join(protocol['case_ids'])}. "
        f"Pause {protocol['pause_s']}s between Cohere calls "
        f"(trial key is 10 requests/minute).",
        "",
        "Wall-clock time includes the 8s pause after each Cohere call; "
        "`retrieve s` is query + rerank only.",
        "",
        "| Index | Rerank | source@5 | page@5 | phrase recall | auto relevant | errors | retrieve (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        summary = row["summary"]
        retrieve_ms = summary.get("retrieve_ms") or summary.get("elapsed_ms") or 0
        lines.append(
            "| {index} | {rerank} | {source:.0%} | {page:.0%} | "
            "{phrase:.0%} | {auto:.0%} | {errors} | {secs:.1f} |".format(
                index=row["index_name"],
                rerank="yes (pool 10 → 5)" if row["rerank"] else "no (top 5)",
                source=summary.get("source_hit_at_k") or 0,
                page=summary.get("page_hit_at_k") or 0,
                phrase=summary.get("phrase_recall") or 0,
                auto=summary.get("auto_relevant") or 0,
                errors=int(summary.get("n_errors") or 0),
                secs=retrieve_ms / 1000,
            )
        )
    verdict = payload.get("verdict") or {}
    if verdict:
        lines.extend(
            [
                "",
                "## Verdict",
                "",
                verdict.get("text", ""),
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rerank_verdict(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare paired vector vs rerank summaries; require zero Cohere errors."""
    by_index: dict[str, dict[str, dict[str, Any]]] = {}
    for row in runs:
        by_index.setdefault(row["index_name"], {})["rerank" if row["rerank"] else "base"] = row
    errors = sum(int(row["summary"].get("n_errors") or 0) for row in runs)
    wins = 0
    ties = 0
    losses = 0
    pairs: list[dict[str, Any]] = []
    for index_name, pair in by_index.items():
        base = pair["base"]["summary"]
        rerank = pair["rerank"]["summary"]
        delta_page = (rerank.get("page_hit_at_k") or 0) - (base.get("page_hit_at_k") or 0)
        delta_phrase = (rerank.get("phrase_recall") or 0) - (base.get("phrase_recall") or 0)
        if delta_page > 0 or (delta_page == 0 and delta_phrase > 0):
            wins += 1
            outcome = "rerank better"
        elif delta_page == 0 and delta_phrase == 0:
            ties += 1
            outcome = "tie"
        else:
            losses += 1
            outcome = "vector better"
        pairs.append(
            {
                "index_name": index_name,
                "delta_page": delta_page,
                "delta_phrase": delta_phrase,
                "outcome": outcome,
            }
        )
    if errors:
        text = (
            f"Incomplete: {errors} Cohere errors. Do not treat rerank metrics as conclusive."
        )
    elif wins and not losses:
        text = (
            f"Rerank improved {wins} of {len(pairs)} indexes on this 5-question set "
            f"({ties} ties)."
        )
    elif losses and not wins:
        text = (
            f"Rerank did not beat raw vector on any index "
            f"({ties} ties, {losses} vector {'win' if losses == 1 else 'wins'}). "
            "The earlier grid claim holds under a fair, rate-limit-free protocol."
        )
    elif ties == len(pairs):
        text = (
            "Rerank tied raw vector on every index. It added latency without "
            "improving source/page/phrase hits on this 5-question set."
        )
    else:
        text = (
            f"Mixed: rerank better on {wins}, vector better on {losses}, "
            f"{ties} ties."
        )
    return {
        "errors": errors,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "pairs": pairs,
        "text": text,
    }


def run_rerank_compare(
    *,
    out_dir: Path | None = None,
    top_k: int = 5,
    retrieve_top_k: int = 10,
    pause_s: float = COHERE_TRIAL_PAUSE_S,
) -> dict[str, Any]:
    """Vector @k vs retrieve-pool then Cohere rerank to k, paced for trial keys."""
    dest = out_dir or (DEFAULT_OUT / "rerank-compare")
    dest.mkdir(parents=True, exist_ok=True)
    cases = _rerank_compare_cases()
    pipeline = RagPipeline(RagConfig(top_k=top_k, rerank=False, rerank_top_n=top_k))
    retrieval_runs: list[dict[str, Any]] = []
    for index_name in INDEXES:
        pipeline.warmup(index_name)
        for rerank in (False, True):
            slug = f"{index_name}_vector_{'rerank' if rerank else 'norerank'}"
            print(  # noqa: T201
                f"{index_name} vector rerank={rerank} "
                f"({'pool 10→5, pause ' + str(pause_s) + 's' if rerank else 'top 5'})"
            )
            run = run_retrieval_config(
                pipeline,
                index_name=index_name,
                strategy="vector",
                rerank=rerank,
                top_k=top_k,
                cases=cases,
                retrieve_top_k=retrieve_top_k if rerank else top_k,
                rerank_top_n=top_k if rerank else None,
                pause_s=pause_s if rerank else 0.0,
                max_retries=3 if rerank else 0,
            )
            (dest / f"{slug}.json").write_text(
                json.dumps(run, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            retrieval_runs.append(run)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "strategy": "vector",
            "score_top_k": top_k,
            "baseline_retrieve_top_k": top_k,
            "rerank_retrieve_top_k": retrieve_top_k,
            "rerank_top_n": top_k,
            "pause_s": pause_s,
            "case_ids": list(RERANK_COMPARE_IDS),
            "n_cases": len(cases),
            "indexes": list(INDEXES),
        },
        "runs": [
            {
                "index_name": row["index_name"],
                "strategy": row["strategy"],
                "rerank": row["rerank"],
                "retrieve_top_k": row.get("retrieve_top_k"),
                "rerank_top_n": row.get("rerank_top_n"),
                "summary": row["summary"],
            }
            for row in retrieval_runs
        ],
        "verdict": _rerank_verdict(retrieval_runs),
    }
    (dest / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_rerank_compare_markdown(payload, dest / "report.md")
    return payload


def main() -> None:
    """CLI for the chunk × strategy experiment grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument(
        "--rerank-compare",
        action="store_true",
        help="Paced 5-question vector vs Cohere rerank check (avoids trial 429s).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=COHERE_TRIAL_PAUSE_S,
        help="Seconds to wait after each Cohere rerank call.",
    )
    args = parser.parse_args()
    if args.rerank_compare:
        dest = args.out if args.out != DEFAULT_OUT else DEFAULT_OUT / "rerank-compare"
        payload = run_rerank_compare(
            out_dir=dest,
            top_k=args.top_k,
            pause_s=args.pause,
        )
        print(f"Wrote {dest / 'summary.json'}")  # noqa: T201
        print(payload["verdict"]["text"])  # noqa: T201
        return
    payload = run_grid(
        out_dir=args.out,
        top_k=args.top_k,
        generate=not args.no_generate,
    )
    print(f"Wrote {args.out / 'summary.json'}")  # noqa: T201
    best = payload["leaderboard"][0] if payload["leaderboard"] else None
    if best:
        print(  # noqa: T201
            "Best page@k: {index} / {strategy} / rerank={rerank} "
            "({page:.0%} page, {source:.0%} source)".format(
                index=best["index_name"],
                strategy=best["strategy"],
                rerank=best["rerank"],
                page=best["summary"].get("page_hit_at_k") or 0,
                source=best["summary"].get("source_hit_at_k") or 0,
            )
        )


if __name__ == "__main__":
    main()
