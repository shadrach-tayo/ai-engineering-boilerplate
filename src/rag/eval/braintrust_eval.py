"""Compare two RAG pipeline versions as named Braintrust experiments.

Layer 1 of the eval architecture (labeled retrieval) is cheap and deterministic:
source@k, page@k, phrase recall. That is what these experiments score.

Baseline is the retrieval-grid winner (chunk_512 vector @5, no rerank).
Candidate is the DeepEval generator-suite config (chunk_512 vector @10,
Cohere rerank to 5). Both are scored at k=5 so rows align in Compare.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from braintrust import Eval
from braintrust.framework import EvalCase as BraintrustEvalCase
from braintrust.framework import EvalResultWithSummary
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.braintrust", override=True)

from rag.eval.cases import CASES, EvalCase  # noqa: E402
from rag.eval.metrics import CaseScore, score_case  # noqa: E402
from rag.pipeline import (  # noqa: E402
    RagConfig,
    RagPipeline,
    RetrievalResult,
    RetrievalStrategy,
)

SCORE_TOP_K = 5
DEFAULT_PROJECT = "langgraph-mcp"
# Cohere trial keys allow 10 rerank calls/minute.
COHERE_TRIAL_PAUSE_S = 8.0


@dataclass(frozen=True)
class PipelineVersion:
    """One named retrieval configuration logged as a Braintrust experiment."""

    experiment: str
    description: str
    index_name: str
    strategy: RetrievalStrategy
    top_k: int
    rerank: bool
    rerank_top_n: int
    score_top_k: int = SCORE_TOP_K

    def rag_config(self) -> RagConfig:
        """Return the pipeline knobs for this version."""
        return RagConfig(
            index_name=self.index_name,
            strategy=self.strategy,
            top_k=self.top_k,
            rerank=self.rerank,
            rerank_top_n=self.rerank_top_n,
        )

    def metadata(self) -> dict[str, str | int | bool]:
        """Return experiment-level metadata for Braintrust filtering."""
        return {
            "index_name": self.index_name,
            "strategy": self.strategy,
            "top_k": self.top_k,
            "rerank": self.rerank,
            "rerank_top_n": self.rerank_top_n,
            "score_top_k": self.score_top_k,
        }


BASELINE = PipelineVersion(
    experiment="chunk512-vector-k5-norerank",
    description="Retrieval grid winner: Voyage vector @5, no Cohere rerank.",
    index_name="chunk_512",
    strategy="vector",
    top_k=5,
    rerank=False,
    rerank_top_n=5,
)

CANDIDATE = PipelineVersion(
    experiment="chunk512-vector-k10-rerank5",
    description="Generator-eval config: retrieve 10, Cohere rerank to 5.",
    index_name="chunk_512",
    strategy="vector",
    top_k=10,
    rerank=True,
    rerank_top_n=5,
)


def labeled_dataset() -> list[BraintrustEvalCase[str, dict[str, Any]]]:
    """Serialize the 10 labeled cases so Braintrust can match rows on input."""
    rows: list[BraintrustEvalCase[str, dict[str, Any]]] = []
    for case in CASES:
        rows.append(
            BraintrustEvalCase(
                input=case.question,
                expected={
                    "id": case.id,
                    "question": case.question,
                    "gold_answer": case.gold_answer,
                    "expected_sources": list(case.expected_sources),
                    "expected_pages": [list(page) for page in case.expected_pages],
                    "must_contain": list(case.must_contain),
                },
                metadata={
                    "case_id": case.id,
                    "sources": list(case.expected_sources),
                },
            )
        )
    return rows


def _eval_data() -> Iterator[BraintrustEvalCase[str, dict[str, Any]]]:
    """Yield labeled cases in the iterator shape Braintrust's Eval expects."""
    yield from labeled_dataset()


def case_from_expected(expected: dict[str, Any]) -> EvalCase:
    """Rebuild a labeled case from the JSON-serializable expected payload."""
    pages = expected["expected_pages"]
    return EvalCase(
        id=str(expected["id"]),
        question=str(expected["question"]),
        gold_answer=str(expected["gold_answer"]),
        expected_sources=tuple(str(source) for source in expected["expected_sources"]),
        expected_pages=tuple((str(item[0]), int(item[1])) for item in pages),
        must_contain=tuple(str(phrase) for phrase in expected["must_contain"]),
    )


def result_from_output(output: dict[str, Any]) -> RetrievalResult:
    """Rebuild a retrieval result from the task payload."""
    return RetrievalResult(
        docs=list(output.get("docs") or []),
        metadata=list(output.get("metadata") or []),
        strategy=str(output.get("strategy") or "vector"),
    )


def score_output(
    output: dict[str, Any],
    expected: dict[str, Any],
    *,
    top_k: int = SCORE_TOP_K,
) -> CaseScore:
    """Score one Braintrust task output against a labeled case."""
    return score_case(case_from_expected(expected), result_from_output(output), top_k=top_k)


def source_at_k(
    input: str,
    output: dict[str, Any],
    expected: dict[str, Any] | None,
) -> float:
    """Return 1 when a gold source appears in the scored window."""
    del input
    if expected is None:
        raise ValueError("source_at_k requires expected labels")
    return float(score_output(output, expected).source_hit)


def page_at_k(
    input: str,
    output: dict[str, Any],
    expected: dict[str, Any] | None,
) -> float:
    """Return 1 when a gold (source, page) appears in the scored window."""
    del input
    if expected is None:
        raise ValueError("page_at_k requires expected labels")
    return float(score_output(output, expected).page_hit)


def phrase_recall(
    input: str,
    output: dict[str, Any],
    expected: dict[str, Any] | None,
) -> float:
    """Return the fraction of required phrases found in retrieved text."""
    del input
    if expected is None:
        raise ValueError("phrase_recall requires expected labels")
    return score_output(output, expected).phrase_recall


def auto_relevant(
    input: str,
    output: dict[str, Any],
    expected: dict[str, Any] | None,
) -> float:
    """Return 1 when the hit is auto-relevant (page hit or source+phrases)."""
    del input
    if expected is None:
        raise ValueError("auto_relevant requires expected labels")
    return float(score_output(output, expected).auto_relevant)


RETRIEVAL_SCORERS = [source_at_k, page_at_k, phrase_recall, auto_relevant]


def _result_payload(result: RetrievalResult) -> dict[str, Any]:
    return {
        "docs": list(result.docs),
        "metadata": [
            {"source": meta.get("source"), "page": meta.get("page")}
            for meta in result.metadata
        ],
        "strategy": result.strategy,
        "n_docs": len(result.docs),
        "reranked": bool(result.rerank),
    }


def make_task(
    version: PipelineVersion,
    *,
    pause_s: float = 0.0,
    retrieve: Callable[[RagPipeline, str], RetrievalResult] | None = None,
) -> Callable[[str], dict[str, Any]]:
    """Return a Braintrust task that retrieves with ``version``'s config."""
    pipeline = RagPipeline(version.rag_config())
    run: Callable[[RagPipeline, str], RetrievalResult]
    if retrieve is None:
        pipeline.warmup(version.index_name)

        def live_retrieve(pipe: RagPipeline, question: str) -> RetrievalResult:
            return pipe.retrieve(question)

        run = live_retrieve
    else:
        run = retrieve

    def task(question: str) -> dict[str, Any]:
        result = run(pipeline, question)
        if version.rerank and pause_s > 0:
            time.sleep(pause_s)
        return _result_payload(result)

    return task


def _project_settings() -> tuple[str, str | None]:
    project = os.getenv("BRAINTRUST_PROJECT") or DEFAULT_PROJECT
    project_id = os.getenv("BRAINTRUST_PROJECT_ID") or None
    return project, project_id


def run_version(
    version: PipelineVersion,
    *,
    base_experiment: str | None = None,
    pause_s: float = 0.0,
    no_send_logs: bool = False,
    retrieve: Callable[[RagPipeline, str], RetrievalResult] | None = None,
) -> EvalResultWithSummary[str, dict[str, Any], dict[str, Any]]:
    """Log one named experiment and return the Braintrust summary."""
    project, project_id = _project_settings()
    return Eval(
        name=project,
        data=_eval_data,
        task=make_task(version, pause_s=pause_s, retrieve=retrieve),
        scores=RETRIEVAL_SCORERS,
        experiment_name=version.experiment,
        description=version.description,
        metadata=version.metadata(),
        tags=["retrieval", version.experiment],
        project_id=project_id,
        base_experiment_name=base_experiment,
        max_concurrency=1,
        no_send_logs=no_send_logs,
    )


def run_comparison(
    *,
    run_baseline: bool = True,
    run_candidate: bool = True,
    base_experiment: str = BASELINE.experiment,
    pause_s: float = COHERE_TRIAL_PAUSE_S,
    no_send_logs: bool = False,
    retrieve: Callable[[RagPipeline, str], RetrievalResult] | None = None,
) -> list[EvalResultWithSummary[str, dict[str, Any], dict[str, Any]]]:
    """Run baseline and/or candidate experiments; candidate compares to baseline."""
    results: list[EvalResultWithSummary[str, dict[str, Any], dict[str, Any]]] = []
    if run_baseline:
        results.append(
            run_version(
                BASELINE,
                pause_s=0.0,
                no_send_logs=no_send_logs,
                retrieve=retrieve,
            )
        )
    if run_candidate:
        results.append(
            run_version(
                CANDIDATE,
                base_experiment=base_experiment,
                pause_s=pause_s,
                no_send_logs=no_send_logs,
                retrieve=retrieve,
            )
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the named-experiment comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--baseline-only",
        action="store_true",
        help=f"Log only {BASELINE.experiment}.",
    )
    group.add_argument(
        "--candidate-only",
        action="store_true",
        help=f"Log only {CANDIDATE.experiment} against --base-experiment.",
    )
    parser.add_argument(
        "--base-experiment",
        default=BASELINE.experiment,
        help="Baseline experiment name attached to the candidate run.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=COHERE_TRIAL_PAUSE_S,
        help="Seconds to wait after each Cohere rerank call (trial keys).",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Skip the Cohere pacing sleep (paid keys).",
    )
    parser.add_argument(
        "--no-send-logs",
        action="store_true",
        help="Score locally without creating Braintrust experiments.",
    )
    return parser.parse_args(argv)


def _missing_keys(run_candidate: bool) -> list[str]:
    required = ["VOYAGE_API_KEY"]
    if run_candidate:
        required.append("COHERE_API_KEY")
    return [name for name in required if not os.getenv(name)]


def _has_errors(result: EvalResultWithSummary[str, dict[str, Any], dict[str, Any]]) -> bool:
    return any(row.error for row in result.results)


def _print_urls(
    results: Sequence[EvalResultWithSummary[str, dict[str, Any], dict[str, Any]]],
) -> None:
    for result in results:
        summary = result.summary
        url = getattr(summary, "experiment_url", None)
        name = getattr(summary, "experiment_name", None) or "experiment"
        if url:
            print(f"{name}: {url}")  # noqa: T201
        compared = getattr(summary, "comparison_experiment_name", None)
        if compared and url:
            print(f"Compare against {compared} from that experiment page.")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> int:
    """Run named Braintrust experiments for the two pipeline versions."""
    args = parse_args(argv)
    run_baseline = not args.candidate_only
    run_candidate = not args.baseline_only
    pause_s = 0.0 if args.no_pause else args.pause

    missing = _missing_keys(run_candidate)
    if missing:
        print(  # noqa: T201
            "Missing required keys: "
            + ", ".join(missing)
            + ". Add them to .env (see .env.example) and rerun."
        )
        return 1
    if not args.no_send_logs and not os.getenv("BRAINTRUST_API_KEY"):
        print(  # noqa: T201
            "BRAINTRUST_API_KEY is not set. Put it in .env.braintrust or the "
            "environment, or pass --no-send-logs to score locally."
        )
        return 1

    results = run_comparison(
        run_baseline=run_baseline,
        run_candidate=run_candidate,
        base_experiment=args.base_experiment,
        pause_s=pause_s,
        no_send_logs=args.no_send_logs,
    )
    _print_urls(results)
    if any(_has_errors(result) for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
