"""Run the 10-question retrieval eval and print a manual review scorecard."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag.eval.cases import CASES  # noqa: E402
from rag.eval.metrics import CaseScore, score_case, summarize  # noqa: E402
from rag.pipeline import RagConfig, RagPipeline, RetrievalStrategy  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "retrieval_eval_report.md"


def _yn(value: bool) -> str:
    return "Y" if value else "N"


def format_scorecard(scores: list[CaseScore], summary: dict[str, float], *, top_k: int) -> str:
    """Build a markdown scorecard for terminal or file review."""
    lines = [
        f"# Retrieval eval (top_k={top_k})",
        "",
        "| id | source@k | page@k | phrases | auto relevant |",
        "|---|---|---|---|---|",
    ]
    for score in scores:
        lines.append(
            f"| {score.case_id} | {_yn(score.source_hit)} | {_yn(score.page_hit)} | "
            f"{score.phrases_found} | {_yn(score.auto_relevant)} |"
        )
    lines.extend(
        [
            "",
            f"- source hit@{top_k}: **{summary['source_hit_at_k']:.0%}**",
            f"- page hit@{top_k}: **{summary['page_hit_at_k']:.0%}**",
            f"- mean phrase recall: **{summary['phrase_recall']:.0%}**",
            f"- auto-relevant: **{summary['auto_relevant']:.0%}**",
            "",
            "Mark `human_relevant` while reading snippets below. "
            "A hit is useful if the gold answer can be supported from the retrieved text.",
            "",
        ]
    )
    for score in scores:
        lines.extend(
            [
                f"## {score.case_id}",
                f"**Q:** {score.question}",
                "",
                f"**Gold:** {score.gold_answer}",
                "",
                f"**Missed phrases:** {', '.join(score.phrase_misses) or '(none)'}",
                "",
            ]
        )
        for hit in score.retrieved:
            lines.append(
                f"{hit['rank']}. `{hit['source']}` p{hit['page']} — {hit['snippet']}"
            )
        lines.append("")
    return "\n".join(lines)


def run_eval(
    *,
    index_name: str = "chunk_256",
    strategy: RetrievalStrategy = "vector",
    top_k: int = 5,
    rerank: bool = True,
) -> tuple[list[CaseScore], dict[str, float]]:
    """Retrieve for every labeled question and score the hits."""
    pipeline = RagPipeline(
        RagConfig(
            index_name=index_name,
            strategy=strategy,
            top_k=top_k,
            rerank=rerank,
        )
    )
    scores = [
        score_case(
            case,
            pipeline.retrieve(
                case.question,
                index_name=index_name,
                strategy=strategy,
                top_k=top_k,
                rerank=rerank,
            ),
            top_k=top_k,
        )
        for case in CASES
    ]
    return scores, summarize(scores)


def main() -> None:
    """Print and save a retrieval scorecard for manual accuracy review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="chunk_256")
    parser.add_argument(
        "--strategy",
        default="vector",
        choices=("vector", "hybrid", "ensemble"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    if args.strategy in {"vector", "ensemble"} and not os.environ.get("VOYAGE_API_KEY"):
        print(  # noqa: T201
            "VOYAGE_API_KEY is not set. Add it to .env (see .env.example) "
            "or export it in your shell, then rerun rag-eval."
        )
        sys.exit(1)
    scores, summary = run_eval(
        index_name=args.index,
        strategy=args.strategy,
        top_k=args.top_k,
        rerank=not args.no_rerank,
    )
    report = format_scorecard(scores, summary, top_k=args.top_k)
    print(report)  # noqa: T201
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    payload = {
        "summary": summary,
        "scores": [
            {
                "id": score.case_id,
                "source_hit": score.source_hit,
                "page_hit": score.page_hit,
                "phrase_recall": score.phrase_recall,
                "auto_relevant": score.auto_relevant,
                "human_relevant": score.human_relevant,
            }
            for score in scores
        ],
    }
    REPORT_PATH.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")  # noqa: T201


if __name__ == "__main__":
    main()
