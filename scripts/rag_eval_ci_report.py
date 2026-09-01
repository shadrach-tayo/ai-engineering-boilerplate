"""Turn a DeepEval JSON run into a PR comment and CI quality gates."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rag.eval.tests.metrics import CI_FAITHFULNESS_MIN, CI_HALLUCINATION_RATE_MAX

COMMENT_MARKER = "<!-- rag-eval-report -->"
FAITHFULNESS_NAME = "Faithfulness"
HALLUCINATION_NAME = "Hallucination"


def latest_run(results_dir: Path) -> Path:
    """Return the newest DeepEval test_run JSON under ``results_dir``."""
    runs = sorted(results_dir.glob("test_run_*.json"))
    if not runs:
        raise FileNotFoundError(f"no test_run_*.json files in {results_dir}")
    return runs[-1]


def _metric_rows(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for entry in payload.get("metricsScores") or []:
        name = entry.get("metric")
        if not name:
            continue
        scores = [score for score in entry.get("scores") or [] if score is not None]
        rows[name] = {
            "pass": int(entry.get("passes") or 0),
            "fail": int(entry.get("fails") or 0),
            "error": int(entry.get("errors") or 0),
            "scores": scores,
            "mean": statistics.mean(scores) if scores else None,
        }
    return rows


def hallucination_rate(alignment_mean: float | None) -> float | None:
    """Convert DeepEval 4.x Hallucination alignment into a contradiction rate."""
    if alignment_mean is None:
        return None
    return 1.0 - alignment_mean


def evaluate_gates(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    faithfulness_min: float = CI_FAITHFULNESS_MIN,
    hallucination_rate_max: float = CI_HALLUCINATION_RATE_MAX,
) -> tuple[bool, list[str], float | None, float | None]:
    """Return whether gates passed, failure reasons, and the two gate values."""
    faithfulness = rows.get(FAITHFULNESS_NAME, {})
    hallucination = rows.get(HALLUCINATION_NAME, {})
    faith_mean = faithfulness.get("mean")
    hall_rate = hallucination_rate(hallucination.get("mean"))
    reasons: list[str] = []

    if faith_mean is None:
        reasons.append("Faithfulness produced no scores")
    elif faith_mean < faithfulness_min:
        reasons.append(
            f"Faithfulness mean {faith_mean:.3f} is below the {faithfulness_min:.1f} gate"
        )

    if hall_rate is None:
        reasons.append("Hallucination produced no scores")
    elif hall_rate > hallucination_rate_max:
        reasons.append(
            f"Hallucination rate {hall_rate:.3f} is above the {hallucination_rate_max:.1f} gate"
        )

    return not reasons, reasons, faith_mean, hall_rate


def render_retrieval(payload: Mapping[str, Any]) -> list[str]:
    """Render the 10-question labeled retrieval lane."""
    summary = payload.get("summary") or {}
    return [
        "### Retriever lane",
        "",
        "10 labeled questions from `src/rag/eval/cases.py`. No LLM judge.",
        "",
        f"- source@k: **{(summary.get('source_hit_at_k') or 0):.0%}**",
        f"- page@k: **{(summary.get('page_hit_at_k') or 0):.0%}**",
        f"- phrase recall: **{(summary.get('phrase_recall') or 0):.0%}**",
        f"- auto relevant: **{(summary.get('auto_relevant') or 0):.0%}**",
        "",
    ]


def render_markdown(
    payload: Mapping[str, Any],
    run_path: Path,
    *,
    faithfulness_min: float,
    hallucination_rate_max: float,
    retrieval: Mapping[str, Any] | None = None,
) -> str:
    """Build a GitHub-flavored markdown summary for the PR comment."""
    rows = _metric_rows(payload)
    passed, reasons, faith_mean, hall_rate = evaluate_gates(
        rows,
        faithfulness_min=faithfulness_min,
        hallucination_rate_max=hallucination_rate_max,
    )
    identifier = payload.get("identifier") or "n/a"
    suite_pass = payload.get("testPassed")
    suite_fail = payload.get("testFailed")
    gate_label = "passed" if passed else "failed"

    lines = [
        COMMENT_MARKER,
        "## RAG eval report",
        "",
        f"- Run: `{run_path.name}` (`{identifier}`)",
        f"- Suite: {suite_pass} passed / {suite_fail} failed (pytest metrics; not the CI gate)",
        f"- Gates: **{gate_label}** — fail if Faithfulness < {faithfulness_min} or Hallucination rate > {hallucination_rate_max}",
        "",
    ]
    if retrieval:
        lines.extend(render_retrieval(retrieval))
        lines.extend(["### Generator lane", ""])
    if faith_mean is not None:
        lines.append(f"- Faithfulness mean: `{faith_mean:.3f}` (higher is better)")
    if hall_rate is not None:
        alignment = 1.0 - hall_rate
        lines.append(
            f"- Hallucination rate: `{hall_rate:.3f}` (contradiction rate; "
            f"DeepEval alignment mean `{alignment:.3f}`)"
        )
    lines.extend(["", "| Metric | Mean | Pass | Fail | Errors |", "| --- | ---: | ---: | ---: | ---: |"])
    for name, row in rows.items():
        mean = f"{row['mean']:.3f}" if row["mean"] is not None else "n/a"
        lines.append(
            f"| {name} | {mean} | {row['pass']} | {row['fail']} | {row['error']} |"
        )
    if reasons:
        lines.extend(["", "### Gate failures", ""])
        lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(
        [
            "",
            "_HallucinationMetric in DeepEval 4.x scores alignment (1.0 = no contradiction). "
            "The CI gate uses `1 - mean` so `Hallucination > 0.1` still means too many contradictions._",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the CI report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/deepeval"),
        help="Directory containing DeepEval test_run_*.json files.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("eval-comment.md"),
        help="Markdown file written for the PR comment step.",
    )
    parser.add_argument(
        "--faithfulness-min",
        type=float,
        default=CI_FAITHFULNESS_MIN,
    )
    parser.add_argument(
        "--hallucination-max",
        type=float,
        default=CI_HALLUCINATION_RATE_MAX,
        help="Maximum allowed contradiction rate (1 - DeepEval Hallucination mean).",
    )
    parser.add_argument(
        "--retrieval-json",
        type=Path,
        help="Optional labeled retrieval report from rag-eval.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the PR summary and exit 1 when quality gates fail."""
    args = parse_args(argv)
    run_path = latest_run(args.results_dir)
    payload = json.loads(run_path.read_text())
    retrieval = None
    if args.retrieval_json and args.retrieval_json.is_file():
        retrieval = json.loads(args.retrieval_json.read_text())
    markdown = render_markdown(
        payload,
        run_path,
        faithfulness_min=args.faithfulness_min,
        hallucination_rate_max=args.hallucination_max,
        retrieval=retrieval,
    )
    args.output_md.write_text(markdown)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    passed, _, _, _ = evaluate_gates(
        _metric_rows(payload),
        faithfulness_min=args.faithfulness_min,
        hallucination_rate_max=args.hallucination_max,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
