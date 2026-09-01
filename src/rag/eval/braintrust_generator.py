"""Upload DeepEval generator runs to Braintrust as named experiments.

Replays scored ``test_run_*.json`` files so Compare shows the real pass rates
from ``rag-latest-dataset-main`` (k=5, no rerank) versus
``rag-topk-10-rerank-5-main`` (k=10, rerank to 5). Scores are 0/1 pass so
experiment means are pass rates.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from braintrust import init
from braintrust.logger import ExperimentSummary
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.braintrust", override=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEEPEVAL_DIR = REPO_ROOT / "data" / "deepeval"
DEFAULT_PROJECT = "langgraph-mcp"
GEVAL_SUFFIX = " [GEval]"

BASELINE = (
    "rag-latest-dataset-main",
    DEEPEVAL_DIR / "test_run_20260827_132733.json",
)
CANDIDATE = (
    "rag-topk-10-rerank-5-main",
    DEEPEVAL_DIR / "test_run_20260827_142224.json",
)


def normalize_metric_name(name: str) -> str:
    """Drop DeepEval's `` [GEval]`` suffix so Compare columns align."""
    return name.removesuffix(GEVAL_SUFFIX)


def cases_from_run(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turn DeepEval test cases into Braintrust log rows."""
    rows: list[dict[str, Any]] = []
    for case in payload.get("testCases") or []:
        scores: dict[str, float] = {}
        details: dict[str, Any] = {}
        for metric in case.get("metricsData") or []:
            raw_name = metric.get("name")
            if not raw_name:
                continue
            name = normalize_metric_name(str(raw_name))
            scores[name] = 1.0 if metric.get("success") else 0.0
            details[name] = {
                "score": metric.get("score"),
                "threshold": metric.get("threshold"),
                "reason": metric.get("reason"),
            }
        rows.append(
            {
                "input": case.get("input"),
                "output": case.get("actualOutput"),
                "expected": case.get("expectedOutput"),
                "scores": scores,
                "metadata": {
                    "case_name": case.get("name"),
                    "deepeval_success": case.get("success"),
                    "metric_details": details,
                },
            }
        )
    return rows


def load_run(path: Path) -> dict[str, Any]:
    """Read a DeepEval test-run JSON file."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _project_settings() -> tuple[str, str | None]:
    project = os.getenv("BRAINTRUST_PROJECT") or DEFAULT_PROJECT
    project_id = os.getenv("BRAINTRUST_PROJECT_ID") or None
    return project, project_id


def upload_run(
    path: Path,
    experiment_name: str,
    *,
    base_experiment: str | None = None,
) -> ExperimentSummary:
    """Log every case from a DeepEval run as one named Braintrust experiment."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing DeepEval run {path}. It is gitignored under data/deepeval/."
        )
    payload = load_run(path)
    rows = cases_from_run(payload)
    if not rows:
        raise ValueError(f"{path} has no testCases")
    project, project_id = _project_settings()
    experiment = init(
        project=project if project_id is None else None,
        project_id=project_id,
        experiment=experiment_name,
        update=True,
        base_experiment=base_experiment,
        metadata={
            "source": path.name,
            "deepeval_identifier": payload.get("identifier"),
            "deepeval_passed": payload.get("testPassed"),
            "deepeval_failed": payload.get("testFailed"),
            "score_kind": "pass_rate",
        },
    )
    for row in rows:
        experiment.log(**row)
    summary = experiment.summarize()
    experiment.flush()
    return summary


def upload_comparison(
    *,
    baseline: tuple[str, Path] = BASELINE,
    candidate: tuple[str, Path] = CANDIDATE,
) -> list[ExperimentSummary]:
    """Upload baseline then candidate with the baseline attached for Compare."""
    summaries = [upload_run(baseline[1], baseline[0])]
    summaries.append(
        upload_run(candidate[1], candidate[0], base_experiment=baseline[0])
    )
    return summaries


def latest_run(results_dir: Path) -> Path:
    """Return the newest DeepEval test_run JSON under ``results_dir``."""
    runs = sorted(results_dir.glob("test_run_*.json"))
    if not runs:
        raise FileNotFoundError(f"no test_run_*.json files in {results_dir}")
    return runs[-1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the DeepEval → Braintrust upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-latest",
        action="store_true",
        help="Upload the newest test_run_*.json using its DeepEval identifier.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEEPEVAL_DIR,
        help="Directory containing DeepEval test_run_*.json files.",
    )
    parser.add_argument(
        "--base-experiment",
        default=None,
        help="Named Braintrust experiment to attach as the Compare baseline.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE[1],
        help="DeepEval JSON for rag-latest-dataset-main.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=CANDIDATE[1],
        help="DeepEval JSON for rag-topk-10-rerank-5-main.",
    )
    return parser.parse_args(argv)


def _print_summary(summary: ExperimentSummary) -> None:
    url = getattr(summary, "experiment_url", None)
    name = getattr(summary, "experiment_name", None) or "experiment"
    compared = getattr(summary, "comparison_experiment_name", None)
    if url:
        print(f"{name}: {url}")  # noqa: T201
    else:
        print(name)  # noqa: T201
    if compared:
        print(f"Compare against {compared} from that experiment page.")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> int:
    """Upload the two named DeepEval runs to Braintrust."""
    args = parse_args(argv)
    if not os.getenv("BRAINTRUST_API_KEY"):
        print(  # noqa: T201
            "BRAINTRUST_API_KEY is not set. Put it in .env.braintrust or the environment."
        )
        return 1
    if args.from_latest:
        path = latest_run(args.results_dir)
        payload = load_run(path)
        name = str(payload.get("identifier") or path.stem)
        summaries = [
            upload_run(path, name, base_experiment=args.base_experiment)
        ]
    else:
        summaries = upload_comparison(
            baseline=(BASELINE[0], args.baseline),
            candidate=(CANDIDATE[0], args.candidate),
        )
    for summary in summaries:
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
