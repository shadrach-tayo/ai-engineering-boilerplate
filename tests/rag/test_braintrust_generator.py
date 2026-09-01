"""Tests for DeepEval → Braintrust generator-run replay."""

from rag.eval.braintrust_generator import (
    cases_from_run,
    normalize_metric_name,
    parse_args,
)


def test_normalize_metric_name_strips_geval_suffix() -> None:
    assert normalize_metric_name("Answer Correctness [GEval]") == "Answer Correctness"
    assert normalize_metric_name("Faithfulness") == "Faithfulness"


def test_cases_from_run_logs_pass_fail_not_raw_scores() -> None:
    payload = {
        "testCases": [
            {
                "name": "q01_agent_definition",
                "input": "What is an agent?",
                "actualOutput": "An agent independently accomplishes tasks.",
                "expectedOutput": "An agent independently accomplishes tasks.",
                "success": False,
                "metricsData": [
                    {
                        "name": "Answer Correctness [GEval]",
                        "score": 0.6,
                        "success": False,
                        "threshold": 0.7,
                        "reason": "Missing claims.",
                    },
                    {
                        "name": "Faithfulness",
                        "score": 1.0,
                        "success": True,
                        "threshold": 0.7,
                        "reason": "Grounded.",
                    },
                ],
            }
        ]
    }
    rows = cases_from_run(payload)
    assert len(rows) == 1
    assert rows[0]["input"] == "What is an agent?"
    assert rows[0]["scores"]["Answer Correctness"] == 0.0
    assert rows[0]["scores"]["Faithfulness"] == 1.0
    assert rows[0]["metadata"]["metric_details"]["Answer Correctness"]["score"] == 0.6


def test_parse_args_from_latest() -> None:
    args = parse_args(
        ["--from-latest", "--results-dir", "data/deepeval", "--base-experiment", "rag-latest-dataset-main"]
    )
    assert args.from_latest is True
    assert args.base_experiment == "rag-latest-dataset-main"
