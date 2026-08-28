"""Tests for named Braintrust retrieval experiments."""

from __future__ import annotations

from rag.eval.braintrust_eval import (
    BASELINE,
    CANDIDATE,
    SCORE_TOP_K,
    auto_relevant,
    case_from_expected,
    labeled_dataset,
    make_task,
    page_at_k,
    parse_args,
    phrase_recall,
    source_at_k,
)
from rag.eval.cases import CASES
from rag.pipeline import RetrievalResult


def _hit_output(case_index: int = 0) -> dict[str, object]:
    case = CASES[case_index]
    source, page = case.expected_pages[0]
    return {
        "docs": [" ".join(case.must_contain)],
        "metadata": [{"source": source, "page": page}],
        "strategy": "vector",
        "n_docs": 1,
        "reranked": False,
    }


def test_labeled_dataset_matches_cases() -> None:
    rows = labeled_dataset()
    assert len(rows) == 10
    assert [row.input for row in rows] == [case.question for case in CASES]
    assert [row.metadata["case_id"] for row in rows if row.metadata] == [
        case.id for case in CASES
    ]


def test_case_from_expected_roundtrip() -> None:
    expected = labeled_dataset()[0].expected
    assert expected is not None
    restored = case_from_expected(expected)
    original = CASES[0]
    assert restored.id == original.id
    assert restored.expected_sources == original.expected_sources
    assert restored.expected_pages == original.expected_pages
    assert restored.must_contain == original.must_contain


def test_scorers_pass_on_gold_hit() -> None:
    output = _hit_output()
    expected = labeled_dataset()[0].expected
    question = CASES[0].question
    assert expected is not None
    assert source_at_k(question, output, expected) == 1.0
    assert page_at_k(question, output, expected) == 1.0
    assert phrase_recall(question, output, expected) == 1.0
    assert auto_relevant(question, output, expected) == 1.0


def test_scorers_fail_on_wrong_source() -> None:
    output = {
        "docs": ["unrelated table of contents"],
        "metadata": [{"source": "other.pdf", "page": 0}],
        "strategy": "vector",
    }
    expected = labeled_dataset()[0].expected
    question = CASES[0].question
    assert expected is not None
    assert source_at_k(question, output, expected) == 0.0
    assert page_at_k(question, output, expected) == 0.0
    assert phrase_recall(question, output, expected) == 0.0
    assert auto_relevant(question, output, expected) == 0.0


def test_pipeline_versions_are_the_documented_pair() -> None:
    baseline = BASELINE.rag_config()
    candidate = CANDIDATE.rag_config()
    assert BASELINE.experiment == "chunk512-vector-k5-norerank"
    assert CANDIDATE.experiment == "chunk512-vector-k10-rerank5"
    assert baseline.index_name == candidate.index_name == "chunk_512"
    assert baseline.strategy == candidate.strategy == "vector"
    assert baseline.top_k == 5
    assert baseline.rerank is False
    assert candidate.top_k == 10
    assert candidate.rerank is True
    assert candidate.rerank_top_n == SCORE_TOP_K
    assert BASELINE.score_top_k == CANDIDATE.score_top_k == SCORE_TOP_K


def test_make_task_serializes_retrieval() -> None:
    case = CASES[0]
    source, page = case.expected_pages[0]

    def retrieve(_pipeline: object, question: str) -> RetrievalResult:
        assert question == case.question
        return RetrievalResult(
            docs=[" ".join(case.must_contain)],
            metadata=[{"source": source, "page": page}],
            strategy="vector",
        )

    task = make_task(BASELINE, retrieve=retrieve)
    payload = task(case.question)
    assert payload["metadata"] == [{"source": source, "page": page}]
    assert payload["reranked"] is False
    expected = labeled_dataset()[0].expected
    assert expected is not None
    assert source_at_k(case.question, payload, expected) == 1.0


def test_parse_args_candidate_only() -> None:
    args = parse_args(["--candidate-only", "--base-experiment", "prior-run", "--no-pause"])
    assert args.candidate_only is True
    assert args.baseline_only is False
    assert args.base_experiment == "prior-run"
    assert args.no_pause is True
