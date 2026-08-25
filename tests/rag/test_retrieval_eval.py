"""Tests for the 10-question retrieval eval set and scoring."""

from __future__ import annotations

import pytest

from rag.eval.cases import CASES
from rag.eval.metrics import score_case, summarize
from rag.pipeline import RetrievalResult

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_eval_set_has_ten_unique_cases() -> None:
    assert len(CASES) == 10
    ids = [case.id for case in CASES]
    assert len(set(ids)) == 10
    for case in CASES:
        assert case.question.strip()
        assert case.gold_answer.strip()
        assert case.expected_sources
        assert case.expected_pages
        assert case.must_contain


def test_score_case_counts_source_page_and_phrases() -> None:
    case = CASES[0]
    source, page = case.expected_pages[0]
    result = RetrievalResult(
        docs=[
            "Agents independently accomplish tasks and control the workflow. "
            "Chatbots that do not manage execution are not agents."
        ],
        metadata=[{"source": source, "page": page}],
        strategy="vector",
    )
    score = score_case(case, result, top_k=5)
    assert score.source_hit is True
    assert score.page_hit is True
    assert score.phrase_recall == 1.0
    assert score.auto_relevant is True
    assert score.phrases_found == "3/3"


def test_score_case_misses_wrong_source() -> None:
    case = CASES[0]
    result = RetrievalResult(
        docs=["unrelated table of contents"],
        metadata=[{"source": "other.pdf", "page": 0}],
        strategy="vector",
    )
    score = score_case(case, result, top_k=5)
    assert score.source_hit is False
    assert score.page_hit is False
    assert score.phrase_recall == 0.0
    assert score.auto_relevant is False


def test_summarize_averages_hits() -> None:
    case = CASES[0]
    hit = score_case(
        case,
        RetrievalResult(
            docs=["independently workflow not agents"],
            metadata=[{"source": case.expected_pages[0][0], "page": case.expected_pages[0][1]}],
        ),
        top_k=1,
    )
    miss = score_case(
        case,
        RetrievalResult(docs=["nope"], metadata=[{"source": "x.pdf", "page": 99}]),
        top_k=1,
    )
    summary = summarize([hit, miss])
    assert summary["n"] == 2
    assert summary["source_hit_at_k"] == 0.5
    assert summary["page_hit_at_k"] == 0.5


@pytest.mark.integration
def test_live_retrieval_accuracy_scorecard() -> None:
    """Retrieve the 10 labeled questions and print a scorecard for manual review."""
    from rag.eval.run import format_scorecard, run_eval

    try:
        scores, summary = run_eval(
            index_name="chunk_256",
            strategy="vector",
            top_k=5,
            rerank=True,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"retrieval backends unavailable: {exc}")

    report = format_scorecard(scores, summary, top_k=5)
    assert "Retrieval eval" in report
    assert summary["n"] == 10
    assert summary["source_hit_at_k"] > 0, (
        "Expected at least one gold-source hit. Re-ingest chunk_256 or review the scorecard."
    )
