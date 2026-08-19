"""Retrieval accuracy helpers for the labeled eval set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.eval.cases import EvalCase
from rag.pipeline import RetrievalResult


@dataclass
class CaseScore:
    """Automatic retrieval checks plus fields for a human reviewer."""

    case_id: str
    question: str
    gold_answer: str
    source_hit: bool
    page_hit: bool
    phrase_hits: list[str]
    phrase_misses: list[str]
    phrase_recall: float
    auto_relevant: bool
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    human_relevant: bool | None = None
    human_note: str = ""

    @property
    def phrases_found(self) -> str:  # noqa: D102
        total = len(self.phrase_hits) + len(self.phrase_misses)
        return f"{len(self.phrase_hits)}/{total}"


def score_case(case: EvalCase, result: RetrievalResult, *, top_k: int) -> CaseScore:
    """Score one retrieval against gold sources, pages, and required phrases."""
    metas = list(result.metadata[:top_k])
    docs = list(result.docs[:top_k])
    sources = {meta.get("source") for meta in metas}
    pages = {(meta.get("source"), meta.get("page")) for meta in metas}
    source_hit = any(source in sources for source in case.expected_sources)
    page_hit = any(page in pages for page in case.expected_pages)
    blob = "\n".join(docs).lower()
    phrase_hits = [phrase for phrase in case.must_contain if phrase.lower() in blob]
    phrase_misses = [
        phrase for phrase in case.must_contain if phrase.lower() not in blob
    ]
    denom = len(case.must_contain) or 1
    phrase_recall = len(phrase_hits) / denom
    auto_relevant = page_hit or (source_hit and phrase_recall >= 0.5)
    retrieved = []
    for i, content in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        retrieved.append(
            {
                "rank": i + 1,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "snippet": " ".join(content.split())[:280],
            }
        )
    return CaseScore(
        case_id=case.id,
        question=case.question,
        gold_answer=case.gold_answer,
        source_hit=source_hit,
        page_hit=page_hit,
        phrase_hits=phrase_hits,
        phrase_misses=phrase_misses,
        phrase_recall=phrase_recall,
        auto_relevant=auto_relevant,
        retrieved=retrieved,
    )


def summarize(scores: list[CaseScore]) -> dict[str, float]:
    """Aggregate hit rates across the labeled set."""
    n = len(scores) or 1
    return {
        "n": float(len(scores)),
        "source_hit_at_k": sum(score.source_hit for score in scores) / n,
        "page_hit_at_k": sum(score.page_hit for score in scores) / n,
        "phrase_recall": sum(score.phrase_recall for score in scores) / n,
        "auto_relevant": sum(score.auto_relevant for score in scores) / n,
    }
