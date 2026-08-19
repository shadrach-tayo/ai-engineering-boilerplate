"""Labeled retrieval evaluation cases and scoring helpers."""

from dotenv import load_dotenv

from rag.eval.cases import CASES, EvalCase
from rag.eval.metrics import CaseScore, score_case, summarize

load_dotenv()

__all__ = ["CASES", "CaseScore", "EvalCase", "score_case", "summarize"]
