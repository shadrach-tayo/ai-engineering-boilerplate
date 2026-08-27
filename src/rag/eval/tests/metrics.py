"""DeepEval generator metrics for answerable vs unsupported RAG goldens."""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models import DeepSeekModel
from deepeval.test_case import SingleTurnParams

DEEPEVAL_FAITHFULNESS_THRESHOLD = 0.7
DEEPEVAL_ANSWER_RELEVANCY_THRESHOLD = 0.7
DEEPEVAL_HALLUCINATION_THRESHOLD = 0.3
DEEPEVAL_REFUSAL_THRESHOLD = 0.7
DEEPEVAL_ANSWER_CORRECTNESS_THRESHOLD = 0.7
DEEPEVAL_CONTEXTUAL_THRESHOLD = 0.5
DEEPEVAL_JUDGE_MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# CI quality gates (aggregate means). HallucinationMetric in DeepEval 4.x is
# alignment (higher is better); the gate uses contradiction_rate = 1 - mean.
CI_FAITHFULNESS_MIN = 0.8
CI_HALLUCINATION_RATE_MAX = 0.1


def eval_judge_model() -> DeepSeekModel:
    """Return the DeepSeek judge used by native DeepEval metrics and synthesis."""
    return DeepSeekModel(
        model=DEEPEVAL_JUDGE_MODEL_NAME,
        temperature=0.0,
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
    )


def _faithfulness() -> FaithfulnessMetric:
    """Return a faithfulness metric judged by DeepSeek."""
    return FaithfulnessMetric(
        threshold=DEEPEVAL_FAITHFULNESS_THRESHOLD,
        model=eval_judge_model(),
        include_reason=True,
    )


def _hallucination() -> HallucinationMetric:
    """Return a hallucination metric judged by DeepSeek."""
    return HallucinationMetric(
        threshold=DEEPEVAL_HALLUCINATION_THRESHOLD,
        model=eval_judge_model(),
        include_reason=True,
    )


def _answer_correctness() -> GEval:
    """Score whether the generated answer matches the labeled expected output."""
    return GEval(
        name="Answer Correctness",
        criteria=(
            "Determine whether the actual output is factually correct relative "
            "to the expected output. Credit equivalent wording that covers the "
            "same key claims. Penalize missing claims, contradictions, or "
            "unsupported extra facts."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=DEEPEVAL_ANSWER_CORRECTNESS_THRESHOLD,
        model=eval_judge_model(),
    )


def grounded_generator_metrics() -> list:
    """Score in-domain answers for grounding, retrieval coverage, and correctness."""
    judge = eval_judge_model()
    return [
        _faithfulness(),
        AnswerRelevancyMetric(
            threshold=DEEPEVAL_ANSWER_RELEVANCY_THRESHOLD,
            model=judge,
            include_reason=True,
        ),
        _hallucination(),
        ContextualPrecisionMetric(
            threshold=DEEPEVAL_CONTEXTUAL_THRESHOLD,
            model=judge,
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=DEEPEVAL_CONTEXTUAL_THRESHOLD,
            model=judge,
            include_reason=True,
        ),
        _answer_correctness(),
    ]


def unsupported_fallback_metrics() -> list[GEval]:
    """Score out-of-domain refusals without retrieval-grounding metrics."""
    return [
        GEval(
            name="Unsupported Refusal",
            criteria=(
                "Determine whether the actual output correctly refuses the "
                "unsupported or out-of-scope input instead of inventing an answer. "
                "A concise 'I don't know' (or equivalent) that matches the expected "
                "output should pass. Do not penalize a refusal for omitting facts "
                "the knowledge base does not contain."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=DEEPEVAL_REFUSAL_THRESHOLD,
            model=eval_judge_model(),
        ),
    ]
