"""DeepEval generator metrics for answerable vs unsupported RAG goldens."""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import SingleTurnParams

DEEPEVAL_FAITHFULNESS_THRESHOLD = 0.7
DEEPEVAL_ANSWER_RELEVANCY_THRESHOLD = 0.7
DEEPEVAL_HALLUCINATION_THRESHOLD = 0.3
DEEPEVAL_REFUSAL_THRESHOLD = 0.7
DEEPEVAL_JUDGE_MODEL = "gpt-4o-mini"


def _faithfulness() -> FaithfulnessMetric:
    return FaithfulnessMetric(
        threshold=DEEPEVAL_FAITHFULNESS_THRESHOLD,
        model=DEEPEVAL_JUDGE_MODEL,
        include_reason=True,
    )


def _hallucination() -> HallucinationMetric:
    return HallucinationMetric(
        threshold=DEEPEVAL_HALLUCINATION_THRESHOLD,
        model=DEEPEVAL_JUDGE_MODEL,
        include_reason=True,
    )


def grounded_generator_metrics() -> list[FaithfulnessMetric | AnswerRelevancyMetric | HallucinationMetric]:
    """Score in-domain answers for grounding, on-topicness, and hallucination."""
    return [
        _faithfulness(),
        AnswerRelevancyMetric(
            threshold=DEEPEVAL_ANSWER_RELEVANCY_THRESHOLD,
            model=DEEPEVAL_JUDGE_MODEL,
            include_reason=True,
        ),
        _hallucination(),
    ]


def unsupported_fallback_metrics() -> list[FaithfulnessMetric | HallucinationMetric | GEval]:
    """Score out-of-domain refusals without Answer Relevancy."""
    return [
        _faithfulness(),
        _hallucination(),
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
            model=DEEPEVAL_JUDGE_MODEL,
        ),
    ]
