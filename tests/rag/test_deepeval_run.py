"""Tests for stable DeepEval run identifiers."""

from rag.eval.deepeval_run import build_identifier


def test_identifier_uses_explicit_override() -> None:
    assert (
        build_identifier(
            "ignored",
            environ={"DEEPEVAL_IDENTIFIER": "checkout-agent-v2"},
            branch="ignored",
        )
        == "checkout-agent-v2"
    )


def test_identifier_encodes_pull_request() -> None:
    assert (
        build_identifier(
            "generator prompt v2",
            environ={"GITHUB_REF": "refs/pull/42/merge"},
        )
        == "rag-generator-prompt-v2-pr-42"
    )


def test_identifier_encodes_experiment_and_branch() -> None:
    assert (
        build_identifier(
            "chunk 512 vector",
            environ={},
            branch="feature/grounded-answers",
        )
        == "rag-chunk-512-vector-feature-grounded-answers"
    )
