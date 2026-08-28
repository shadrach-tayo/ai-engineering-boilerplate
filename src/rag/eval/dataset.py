"""Generate and push the single-turn RAG evaluation dataset."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, StylingConfig
from deepeval.synthesizer.types import Evolution
from deepeval.test_case import RetrievedContextData

from rag.eval.cases import CASES
from rag.eval.tests.metrics import (
    DEEPEVAL_JUDGE_MODEL_NAME,
    DEEPSEEK_BASE_URL,
    eval_judge_model,
)
from rag.pipeline import RagConfig, RagPipeline

DATASET_ALIAS = "rag-agent-guides-single-turn-v1"
DATASET_PATH = Path(__file__).parent / "tests" / "goldens.json"
GENERATION_MODEL = DEEPEVAL_JUDGE_MODEL_NAME

RAG_CONFIG = RagConfig(
    index_name="chunk_512",
    strategy="vector",
    top_k=5,
    rerank=False,
    llm_model=GENERATION_MODEL,
    llm_base_url=DEEPSEEK_BASE_URL,
    llm_api_key=os.getenv("DEEPSEEK_API_KEY"),
)


def _normal_goldens(pipeline: RagPipeline) -> list[Golden]:
    """Convert curated cases to Goldens with their live retrieval contexts."""
    goldens: list[Golden] = []
    for case in CASES:
        retrieval = pipeline.retrieve(case.question)
        if not retrieval.docs:
            raise RuntimeError(f"{case.id} returned no retrieval context")
        retrieval_docs: list[str | RetrievedContextData] = list(retrieval.docs)
        goldens.append(
            Golden(
                name=case.id,
                input=case.question,
                expected_output=case.gold_answer,
                context=retrieval.docs,
                retrieval_context=retrieval_docs,
                additional_metadata={
                    "category": "normal",
                    "expected_sources": list(case.expected_sources),
                    "expected_pages": [list(page) for page in case.expected_pages],
                },
                multimodal=False,
            )
        )
    return goldens


def _edge_goldens(seed: list[Golden]) -> list[Golden]:
    """Generate constrained and multi-context variants of curated questions."""
    synthesizer = Synthesizer(
        model=eval_judge_model(),
        evolution_config=EvolutionConfig(
            num_evolutions=2,
            evolutions={
                Evolution.CONSTRAINED: 0.4,
                Evolution.COMPARATIVE: 0.3,
                Evolution.MULTICONTEXT: 0.3,
            },
        ),
        styling_config=StylingConfig(
            scenario=(
                "Developers and support leaders asking nuanced questions about "
                "building, evaluating, and operating AI agents."
            ),
            task=(
                "Create difficult but answerable questions grounded entirely in "
                "the supplied agent-guide context."
            ),
            input_format=(
                "One natural-language question with constraints, comparisons, "
                "or details that may require multiple context passages."
            ),
            expected_output_format=(
                "A concise answer of no more than three sentences supported by context."
            ),
        ),
    )
    goldens = synthesizer.generate_goldens_from_goldens(
        seed,
        max_goldens_per_golden=1,
        include_expected_output=True,
    )
    if len(goldens) != len(seed):
        raise RuntimeError(f"Expected {len(seed)} edge goldens, got {len(goldens)}")
    for index, golden in enumerate(goldens, start=1):
        golden.name = f"edge_{index:02d}"
        golden.additional_metadata = {
            **(golden.additional_metadata or {}),
            "category": "edge",
        }
    return goldens


def _failure_goldens(count: int) -> list[Golden]:
    """Generate unsupported or adversarial requests for safe fallback behavior."""
    synthesizer = Synthesizer(
        model=eval_judge_model(),
        evolution_config=EvolutionConfig(
            num_evolutions=2,
            evolutions={
                Evolution.CONSTRAINED: 0.4,
                Evolution.HYPOTHETICAL: 0.3,
                Evolution.IN_BREADTH: 0.3,
            },
        ),
        styling_config=StylingConfig(
            scenario=(
                "Users send unsupported, misleading, or adversarial requests to a "
                "RAG assistant whose knowledge base only contains AI-agent guides."
            ),
            task=(
                "Create requests that cannot be answered from those guides or that "
                "attempt to make the assistant trust instructions embedded in source "
                "documents. The correct behavior is to avoid unsupported claims and "
                "answer that it does not know."
            ),
            input_format=(
                "One realistic natural-language request containing no answer or metadata."
            ),
            expected_output_format='The exact fallback response: "I don\'t know."',
        ),
    )
    candidates = synthesizer.generate_goldens_from_scratch(num_goldens=count * 3)
    domain_terms = re.compile(
        r"\b(agent|rag|retrieval|knowledge base|kb|prompt|support|customer|llm|"
        r"artificial intelligence|model|tool|workflow|guardrail|policy|privacy|"
        r"gdpr|ccpa|automation|evaluation|eval|api|sandbox|compliance)\b",
        re.IGNORECASE,
    )
    goldens = [
        golden for golden in candidates if not domain_terms.search(golden.input)
    ][:count]
    if len(goldens) != count:
        raise RuntimeError(
            f"Expected {count} out-of-domain failure goldens, got {len(goldens)}"
        )
    for index, golden in enumerate(goldens, start=1):
        golden.name = f"failure_{index:02d}"
        golden.expected_output = "I don't know."
        golden.additional_metadata = {
            **(golden.additional_metadata or {}),
            "category": "failure",
            "expected_behavior": "unsupported_fallback",
        }
    return goldens


def build_dataset() -> EvaluationDataset:
    """Generate the balanced 30-golden single-turn dataset."""
    normal = _normal_goldens(RagPipeline(RAG_CONFIG))
    edge = _edge_goldens(normal)
    failure = _failure_goldens(len(normal))
    return EvaluationDataset(goldens=[*normal, *edge, *failure])


def save_and_push_dataset(alias: str = DATASET_ALIAS) -> Path:
    """Generate, validate, save, reload, and push the dataset."""
    dataset = build_dataset()
    saved_path = Path(
        dataset.save_as(
            file_type="json",
            directory=str(DATASET_PATH.parent),
            file_name=DATASET_PATH.stem,
        )
    )
    reloaded = EvaluationDataset()
    reloaded.add_goldens_from_json_file(file_path=str(saved_path))
    if len(reloaded.goldens) != 30:
        raise RuntimeError(f"Expected 30 saved goldens, got {len(reloaded.goldens)}")
    reloaded.push(alias=alias)
    return saved_path


def main() -> None:
    """Generate and upload the app's single-turn evaluation dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", default=DATASET_ALIAS)
    args = parser.parse_args()
    save_and_push_dataset(alias=args.alias)


if __name__ == "__main__":
    main()
