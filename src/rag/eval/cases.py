"""Gold question-answer pairs for manual retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """One labeled retrieval example drawn from the indexed agent guides."""

    id: str
    question: str
    gold_answer: str
    expected_sources: tuple[str, ...]
    expected_pages: tuple[tuple[str, int], ...]
    must_contain: tuple[str, ...]


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="q01_agent_definition",
        question="What is an agent?",
        gold_answer=(
            "An agent independently accomplishes tasks on a user's behalf using an LLM "
            "to manage workflow execution and tools to take action. Simple chatbots or "
            "single-turn LLM apps that do not control workflow execution are not agents."
        ),
        expected_sources=("openai-guide.pdf",),
        expected_pages=(("openai-guide.pdf", 3),),
        must_contain=("independently", "workflow", "not agents"),
    ),
    EvalCase(
        id="q02_guardrails",
        question="What are the guardrails used in building AI agents?",
        gold_answer=(
            "Guardrails are a layered defense: relevance and safety classifiers, PII "
            "filters, moderation, tool risk ratings, plus rules-based checks such as regex. "
            "They reduce prompt leaks, off-topic answers, and unsafe tool use."
        ),
        expected_sources=("openai-guide.pdf",),
        expected_pages=(
            ("openai-guide.pdf", 23),
            ("openai-guide.pdf", 24),
            ("openai-guide.pdf", 25),
            ("openai-guide.pdf", 26),
        ),
        must_contain=("relevance classifier", "PII", "moderation"),
    ),
    EvalCase(
        id="q03_architecture_patterns",
        question="What are the common architecture patterns for building AI agents?",
        gold_answer=(
            "Multi-agent systems use centralized (supervisor/orchestrator/router) "
            "hierarchies or decentralized peer-to-peer collaboration (swarm or federated). "
            "Agentic workflows orchestrate multi-step execution."
        ),
        expected_sources=("claude-guide.pdf",),
        expected_pages=(("claude-guide.pdf", 9), ("claude-guide.pdf", 10)),
        must_contain=("centralized", "decentralized", "swarm"),
    ),
    EvalCase(
        id="q04_use_cases",
        question="What are the common use cases and applications for AI agents?",
        gold_answer=(
            "Documented deployments include coding assistants (Augment Code), "
            "natural-language data/observability analysis (Grafana), and customer "
            "support and operations agents."
        ),
        expected_sources=("claude-guide.pdf",),
        expected_pages=(("claude-guide.pdf", 5), ("claude-guide.pdf", 6)),
        must_contain=("coding", "Grafana", "customer support"),
    ),
    EvalCase(
        id="q05_evaluations",
        question="What are evaluations and how do we build evaluations for agentic systems?",
        gold_answer=(
            "Evals are offline tests, often LLM-as-a-judge, built from real interactions "
            "with multi-annotator majority-vote labels. They drive prompt iteration and "
            "are validated against online metrics such as A/B tests."
        ),
        expected_sources=("support-agent.pdf",),
        expected_pages=(("support-agent.pdf", 5), ("support-agent.pdf", 2)),
        must_contain=("LLM-as-a-judge", "majority vote", "eval"),
    ),
    EvalCase(
        id="q06_when_to_build",
        question="When should you build an agent instead of conventional automation?",
        gold_answer=(
            "Agents fit workflows where deterministic rules fall short, such as payment "
            "fraud analysis: a rules engine uses a checklist, while an LLM agent acts "
            "like an investigator over ambiguous context."
        ),
        expected_sources=("openai-guide.pdf",),
        expected_pages=(("openai-guide.pdf", 4),),
        must_contain=("fraud", "rules", "investigator"),
    ),
    EvalCase(
        id="q07_core_components",
        question="What are the three core components of an agent?",
        gold_answer=(
            "An agent is built from a model (the LLM), tools (external functions or APIs), "
            "and instructions (guidelines and guardrails for behavior)."
        ),
        expected_sources=("openai-guide.pdf",),
        expected_pages=(("openai-guide.pdf", 6),),
        must_contain=("Model", "Tools", "Instructions"),
    ),
    EvalCase(
        id="q08_model_selection",
        question="How should you select models when building an agent?",
        gold_answer=(
            "Prototype with the most capable model to set a baseline, then swap in "
            "smaller models where accuracy still holds to cut cost and latency. Not every "
            "task needs the smartest model."
        ),
        expected_sources=("openai-guide.pdf",),
        expected_pages=(("openai-guide.pdf", 7),),
        must_contain=("baseline", "smaller models", "cost"),
    ),
    EvalCase(
        id="q09_coinbase_agents",
        question="How did Coinbase use Claude-powered agents for customer support?",
        gold_answer=(
            "Coinbase built Claude-powered customer support agents that handle thousands "
            "of messages per hour at 99.99% availability, with dozens of internal AI apps "
            "around that platform."
        ),
        expected_sources=("claude-guide.pdf",),
        expected_pages=(("claude-guide.pdf", 3),),
        must_contain=("Coinbase", "99.99%", "thousands of messages"),
    ),
    EvalCase(
        id="q10_cs_agent_challenges",
        question="What unique challenges do customer support AI agents face compared with general-purpose agents?",
        gold_answer=(
            "CS agents need a higher quality bar because failures erode trust, must handle "
            "sensitive customer data, use a small specialized toolset, and hand off to humans "
            "with full context when confidence is low."
        ),
        expected_sources=("support-agent.pdf",),
        expected_pages=(("support-agent.pdf", 1),),
        must_contain=("quality bar", "sensitive data", "handoff"),
    ),
)
