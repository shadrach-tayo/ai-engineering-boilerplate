"""Patch LLM clients for Braintrust before DeepEval imports them."""

from collections.abc import Iterator

import pytest

from rag.tracing import flush_braintrust, setup_braintrust

setup_braintrust()


@pytest.fixture(scope="session", autouse=True)
def _flush_braintrust_traces() -> Iterator[None]:
    """Upload eval spans so they appear on the Braintrust dashboard."""
    yield
    flush_braintrust()
