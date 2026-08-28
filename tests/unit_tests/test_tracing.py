from rag import tracing


def test_setup_braintrust_is_idempotent() -> None:
    tracing.setup_braintrust()
    tracing.setup_braintrust()


def test_trace_span_is_a_context_manager() -> None:
    with tracing.trace_span("unit.noop"):
        pass
