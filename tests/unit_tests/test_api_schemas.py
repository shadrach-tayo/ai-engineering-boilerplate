from api.schemas import EsPagination, SearchForm, SearchPageContext
from api.service import es_total_count, format_es_results, format_vector_results


def test_search_form_coerces_invalid_values() -> None:
    form = SearchForm(
        query="  agents  ",
        index_name="unknown",
        top_n=99,
        from_=-4,
        ask_agent="1",
    )
    assert form.query == "agents"
    assert form.index_name == "chunk_256"
    assert form.top_n == 20
    assert form.from_ == 0
    assert form.ask_agent is True


def test_format_vector_results_maps_rerank_and_metadata() -> None:
    rows = format_vector_results(
        {
            "docs": ["chunk text"],
            "original_docs": [{"source": "guide.pdf", "page": 3}],
            "rerank": [{"index": 0, "relevance_score": 0.91}],
        },
        "what is an agent",
    )
    assert len(rows) == 1
    assert rows[0].name == "guide.pdf"
    assert rows[0].page == 3
    assert rows[0].score == 0.91
    assert rows[0].query == "what is an agent"


def test_format_es_results_reads_hit_aliases() -> None:
    rows = format_es_results(
        {
            "results": [
                {
                    "_id": "abc123",
                    "_score": 2.5,
                    "_source": {
                        "content": "hello",
                        "source": "notes.md",
                        "page": 1,
                    },
                }
            ],
            "total": {"value": 9},
        },
        "hello",
    )
    assert rows[0].id == "abc123"
    assert rows[0].score == 2.5
    assert rows[0].source == "notes.md"
    assert es_total_count({"total": {"value": 9}}, fallback=1) == 9


def test_search_page_context_includes_pagination() -> None:
    form = SearchForm(query="q", top_n=5, from_=5)
    context = SearchPageContext.from_search(form, es_total=12)
    page = EsPagination.from_offset(5, 5, 12)
    assert context.es_page == page.es_page == 2
    assert context.es_has_prev is True
    assert context.es_has_next is True
    assert context.es_prev_from == 0
    assert context.es_next_from == 10
