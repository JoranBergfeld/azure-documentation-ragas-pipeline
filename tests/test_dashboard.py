from __future__ import annotations

from app.dashboard import stage_chunk_tables
from ragpipe.models import Chunk, PipelineState


def test_stage_chunk_tables_uses_dynamic_stages():
    s = PipelineState(query="q")
    s.set_stage("local", [Chunk(id="1", title="t", url="u", content="hello")])
    s.set_reranked([Chunk(id="1", title="t", url="u", content="hello")])
    tables = stage_chunk_tables(s)
    assert "local" in tables
    assert "reranked" in tables


def test_mode_options_lists_all_registered_modes_in_registry_order():
    from app.dashboard import mode_options
    from ragpipe.retrieval.registry import registered_modes

    assert mode_options() == [m.value for m in registered_modes()]
    assert len(mode_options()) == 9
    assert mode_options()[0] == "contextual"


def test_main_prewarms_ragas_imports_at_load_time():
    """The Run tab builds the RAGAS judge during a Streamlit rerun, which
    corrupts the first ``langchain_openai`` import and raises a pydantic
    ``ValidationError`` for ``RunnablePassthrough``. ``main()`` must prewarm
    those imports at a clean time before any build can be triggered."""
    import inspect

    from app import dashboard

    assert "prewarm_ragas_imports()" in inspect.getsource(dashboard.main)
