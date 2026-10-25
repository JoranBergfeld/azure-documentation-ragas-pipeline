from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_reranker(monkeypatch):
    """Tests never probe a real reranker: RERANKER_URL starts unset and the
    cached /health answers start empty. Tests that want one set it themselves."""
    from ragpipe.retrieval.cross_encoder import health

    for name in ("RERANKER_URL", "RERANKER_API_KEY", "RERANKER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    health.clear()
    yield
    health.clear()
