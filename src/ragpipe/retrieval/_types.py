from __future__ import annotations

from typing import Any, Protocol


class Searchable(Protocol):
    """Minimal interface of azure.search.documents.SearchClient used by retrievers."""

    def search(self, search_text: str | None = None, **kwargs: Any): ...
