from __future__ import annotations

import re
from enum import Enum


class QueryClass(str, Enum):
    """How a query should engage the GraphRAG legs.

    LOCAL  -- factoid / entity-lookup queries answered by precise leaf chunks;
              the global community-summary leg only dilutes these (issue #8).
    GLOBAL -- sensemaking / breadth queries ("compare ...", "overview of ...",
              "themes across ...") that GraphRAG's community reports are built for.
    """

    LOCAL = "local"
    GLOBAL = "global"


# Breadth / sensemaking markers. Presence of any one promotes a query to GLOBAL;
# everything else stays LOCAL. The default is deliberately LOCAL: the evaluated
# workload is factoid-heavy and that is exactly where fusing global summaries
# evicts the correct leaf chunk (issue #8). Keep this list conservative -- only
# add markers that unambiguously signal corpus-wide reasoning.
_GLOBAL_MARKERS = (
    "compare",
    "comparison",
    "contrast",
    "versus",
    "vs",
    "difference between",
    "differences between",
    "overview",
    "summarize",
    "summarise",
    "summary",
    "overall",
    "in general",
    "broadly",
    "themes",
    "theme",
    "trends",
    "relationship between",
    "relationships between",
    "relationships",
    "relate",
    "interplay",
    "what are the main",
    "what are the key",
    "what are the common",
    "categories",
    "types of",
    "kinds of",
    "pros and cons",
    "advantages and disadvantages",
    "trade-offs",
    "tradeoffs",
    "high-level",
    "big picture",
    "landscape",
    "ecosystem",
    "across",
)

_GLOBAL_RE = re.compile(
    "|".join(rf"\b{re.escape(marker)}\b" for marker in _GLOBAL_MARKERS),
    re.IGNORECASE,
)


def classify_query(query: str) -> QueryClass:
    """Heuristically classify a query as LOCAL or GLOBAL.

    Deterministic and network-free (a substring/word-boundary match over a fixed
    marker list) so it is cheap, reproducible, and unit-testable. Returns GLOBAL
    when any breadth/sensemaking marker is present, else LOCAL.
    """
    if query and _GLOBAL_RE.search(query):
        return QueryClass.GLOBAL
    return QueryClass.LOCAL
