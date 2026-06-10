"""HTML → markdown extraction that preserves document structure (ADR-0004).

MS Learn pages carry their content in a <main> element; code fences, tables,
and the heading hierarchy must survive into chunking (the old whitespace-
flattening path destroyed exactly the content that answers most questions).
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify as _markdownify

_BLANK_RUN_RE = re.compile(r"\n{3,}")


def html_to_markdown(html: str) -> tuple[str, bool]:
    """Convert page HTML to markdown.

    Returns (markdown, used_main): used_main is False when no <main>/<article>
    container was found and the whole (noise-stripped) page was converted —
    callers count these fallbacks so a drift in MS Learn's layout is visible
    in ingest output (spec §8).
    """
    if not html.strip():
        return "", False
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    container = soup.find("main") or soup.find("article")
    used_main = container is not None
    target = container if used_main else soup
    md = _markdownify(str(target), heading_style="ATX")
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return _BLANK_RUN_RE.sub("\n\n", md).strip(), used_main
