from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping windows of at most `max_chars` characters.

    Prefers to break on a whitespace boundary near the window end so words are
    not split. Consecutive chunks overlap by `overlap` characters.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            break_at = text.rfind(" ", start, end)
            if break_at > start:
                end = break_at
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


@dataclass(frozen=True)
class MarkdownChunk:
    text: str
    breadcrumb: str  # "page title > h1 > h2 > h3" path of this chunk's section


def chunk_markdown(
    markdown: str, page_title: str, max_chars: int = 2000, overlap: int = 200
) -> list[MarkdownChunk]:
    """Split markdown into chunks on heading boundaries (ADR-0004).

    Sections longer than max_chars are size-bounded with chunk_text. Headings
    inside fenced code blocks are not section boundaries. Every chunk carries
    its heading-path breadcrumb, the deterministic decoration floor (ADR-0001).
    """
    heading_stack: dict[int, str] = {}
    sections: list[tuple[str, list[str]]] = []  # (breadcrumb, lines)

    def crumb() -> str:
        path = [page_title] + [heading_stack[lvl] for lvl in sorted(heading_stack)]
        return " > ".join(p for p in path if p)

    current: list[str] = []
    in_fence = False
    sections.append((crumb(), current))
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            heading_stack[level] = title
            for deeper in [lvl for lvl in heading_stack if lvl > level]:
                del heading_stack[deeper]
            current = [line]
            sections.append((crumb(), current))
        else:
            current.append(line)

    chunks: list[MarkdownChunk] = []
    for breadcrumb, lines in sections:
        for piece in chunk_text("\n".join(lines), max_chars=max_chars, overlap=overlap):
            chunks.append(MarkdownChunk(text=piece, breadcrumb=breadcrumb))
    return chunks
