"""Deterministic, LLM-free retrieval metrics (ADR-0002).

Judged by URL match: factoid items carry one or more ground-truth source URLs
and every indexed chunk carries its page URL. Multiple gold URLs use any-match
semantics. Exact, reproducible, and free, unlike the LLM-judged context metrics
(which remain as a complement).
"""
from __future__ import annotations

from collections.abc import Iterable
import re

# Strip a learn.microsoft.com-style locale segment directly after the host
# (e.g. /en-us/). Only the first path segment is considered, so service names
# that happen to look locale-ish deeper in the path are untouched.
_LOCALE_RE = re.compile(r"^(https?://[^/]+)/[a-z]{2}-[a-z]{2}(/|$)")


def normalize_url(url: str) -> str:
    """Canonical form for URL equality: lowercase, no locale segment, no
    query/fragment, no trailing slash."""
    url = url.strip().lower()
    url = url.split("#", 1)[0].split("?", 1)[0]
    url = _LOCALE_RE.sub(r"\1\2", url, count=1)
    return url.rstrip("/")


def _gold_set(gold: str | Iterable[str]) -> set[str]:
    return {gold} if isinstance(gold, str) else set(gold)


def hit_rate(urls: list[str], gold: str | Iterable[str]) -> float:
    """1.0 if any gold URL appears anywhere in the list, else 0.0.

    URLs are compared verbatim — pass both sides through normalize_url first
    (stage_retrieval_metrics does this for you).
    """
    golds = _gold_set(gold)
    return 1.0 if any(url in golds for url in urls) else 0.0


def mrr(urls: list[str], gold: str | Iterable[str]) -> float:
    """Reciprocal rank (1-based) of the first matching gold URL; 0.0 if absent.

    URLs are compared verbatim — pass both sides through normalize_url first
    (stage_retrieval_metrics does this for you).
    """
    golds = _gold_set(gold)
    for i, url in enumerate(urls):
        if url in golds:
            return 1.0 / (i + 1)
    return 0.0


def stage_retrieval_metrics(
    stage_urls: dict[str, list[str]], gold: str | Iterable[str]
) -> dict[str, float]:
    """hit_rate/mrr per stage, keyed 'hit_rate@<stage>' / 'mrr@<stage>'."""
    golds = {normalized for url in _gold_set(gold) if (normalized := normalize_url(url))}
    if not golds:
        return {}
    out: dict[str, float] = {}
    for stage, urls in stage_urls.items():
        normalized = [normalize_url(u) for u in urls]
        out[f"hit_rate@{stage}"] = hit_rate(normalized, golds)
        out[f"mrr@{stage}"] = mrr(normalized, golds)
    return out
