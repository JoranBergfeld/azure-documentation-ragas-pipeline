"""Deterministic, LLM-free retrieval metrics (ADR-0002, ADR-0019).

Judged by URL match: every test item carries one or more ground-truth source URLs
and every indexed chunk carries its page URL. Exact, reproducible, and free, unlike
the LLM-judged context metrics (which remain as a complement).

The gold label is a set of one or more URLs (ADR-0019): single-hop factoid items
carry one URL; multi-hop and global/sensemaking items carry several. ``hit_rate`` is
recall over that set -- for a single gold URL it is the original binary 1.0/0.0, so
the existing factoid items and their committed scores are unchanged.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

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


def gold_set(ground_truth: str | Iterable[str]) -> set[str]:
    """Normalized set of gold URLs from a single URL or a list of them.

    Accepts the two shapes ``TestItem.ground_truth_context`` can take (ADR-0019):
    a ``str`` (single-hop factoid) or an iterable of ``str`` (multi-hop / global).
    Empty/blank entries are dropped, so a gold-less item yields an empty set and
    contributes to RAGAS metrics only (ADR-0016).
    """
    raw = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    return {normalize_url(u) for u in raw if u and u.strip()}


def hit_rate(urls: list[str], ground_truth: str | Iterable[str]) -> float:
    """Recall over the gold set: fraction of gold URLs present in ``urls``.

    With one gold URL this is the original binary 1.0/0.0 (any chunk matches). With
    several (multi-hop / global), partial credit reflects how much of the required
    page set was retrieved -- the headroom single-gold hit-rate lacks (ADR-0019).

    URLs are compared verbatim -- pass both sides through ``normalize_url`` first
    (``stage_retrieval_metrics`` does this for you).
    """
    gold = ground_truth if isinstance(ground_truth, set) else gold_set(ground_truth)
    if not gold:
        return 0.0
    found = sum(1 for g in gold if g in urls)
    return found / len(gold)


def mrr(urls: list[str], ground_truth: str | Iterable[str]) -> float:
    """Reciprocal rank (1-based) of the first chunk matching ANY gold URL; 0.0 if absent.

    URLs are compared verbatim -- pass both sides through ``normalize_url`` first
    (``stage_retrieval_metrics`` does this for you).
    """
    gold = ground_truth if isinstance(ground_truth, set) else gold_set(ground_truth)
    if not gold:
        return 0.0
    for i, url in enumerate(urls):
        if url in gold:
            return 1.0 / (i + 1)
    return 0.0


def stage_retrieval_metrics(
    stage_urls: dict[str, list[str]], ground_truth: str | Iterable[str]
) -> dict[str, float]:
    """hit_rate/mrr per stage, keyed 'hit_rate@<stage>' / 'mrr@<stage>'.

    ``ground_truth`` is one gold URL or a list of them (ADR-0019). A gold-less item
    yields no deterministic keys at all -- it is scored by RAGAS only (ADR-0016).
    """
    gold = gold_set(ground_truth)
    if not gold:
        return {}
    out: dict[str, float] = {}
    for stage, urls in stage_urls.items():
        normalized = [normalize_url(u) for u in urls]
        out[f"hit_rate@{stage}"] = hit_rate(normalized, gold)
        out[f"mrr@{stage}"] = mrr(normalized, gold)
    return out