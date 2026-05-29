"""Build a broad Microsoft Learn (Azure) corpus URL list by crawling the sitemaps.

The sitemap index (path declared in learn.microsoft.com/robots.txt) lists per-product,
per-locale shards; the English Azure ones are named azure*_en-us_N.xml.

`select_urls` is pure and unit-tested; `crawl_azure_sitemaps` is the live wrapper.
"""
from __future__ import annotations

import collections
import gzip
import re
import urllib.request

SITEMAP_INDEX = "https://learn.microsoft.com/_sitemaps/sitemapindex.xml"
# The index lists per-product, per-locale shards, e.g.
#   .../_sitemaps/azure_en-us_3.xml, .../azure-stack_en-us_1.xml
# Match the English Azure shards (azure / azure-stack / azure-sphere / ...).
_SHARD_RE = re.compile(
    r"https://learn\.microsoft\.com/_sitemaps/azure[a-z-]*_en-us_\d+\.xml"
)
_UA = {"User-Agent": "Mozilla/5.0 ragpipe-corpus-builder"}
_SERVICE_RE = re.compile(r"^https://learn\.microsoft\.com/en-us/azure/([^/]+)/")
_LOC_RE = re.compile(r"<loc>(.*?)</loc>", re.S)
_NOISE = ("/media/", "/includes/")

# Lower score = preferred. Conceptual/overview pages first so a "few pages per
# service" slice is the most representative content, not deep API reference.
_PREFERENCE = (
    ("/overview", 0),
    ("-overview", 0),
    ("/what-is", 1),
    ("/introduction", 1),
    ("/concepts", 2),
    ("/concept-", 2),
    ("/get-started", 3),
    ("/quickstart", 3),
    ("/how-to", 4),
    ("/tutorial", 5),
)


def _preference(url: str) -> int:
    for token, score in _PREFERENCE:
        if token in url:
            return score
    return 9


def select_urls(urls: list[str], per_service: int = 4, cap: int = 1000) -> list[str]:
    """Pick up to `per_service` pages from each Azure service, capped at `cap` total.

    Groups URLs by the `/azure/<service>/` segment, drops media/include noise, and
    within each service prefers overview/concept/quickstart pages. Services are
    taken round-robin so the cap spreads across many services (broad coverage)
    rather than exhausting on the few largest ones.
    """
    by_service: dict[str, list[str]] = collections.defaultdict(list)
    for url in urls:
        m = _SERVICE_RE.match(url)
        if not m or any(n in url for n in _NOISE):
            continue
        by_service[m.group(1)].append(url)

    ranked: dict[str, list[str]] = {}
    for service, group in by_service.items():
        ranked[service] = sorted(set(group), key=lambda u: (_preference(u), u))[:per_service]

    selected: list[str] = []
    for rank in range(per_service):
        for service in sorted(ranked):
            if rank < len(ranked[service]) and len(selected) < cap:
                selected.append(ranked[service][rank])
    return selected


def _fetch_text(url: str) -> str:  # pragma: no cover - network
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers=_UA), timeout=60
    ).read()
    try:
        return gzip.decompress(raw).decode("utf-8", "replace")
    except (OSError, gzip.BadGzipFile):
        return raw.decode("utf-8", "replace")


def crawl_azure_sitemaps(
    per_service: int = 4, cap: int = 1000
) -> list[str]:  # pragma: no cover - network
    """Fetch the Azure sitemap shards and return a selected corpus URL list."""
    index = _fetch_text(SITEMAP_INDEX)
    shards = _SHARD_RE.findall(index)
    print(f"Found {len(shards)} Azure sitemap shard(s).", flush=True)
    all_locs: list[str] = []
    for i, shard in enumerate(shards, 1):
        try:
            all_locs.extend(loc.strip() for loc in _LOC_RE.findall(_fetch_text(shard)))
            print(f"  shard {i}/{len(shards)}: {len(all_locs)} urls so far", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad shard must not abort
            print(f"  shard {i} failed: {exc}", flush=True)
    return select_urls(all_locs, per_service=per_service, cap=cap)
