"""Generate synthetic test-item CANDIDATES with the Claude judge-family model.

Pages are named explicitly so gold URLs are provenance, not recovered (ADR-0010).
Prints screened candidate rows as JSON to stdout for manual review — nothing is
written to data/testset.jsonl by this script.

Usage:
    uv run python scripts/generate_synthetic_testset.py <url> [<url> ...] [--per-page N]
"""
import argparse
import json

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

from ragpipe.config import Settings
from ragpipe.eval.synthetic import make_candidates, page_text_from_index
from ragpipe.foundry_judge import build_judge_complete_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+", help="corpus page URLs to author questions from")
    parser.add_argument("--per-page", type=int, default=5)
    args = parser.parse_args()

    settings = Settings.from_env()
    search = SearchClient(
        settings.search_endpoint, settings.search_index, DefaultAzureCredential()
    )
    complete = build_judge_complete_fn(settings)

    rows = []
    for url in args.urls:
        document = page_text_from_index(search, url)
        if not document:
            print(f"-- no indexed chunks for {url}; skipped", flush=True)
            continue
        rows.extend(make_candidates(complete, url=url, document=document, n=args.per_page))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
