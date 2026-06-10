"""Generate synthetic test-item CANDIDATES from the indexed corpus (spec §6).

Prints candidate rows as JSON to stdout for manual screening — nothing is
written to data/testset.jsonl by this script. Items whose source URL cannot be
recovered are dropped (URL-match metrics need a gold URL, ADR-0002).
"""
import json
import sys

from ragpipe.config import Settings
from ragpipe.eval.run import _sample_corpus_docs
from ragpipe.eval.testset import build_synthetic_generator


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    settings = Settings.from_env()
    docs = _sample_corpus_docs(settings)
    items = build_synthetic_generator(settings, docs, testset_size=n)()
    rows = []
    for it in items:
        probe = (it.ground_truth_context or "")[:200]
        url = next((d["url"] for d in docs if probe and probe in d["content"]), "")
        if not url or not it.ground_truth:
            continue
        rows.append(
            {
                "question": it.question,
                "ground_truth": it.ground_truth,
                "ground_truth_context": url,
                "tags": ["synthetic"],
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
