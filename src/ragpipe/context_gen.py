"""Per-chunk situating-context generation with deterministic controls (ADR-0001/0005).

One LLM call per (document, chunk) pair, content-address-cached so re-ingests
only pay for changed chunks. Failures fall back to "" — the caller then
decorates with the breadcrumb only, so ingest never blocks on the LLM.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Callable

# Bump to deliberately invalidate every cached context (ADR-0005).
PROMPT_VERSION = "v1"

# Anthropic's published situating prompt (Introducing Contextual Retrieval, 2024).
SITUATE_PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall \
document for the purposes of improving search retrieval of the chunk. Answer only \
with the succinct context and nothing else."""


class ContextGenerator:
    """Wraps a `complete(prompt) -> str` callable with cache, retries, fallback."""

    def __init__(
        self,
        complete_fn: Callable[[str], str],
        cache_path: str | Path = ".context_cache.json",
        max_retries: int = 2,
    ) -> None:
        self._complete = complete_fn
        self._cache_path = Path(cache_path)
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._cache = self._load_cache()
        self.fallback_count = 0

    def _load_cache(self) -> dict[str, str]:
        try:
            return json.loads(self._cache_path.read_text())
        except (OSError, ValueError):
            return {}  # missing or corrupt -> empty, never an error (ADR-0005)

    def _save_cache(self) -> None:
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache))
        tmp.replace(self._cache_path)

    @staticmethod
    def _key(document: str, chunk: str) -> str:
        payload = "\x00".join((PROMPT_VERSION, document, chunk))
        return hashlib.sha256(payload.encode()).hexdigest()

    def generate(self, document: str, chunk: str) -> str:
        """Situating context for `chunk`, or "" after retries are exhausted."""
        key = self._key(document, chunk)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        prompt = SITUATE_PROMPT.format(document=document, chunk=chunk)
        for _ in range(self._max_retries):
            try:
                context = self._complete(prompt).strip()
            except Exception:  # noqa: BLE001 - one bad chunk must not abort ingest
                continue
            with self._lock:
                self._cache[key] = context
                self._save_cache()
            return context
        with self._lock:
            self.fallback_count += 1
        return ""


def build_context_complete_fn(
    settings, api_version: str = "2024-10-21", timeout: float = 60.0, max_retries: int = 5
) -> Callable[[str], str]:  # pragma: no cover - live Azure call
    """`complete(prompt) -> str` over the account's /openai endpoint, temperature=0."""
    from ragpipe.embeddings import _build_client

    client = _build_client(settings, api_version, timeout, max_retries)

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=settings.foundry_chat_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return complete
