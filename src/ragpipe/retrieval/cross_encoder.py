"""Cross-encoder rerank on the DGX Spark (issue #40, ADR-0021).

Azure's semantic ranker went with the search service. Its replacement is
``nvidia/llama-nemotron-rerank-1b-v2`` on vLLM in homelab, behind vLLM's
``/v1/rerank``. The ``question:``/``passage:`` template is applied server side,
so documents go over as plain text.

Document text is built the way the old semantic configs did, so the swap is like
for like: title, then content, then the situating context for the indexes whose
config listed it (``ms_docs``, ``raptor_sac``). ``baseline`` had title + content.

If the reranker can't answer mid-query the mode is unavailable, full stop. No
quiet fallback to passthrough: the numbers would describe a different pipeline
under the same name.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, replace

import httpx

from ragpipe.config import RERANKER_OFFLINE, ModeUnavailable, RetrievalMode
from ragpipe.model_roles import (
    DEFAULT_RERANKER_MODEL,
    ModelRole,
    RoleEndpoint,
    endpoints_from_env,
)
from ragpipe.models import Chunk

# How long a /health answer stands before the next probe.
HEALTH_TTL_SECONDS = 60.0
# The probe runs inside /modes, so it gives up fast.
HEALTH_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class RerankerConfig:
    url: str
    api_key: str | None = None
    model: str = DEFAULT_RERANKER_MODEL

    @classmethod
    def from_endpoint(cls, endpoint: RoleEndpoint) -> "RerankerConfig | None":
        """The reranker role's endpoint. None when no URL serves it: Foundry has no
        cross-encoder, so there's nothing to fall back to."""
        if not endpoint.url:
            return None
        return cls(
            url=endpoint.url,
            api_key=endpoint.api_key,
            model=endpoint.model or DEFAULT_RERANKER_MODEL,
        )

    @classmethod
    def from_env(cls) -> "RerankerConfig | None":
        """The same, read straight from the env homelab passes the container."""
        endpoint = endpoints_from_env(os.environ).get(ModelRole.RERANKER)
        return cls.from_endpoint(endpoint) if endpoint else None

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def document_text(chunk: Chunk, *, with_context: bool) -> str:
    parts = [chunk.title, chunk.content]
    if with_context:
        parts.append(chunk.context)
    return "\n\n".join(p for p in parts if p)


class _Health:
    """Last ``/health`` answer per base URL, kept for ``HEALTH_TTL_SECONDS``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: dict[str, tuple[float, bool]] = {}

    def reachable(self, config: RerankerConfig, client: httpx.Client | None = None) -> bool:
        now = time.monotonic()
        with self._lock:
            hit = self._seen.get(config.url)
        if hit is not None and now - hit[0] < HEALTH_TTL_SECONDS:
            return hit[1]
        ok = _probe(config, client)
        with self._lock:
            self._seen[config.url] = (now, ok)
        return ok

    def mark_down(self, config: RerankerConfig) -> None:
        """A failed rerank call greys the modes out now, not at the next probe."""
        with self._lock:
            self._seen[config.url] = (time.monotonic(), False)

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()


def _probe(config: RerankerConfig, client: httpx.Client | None) -> bool:
    try:
        if client is None:
            with httpx.Client(timeout=HEALTH_TIMEOUT_SECONDS) as c:
                res = c.get(f"{config.url}/health", headers=config.headers())
        else:
            res = client.get(f"{config.url}/health", headers=config.headers())
        return res.status_code == 200
    except httpx.HTTPError:
        return False


health = _Health()


def reranker_reachable() -> bool:
    """True when ``RERANKER_URL`` is set and its ``/health`` answered 200 lately."""
    config = RerankerConfig.from_env()
    return config is not None and health.reachable(config)


class CrossEncoderReranker:
    """``rerank(query, fused, top_k)`` over vLLM's ``/v1/rerank``.

    ``timeout`` and ``max_retries`` for the same reason as the embeddings client:
    without a timeout a stalled call hangs the request, and a bounded retry rides
    out a blip without hiding an outage. Timeouts, connection errors, 429 and 5xx
    are retried; anything else fails straight away.
    """

    def __init__(
        self,
        config: RerankerConfig,
        mode: RetrievalMode,
        *,
        top_k: int = 5,
        with_context: bool = True,
        timeout: float = 15.0,
        max_retries: int = 2,
        backoff: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._mode = mode
        self._top_k = top_k
        self._with_context = with_context
        self._max_retries = max_retries
        self._backoff = backoff
        self._client = client or httpx.Client(timeout=timeout)

    def rerank(self, query: str, fused: list[Chunk], top_k: int | None = None) -> list[Chunk]:
        if not fused:
            return []
        k = min(top_k or self._top_k, len(fused))
        body = {
            "model": self._config.model,
            "query": query,
            "documents": [document_text(c, with_context=self._with_context) for c in fused],
            "top_n": k,
        }
        results = self._post(body)
        try:
            return [
                replace(fused[r["index"]], score=float(r["relevance_score"]))
                for r in results[:k]
            ]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise self._offline(f"unexpected response: {exc!r}") from exc

    def _post(self, body: dict) -> list[dict]:
        url = f"{self._config.url}/v1/rerank"
        last = ""
        for attempt in range(self._max_retries + 1):
            if attempt:
                time.sleep(self._backoff * 2 ** (attempt - 1))
            try:
                res = self._client.post(url, json=body, headers=self._config.headers())
            except httpx.TimeoutException as exc:
                last = f"timeout: {exc!r}"
                continue
            except httpx.TransportError as exc:
                last = f"transport: {exc!r}"
                continue
            if res.status_code == 429 or res.status_code >= 500:
                last = f"HTTP {res.status_code}"
                continue
            if res.status_code != 200:
                raise self._offline(f"HTTP {res.status_code}: {res.text[:200]}")
            try:
                return res.json()["results"]
            except (ValueError, KeyError) as exc:
                raise self._offline(f"unexpected response: {exc!r}") from exc
        raise self._offline(f"gave up after {self._max_retries + 1} attempts, last {last}")

    def _offline(self, why: str) -> ModeUnavailable:
        print(f"rerank ({self._mode.value}): {why}", file=sys.stderr, flush=True)
        health.mark_down(self._config)
        return ModeUnavailable(self._mode, RERANKER_OFFLINE)
