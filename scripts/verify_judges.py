"""One-shot LIVE smoke for every text model role, wherever each one is served (run
BEFORE any ingest or eval run). Verifies: online judge gate scoring, the offline
judge, the ingest writer, the planner, a raw online judge completion. Exits non-zero
on failure."""
import asyncio
import sys

from ragpipe.config import Settings


def main() -> int:
    settings = Settings.from_env()
    failures = 0

    def check(name, fn):
        nonlocal failures
        try:
            result = fn()
            print(f"PASS {name}: {str(result)[:120]}")
        except Exception as exc:  # noqa: BLE001 - report every route
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")

    def gate():
        from ragpipe.faithfulness_gate import build_faithfulness_gate

        return asyncio.run(
            build_faithfulness_gate(settings).score(
                "What color is the sky?",
                "The sky is blue.",
                ["The sky is blue during the day."],
            )
        )

    def offline_judge():
        from ragpipe.eval.harness import _build_ragas_clients

        llm, _ = _build_ragas_clients(settings)
        return llm.langchain_llm.invoke("Reply with exactly: OK").content

    def completion(role):
        from ragpipe.model_roles import complete_fn

        return lambda: complete_fn(settings, role)("Reply with exactly: OK")

    from ragpipe.model_roles import ModelRole

    check("online judge gate (RAGAS faithfulness)", gate)
    check("offline judge", offline_judge)
    check("ingest writer completion", completion(ModelRole.INGEST_WRITER))
    check("planner completion", completion(ModelRole.PLANNER))
    check("online judge raw completion", completion(ModelRole.ONLINE_JUDGE))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
