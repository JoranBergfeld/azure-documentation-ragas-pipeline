"""One-shot LIVE smoke for every model route (run on a machine with Azure access
BEFORE any ingest or eval run). Verifies: online judge gate scoring, DeepSeek
offline judge, gpt decoration call, raw judge completion. Exits non-zero on failure."""
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
        from ragpipe.guardrail import build_ragas_faithfulness

        metric_fn = build_ragas_faithfulness(settings)
        return asyncio.run(
            metric_fn(
                question="What color is the sky?",
                answer="The sky is blue.",
                contexts=["The sky is blue during the day."],
            )
        )

    def offline_judge():
        from ragpipe.eval.harness import _build_ragas_clients

        llm, _ = _build_ragas_clients(settings)
        return llm.langchain_llm.invoke("Reply with exactly: OK").content

    def decoration():
        from ragpipe.context_gen import build_context_complete_fn

        return build_context_complete_fn(settings)("Reply with exactly: OK")

    def judge_raw():
        from ragpipe.foundry_judge import build_judge_complete_fn

        return build_judge_complete_fn(settings)("Reply with exactly: OK")

    check("online judge gate (RAGAS faithfulness)", gate)
    check("deepseek offline judge", offline_judge)
    check("gpt decoration completion", decoration)
    check("judge raw completion", judge_raw)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
