# syntax=docker/dockerfile:1

# Container image for the ragpipe HTTP API. Mirrors how the service is run
# locally (see README "Serve the HTTP API"):
#
#     uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
#
# Built and pushed to GHCR by .github/workflows/publish-image.yml.
FROM python:3.11-slim

# uv ships as a static binary in its own published image; copy it in rather
# than pip-installing it. Pinned to the 0.11 line we develop against.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Use the interpreter from python:3.11-slim instead of downloading one.
    UV_PYTHON_DOWNLOADS=0

# Resolve dependencies against the lockfile first so this layer is cached
# across source-only edits. --no-install-project: the project source (src/)
# isn't in the context yet.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Bring in the source and install the project itself (the `ragpipe` package
# lives under src/; `app` is imported from the working directory).
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Put the project venv on PATH so `uvicorn` resolves without `uv run`.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Bind all interfaces so the container is reachable. The service is meant to
# sit behind the website's Spring backend — do not expose it publicly unguarded.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
