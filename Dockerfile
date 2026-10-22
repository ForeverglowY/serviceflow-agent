FROM ghcr.io/astral-sh/uv:0.11.24 AS uv
FROM python:3.13-slim-bookworm

COPY --from=uv /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY serviceflow ./serviceflow
COPY scripts ./scripts
COPY data ./data
COPY knowledge ./knowledge
COPY migrations ./migrations
COPY logging_config.json ./logging_config.json
RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /models && chown appuser:appuser /models
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "serviceflow.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging_config.json"]
