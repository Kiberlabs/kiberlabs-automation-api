# Multi-stage production Dockerfile for KiberLabs B2B Automation API
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim AS runtime

RUN groupadd --system appuser && useradd --system --gid appuser --create-home appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src ./src

RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

ENV KIBERLABS_DB_PATH=/app/data/gateway.db \
    KIBERLABS_RATE_LIMIT_RPM=60 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "kiberlabs_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
