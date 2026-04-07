FROM python:3.12-slim AS base

ARG GOOSE_VERSION=v1.29.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bzip2 libgomp1 && rm -rf /var/lib/apt/lists/*

# Install goose CLI for ACP subprocess integration
RUN curl -fsSL "https://github.com/block/goose/releases/download/${GOOSE_VERSION}/goose-x86_64-unknown-linux-gnu.tar.bz2" \
    | tar xj --strip-components=0 -C /usr/local/bin/ \
    && chmod +x /usr/local/bin/goose

RUN groupadd -r app && useradd -r -g app -d /app app
WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY . .
RUN mkdir -p /app/.cache && chown -R app:app /app/.cache

ENV HOME=/app
ENV XDG_CACHE_HOME=/app/.cache

RUN python -m playwright install-deps firefox
RUN python -m camoufox fetch && chown -R app:app /app/.cache

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "life_core.api:app", "--host", "0.0.0.0", "--port", "8000"]
