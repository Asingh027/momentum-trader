FROM python:3.11-slim

# ── uv (fast Python package manager) ─────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ── supercronic (lightweight cron for containers) ─────────────────────────────
ENV SUPERCRONIC_VERSION=v0.2.33
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL \
       "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-amd64" \
       -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic

WORKDIR /app

# ── Dependencies (cached layer — only rebuilt when lock file changes) ──────────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Source code (separate layer — dep cache survives code-only changes) ────────
COPY src/     ./src/
COPY scripts/ ./scripts/
COPY crontab  /app/crontab
RUN uv sync --frozen --no-dev

# ── Runtime defaults (overridden by docker-compose environment block) ─────────
ENV TZ=America/New_York
ENV ALPACA_ENV_PATH=/data/alpaca.env
ENV DB_PATH=/data/trader.db
ENV REPORTS_DIR=/data/daily_reports

CMD ["supercronic", "/app/crontab"]
