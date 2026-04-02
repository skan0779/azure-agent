# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Create user
RUN useradd -m -u 1000 appuser

# Copy dependency files and package source
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md ./
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies and the local package
RUN uv sync --frozen --no-editable \
  && rm -rf /root/.cache/uv /root/.cache/pip /tmp/*

# Copy app env files
COPY environments/env ./environments/env

# Set ownership and drop privileges
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Default log level
ENV LOG_LEVEL=info

# Run Gunicorn via uv (Uvicorn worker)
CMD ["sh","-lc","uv run gunicorn azure_agent.api.main:app \
  -k uvicorn.workers.UvicornWorker \
  --access-logfile - \
  --error-logfile - \
  --log-level ${LOG_LEVEL:-info} \
  -w ${WEB_CONCURRENCY:-4} \
  -b 0.0.0.0:${PORT:-8080} \
  --timeout ${WORKER_TIMEOUT:-120} \
  --graceful-timeout ${GRACEFUL_TIMEOUT:-30} \
  --keep-alive ${KEEP_ALIVE:-75}"]
