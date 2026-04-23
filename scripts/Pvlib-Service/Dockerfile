# ============================================================
# Pvlib-Service — multi-stage Dockerfile
# Target: ARM64 (t4g.small) and AMD64 (dev/CI)
# Final image ≈ 600–700 MB (dominated by pandas/numpy/pvlib wheels)
# ============================================================

# ── Stage 1: build wheels ─────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels on ARM
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Download/build all wheels into a local cache directory
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="pvlib-power-prediction-service"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="Real-time PV expected-generation service (H-A3 physics, H-B6 data strategy)"

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install pre-built wheels (no internet needed at runtime)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl \
 && rm -rf /wheels

# Copy application code
COPY app/ ./app/

# Create logs directory with correct ownership
RUN mkdir -p logs && chown appuser:appuser logs

USER appuser

# Uvicorn listens on all interfaces; Caddy/Nginx terminates TLS in front
EXPOSE 8000

# Health check (Docker / ECS will mark the container healthy after this passes)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
