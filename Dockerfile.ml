# ============================================================
# TruthLens ML Service — Dockerfile for Hugging Face Spaces
# ============================================================
# This Dockerfile is placed at the project ROOT because it needs
# access to: ml-service/, ml-source/, ml-models/, .env
#
# Build:  docker build -f Dockerfile.ml -t truthlens-ml .
# Run:    docker run -p 7860:7860 truthlens-ml
# ============================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── 1. Install Python dependencies (cached layer) ──
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── 2. Copy ML source modules (phishing feature extraction) ──
COPY ml-source/ ./ml-source/

# ── 3. Copy ML models (~900 MB — this layer takes the longest) ──
COPY ml-models/ ./ml-models/

# ── 4. Copy the FastAPI application ──
COPY ml-service/app.py ./ml-service/app.py

# ── 5. Copy environment file for API keys ──
COPY .env ./.env

# Fix ownership
RUN chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Health check (give 120s for model loading)
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start the ML service
# --timeout-keep-alive 120 keeps connections alive during slow model inference
CMD ["python", "-m", "uvicorn", "ml-service.app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
