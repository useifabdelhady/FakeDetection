# ============================================================
# TruthLens ML Service — Dockerfile for Hugging Face Spaces
# ============================================================
# HF Spaces will auto-build this Dockerfile
# ============================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy ML source modules (phishing feature extraction)
COPY ml-source/ ./ml-source/

# Copy ML models (~900 MB total)
COPY ml-models/ ./ml-models/

# Copy the FastAPI application
COPY ml-service/app.py ./ml-service/app.py

# Copy environment file
COPY .env ./.env

# Fix ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "ml-service.app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
