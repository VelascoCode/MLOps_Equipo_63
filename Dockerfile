### Multi-stage Dockerfile for ml-service
### Builder stage: build wheels for all Python dependencies to improve caching
FROM python:3.12.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build deps required to build some Python wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       python3-dev \
       libffi-dev \
       libssl-dev \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip wheel --wheel-dir=/wheels -r requirements.txt


### Final stage: lightweight runtime image
FROM python:3.12.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheels from builder and install them (offline install for reproducibility)
COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application code
COPY . /app

# Ensure NLTK data needed at runtime is available in the image
RUN python -m nltk.downloader punkt stopwords averaged_perceptron_tagger punkt_tab || true

# Create a non-root user to run the app
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck: require curl to be present in image
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Run the service with a single worker; orchestrators should scale with replicas
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# Start from the official slim Python image for a balance of size and compatibility
FROM python:3.12.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal OS packages required to build some Python packages and run ML deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       git \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application sources
COPY . /app

# Ensure NLTK data needed at runtime is available in the image
RUN python -m nltk.downloader punkt stopwords averaged_perceptron_tagger punkt_tab || true

# Default port for FastAPI/uvicorn
EXPOSE 8000

# Production entrypoint: run uvicorn with one worker (change --workers for concurrency)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
