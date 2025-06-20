# syntax=docker/dockerfile:1.4
# Stage 1: Build dependencies and application
FROM python:3.12.4-slim-bookworm AS builder

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv using the official installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create application directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy source code and install the application
COPY src/ ./src/
RUN uv pip install --no-deps .

# Stage 2: Production runtime image
FROM python:3.12.4-slim-bookworm AS runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv in runtime for `uv run` commands
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set up application directory with proper permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy project configuration file (needed for project root detection)
COPY --chown=appuser:appuser pyproject.toml ./

# Copy README.md file (required by pyproject.toml)
COPY --chown=appuser:appuser README.md ./

# Copy the entire source directory
COPY --chown=appuser:appuser src/ ./src/

# Copy application assets

# COPY --chown=appuser:appuser scripts/ ./scripts/

# Exclude data/ as we are not going to use training in AWS ECS due to expensiveness
# COPY --chown=appuser:appuser data/ ./data/

COPY --chown=appuser:appuser assets/ ./assets/

# Create directories for persistent data with proper permissions
RUN mkdir -p /app/mlruns /app/mlartifacts /app/experiments && \
    chown -R appuser:appuser /app/mlruns /app/mlartifacts /app/experiments

# Switch to non-root user
USER appuser

# Set up environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Expose ports
EXPOSE 8000 8501 8080 6006

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1
