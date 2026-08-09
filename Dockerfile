FROM python:3.12-slim

WORKDIR /workspace

# Install system dependencies needed for compiling python dependencies (e.g. psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast system package installation
RUN pip install --no-cache-dir uv

# Copy project definition
COPY pyproject.toml /workspace/

# Synchronize python dependencies directly into the system site-packages
RUN uv pip install --system -r pyproject.toml

# Copy application files (directories will be mounted as volumes in docker-compose, but we copy them for standalone container health)
COPY app/ /workspace/app/
COPY feature_store/ /workspace/feature_store/

# Expose port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
