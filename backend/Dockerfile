FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed

# Install build dependencies and clean up in the same layer
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y clang git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend code - use . because context is already root and dockerfile is in backend/
COPY backend/ .

# Install Python dependencies
RUN uv sync --no-dev

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose 8000 port
EXPOSE 8000

# Run uvicorn server for fastapi application 
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]