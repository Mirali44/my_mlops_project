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

# Copy frontend code - use . because context is already root and dockerfile is in frontend/
COPY frontend/ .

# Install Python dependencies
RUN uv sync --no-dev

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Expose 8501 port
EXPOSE 8501

# Run streamlit application 
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"]